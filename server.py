from __future__ import annotations
import json
import re
import os
from typing import TypedDict, Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Máy Dịch Khái Niệm API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Course domain context ─────────────────────────────────────────────────────
COURSE_CONCEPTS = """
LangGraph, LangChain, LlamaIndex, AutoGen, CrewAI,
RAG (Retrieval-Augmented Generation), ReAct, Chain-of-Thought,
Prompt Engineering, Tool Use, Function Calling, Agent, Multi-agent,
Embedding, Vector DB, Context Window, Fine-tuning, RLHF,
Transformer, Attention, Token, LLM, GPT, Claude, Gemini,
FastAPI, Streamlit, LangServe, LangSmith, MCP (Model Context Protocol),
Semantic Search, Chunking, Reranking, Hallucination, Grounding,
System Prompt, Few-shot, Zero-shot, Temperature, Top-p,
"""

# ── LLM ───────────────────────────────────────────────────────────────────────
def build_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL", "gpt-4o"),
        temperature=0.0,
        base_url=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("API_KEY"),
    )


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    concept: str
    interest: str
    clarified_interest: str
    confidence: float
    narrative: str
    mapping_table: list[dict]
    is_fallback: bool
    fallback_reason: str
    suggestions: list[str]
    needs_clarification: bool
    clarification_question: str


# ── Helper ────────────────────────────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    text = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        return {}
    return json.loads(text[start:end + 1])


# ── Node 1: Kiểm tra input ────────────────────────────────────────────────────
def check_input(state: AgentState) -> AgentState:
    vague = {"thể thao", "âm nhạc", "nghệ thuật", "sport", "music", "game", "phim"}
    interest = state["interest"].strip().lower()
    is_vague = interest in vague
    return {
        **state,
        "needs_clarification": is_vague,
        "clarification_question": (
            f"Bạn thích '{state['interest']}' — "
            f"bạn có thể nói cụ thể hơn không? "
            f"(ví dụ: môn/thể loại/trò chơi cụ thể)"
        ) if is_vague else "",
        "clarified_interest": "" if is_vague else state["interest"],
    }


# ── Node 2: Đánh giá confidence ───────────────────────────────────────────────
def evaluate_confidence(state: AgentState) -> AgentState:
    llm = build_llm()
    interest = state.get("clarified_interest") or state["interest"]
    prompt = f"""Bạn là chuyên gia AI education. Đánh giá khả năng tạo analogy chất lượng.

Đây là danh sách thuật ngữ trong khóa học này (không phải toàn bộ):
{COURSE_CONCEPTS}

Khái niệm cần đánh giá: "{state['concept']}"
Sở thích user: "{interest}"

Quy tắc:
- Khái niệm có trong danh sách trên → confidence = 0.9
- Thuật ngữ AI/ML/software phổ biến ngoài danh sách → confidence >= 0.8
- Chỉ trả confidence < 0.6 khi khái niệm THỰC SỰ vô nghĩa hoặc không liên quan công nghệ

Trả về JSON duy nhất, không giải thích thêm:
{{
  "confidence": <số thực 0.0–1.0>,
  "reason": "<để trống nếu confidence >= 0.8>"
}}"""
    print(f"[DEBUG evaluate] concept={state['concept']!r}")
    raw = llm.invoke(prompt).content
    print(f"[DEBUG evaluate] raw={raw}")
    data = _parse_json(raw)
    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    reason = data.get("reason", "")
    is_fallback = confidence < 0.6
    suggestions = (
        [
            f"Nhập khái niệm chính xác hơn, ví dụ: 'Neural Network' thay vì '{state['concept']}'",
            f"Thử sở thích cụ thể hơn, ví dụ: 'bóng đá' thay vì '{interest}'",
            "Thử khái niệm AI phổ biến: Supervised Learning, Transformer, Embedding...",
        ]
        if is_fallback else []
    )
    return {
        **state,
        "confidence": confidence,
        "is_fallback": is_fallback,
        "fallback_reason": reason,
        "suggestions": suggestions,
    }


# ── Node 3: Sinh analogy ──────────────────────────────────────────────────────
def generate_analogy(state: AgentState) -> AgentState:
    llm = build_llm()
    interest = state.get("clarified_interest") or state["interest"]
    prompt = f"""Bạn là chuyên gia giải thích AI/ML bằng ngôn ngữ đời thường.

Khái niệm kỹ thuật AI: "{state['concept']}"
Lưu ý: Đây là thuật ngữ AI/ML chuyên ngành. Ví dụ RAG = Retrieval-Augmented Generation.
Sở thích của user: "{interest}"

Yêu cầu:
1. Viết narrative (~120 chữ tiếng Việt): dùng "{interest}" kể câu chuyện giải thích "{state['concept']}".
   Phải có mở đầu → diễn biến → kết luận rõ ràng.
2. Mapping: liệt kê từng thành phần kỹ thuật của "{state['concept']}" ứng với gì trong "{interest}".

Trả về JSON duy nhất, không giải thích thêm:
{{
  "narrative": "<câu chuyện tiếng Việt>",
  "mapping": [
    {{"technical_component": "...", "analogy_element": "..."}},
    ...
  ]
}}"""
    data = _parse_json(llm.invoke(prompt).content)
    return {
        **state,
        "narrative": data.get("narrative", ""),
        "mapping_table": data.get("mapping", []),
    }


# ── Routers ───────────────────────────────────────────────────────────────────
def route_after_check(state: AgentState) -> Literal["ask_clarify", "evaluate_confidence"]:
    return "ask_clarify" if state["needs_clarification"] else "evaluate_confidence"


def route_after_evaluate(state: AgentState) -> Literal["generate_analogy", "fallback"]:
    return "fallback" if state["is_fallback"] else "generate_analogy"


# ── Build graph ───────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("check_input", check_input)
    builder.add_node("ask_clarify", lambda s: s)
    builder.add_node("evaluate_confidence", evaluate_confidence)
    builder.add_node("generate_analogy", generate_analogy)
    builder.add_node("fallback", lambda s: s)
    builder.set_entry_point("check_input")
    builder.add_conditional_edges("check_input", route_after_check)
    builder.add_conditional_edges("evaluate_confidence", route_after_evaluate)
    builder.add_edge("generate_analogy", END)
    builder.add_edge("ask_clarify", END)
    builder.add_edge("fallback", END)
    return builder.compile()


# ── API Schema ────────────────────────────────────────────────────────────────
class AnalogyRequest(BaseModel):
    concept: str
    interest: str


INITIAL_STATE: AgentState = {
    "concept": "",
    "interest": "",
    "clarified_interest": "",
    "confidence": 0.0,
    "narrative": "",
    "mapping_table": [],
    "is_fallback": False,
    "fallback_reason": "",
    "suggestions": [],
    "needs_clarification": False,
    "clarification_question": "",
}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/api/analogy")
async def create_analogy(req: AnalogyRequest):
    graph = build_graph()
    result = graph.invoke({
        **INITIAL_STATE,
        "concept": req.concept,
        "interest": req.interest,
    })
    return {
        "concept": result["concept"],
        "interest": result["interest"],
        "needs_clarification": result["needs_clarification"],
        "clarification_question": result["clarification_question"],
        "is_fallback": result["is_fallback"],
        "fallback_reason": result["fallback_reason"],
        "suggestions": result["suggestions"],
        "confidence": result["confidence"],
        "narrative": result["narrative"],
        "mapping_table": result["mapping_table"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# Serve index.html at root
app.mount("/public", StaticFiles(directory="public"), name="public")

app.mount("/", StaticFiles(directory="static", html=True), name="static")