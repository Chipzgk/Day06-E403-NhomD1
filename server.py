from __future__ import annotations

import asyncio
import json
import logging
import re
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("analogy_api")

# ── Feedback log ──────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"

def append_feedback(record: dict) -> None:
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("feedback logged: action=%r concept=%r", record.get("action"), record.get("concept"))

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

LLM_TIMEOUT_SECONDS = 30

# ── LLM singleton ─────────────────────────────────────────────────────────────
_llm: ChatOpenAI | None = None

def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("MODEL", "gpt-4o"),
            temperature=0.0,
            base_url=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("API_KEY"),
        )
        logger.info("LLM instance created: model=%s", os.getenv("MODEL"))
    return _llm


# ── Agent State ───────────────────────────────────────────────────────────────
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


def make_initial_state(concept: str, interest: str) -> AgentState:
    return {
        "concept": concept,
        "interest": interest,
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


# ── Helper ────────────────────────────────────────────────────────────────────
def _parse_json(raw: str, context: str = "") -> dict:
    text = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        logger.error("No JSON found in LLM output [%s]: %r", context, raw[:200])
        raise ValueError(f"LLM không trả về JSON hợp lệ [{context}]")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        logger.error("JSON parse error [%s]: %s | raw: %r", context, e, raw[:200])
        raise ValueError(f"JSON parse lỗi [{context}]: {e}") from e


# ── Node 1: Kiểm tra input ────────────────────────────────────────────────────
VAGUE_INTERESTS = {
    "thể thao", "the thao", "âm nhạc", "am nhac",
    "nghệ thuật", "nghe thuat", "sport", "sports",
    "music", "game", "games", "phim", "film",
    "giải trí",
}

def check_input(state: AgentState) -> AgentState:
    interest_normalized = state["interest"].strip().lower()
    is_vague = interest_normalized in VAGUE_INTERESTS
    logger.info(
        "check_input: concept=%r interest=%r is_vague=%s",
        state["concept"], state["interest"], is_vague,
    )
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
    llm = get_llm()
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
    logger.info("evaluate_confidence: concept=%r", state["concept"])
    raw = llm.invoke(prompt).content
    data = _parse_json(raw, context="evaluate_confidence")
    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    reason = data.get("reason", "")
    is_fallback = confidence < 0.6
    logger.info("evaluate_confidence: confidence=%.2f is_fallback=%s", confidence, is_fallback)
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
    llm = get_llm()
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
    logger.info("generate_analogy: concept=%r interest=%r", state["concept"], interest)
    raw = llm.invoke(prompt).content
    data = _parse_json(raw, context="generate_analogy")
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


# ── Graph — compiled once at startup ──────────────────────────────────────────
def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("check_input",         check_input)
    builder.add_node("ask_clarify",         lambda s: s)
    builder.add_node("evaluate_confidence", evaluate_confidence)
    builder.add_node("generate_analogy",    generate_analogy)
    builder.add_node("fallback",            lambda s: s)
    builder.set_entry_point("check_input")
    builder.add_conditional_edges("check_input",         route_after_check)
    builder.add_conditional_edges("evaluate_confidence", route_after_evaluate)
    builder.add_edge("generate_analogy", END)
    builder.add_edge("ask_clarify",      END)
    builder.add_edge("fallback",         END)
    return builder.compile()

graph = _build_graph()
logger.info("LangGraph compiled successfully")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Máy Dịch Khái Niệm API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas: Analogy ──────────────────────────────────────────────────────────
class AnalogyRequest(BaseModel):
    concept: str
    interest: str

    @field_validator("concept", "interest")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Không được để trống")
        return v.strip()


# ── Schemas: Feedback ─────────────────────────────────────────────────────────
VALID_ACTIONS = {"accept", "regenerate", "fix_narrative", "missing_mapping"}

class FeedbackRequest(BaseModel):
    action: str
    concept: str
    interest: str
    confidence: float
    narrative: str
    mapping_table: list[dict]
    comment: str = ""

    @field_validator("action")
    @classmethod
    def valid_action(cls, v: str) -> str:
        if v not in VALID_ACTIONS:
            raise ValueError(f"action phải là một trong: {sorted(VALID_ACTIONS)}")
        return v

    @field_validator("comment")
    @classmethod
    def comment_required_for_detail_actions(cls, v: str, info) -> str:
        action = info.data.get("action", "")
        if action in {"fix_narrative", "missing_mapping"} and not v.strip():
            raise ValueError(f"comment không được trống khi action='{action}'")
        return v.strip()


# ── Schemas: Chat ─────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        if v not in {"user", "assistant"}:
            raise ValueError("role phải là 'user' hoặc 'assistant'")
        return v

    @field_validator("content")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content không được để trống")
        return v.strip()


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    context: str = ""   # nội dung slide hiện tại, để trống nếu không có

    @field_validator("messages")
    @classmethod
    def not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("messages không được rỗng")
        if v[-1].role != "user":
            raise ValueError("Tin nhắn cuối phải là của user")
        return v


# ── Endpoints: Analogy ────────────────────────────────────────────────────────
@app.post("/api/analogy")
async def create_analogy(req: AnalogyRequest):
    logger.info("POST /api/analogy concept=%r interest=%r", req.concept, req.interest)
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(graph.invoke, make_initial_state(req.concept, req.interest)),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("LLM timeout after %ds for concept=%r", LLM_TIMEOUT_SECONDS, req.concept)
        raise HTTPException(status_code=504, detail="LLM xử lý quá lâu, vui lòng thử lại.")
    except ValueError as e:
        logger.error("Parse error: %s", e)
        raise HTTPException(status_code=502, detail=f"Lỗi xử lý phản hồi từ LLM: {e}")
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {e}")

    return {
        "concept":                result["concept"],
        "interest":               result["interest"],
        "needs_clarification":    result["needs_clarification"],
        "clarification_question": result["clarification_question"],
        "is_fallback":            result["is_fallback"],
        "fallback_reason":        result["fallback_reason"],
        "suggestions":            result["suggestions"],
        "confidence":             result["confidence"],
        "narrative":              result["narrative"],
        "mapping_table":          result["mapping_table"],
    }


# ── Endpoints: Feedback ───────────────────────────────────────────────────────
@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Ghi feedback từ 4 nút vào logs/feedback.jsonl.

    Schema mỗi record:
    {
      "timestamp":     "2026-06-04T10:00:00+00:00",
      "action":        "accept" | "regenerate" | "fix_narrative" | "missing_mapping",
      "concept":       "RAG",
      "interest":      "bóng đá",
      "confidence":    0.9,
      "comment":       "Narrative chưa đề cập retrieval step",
      "narrative":     "...",
      "mapping_table": [...]
    }
    """
    record = {
        "action":        req.action,
        "concept":       req.concept,
        "interest":      req.interest,
        "confidence":    req.confidence,
        "comment":       req.comment,
        "narrative":     req.narrative,
        "mapping_table": req.mapping_table,
    }
    try:
        await asyncio.to_thread(append_feedback, record)
    except Exception as e:
        logger.exception("Failed to write feedback log")
        raise HTTPException(status_code=500, detail=f"Không ghi được log: {e}")

    return {"status": "ok", "action": req.action}


# ── Endpoints: Chat ───────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Multi-turn chat giới hạn chủ đề AI/khóa học.
    Client giữ toàn bộ lịch sử và gửi lên mỗi lượt.
    Truyền thêm context (nội dung slide hiện tại) để AI bám sát hơn.

    Request:
    {
      "context": "Nội dung slide hiện tại...",   // tuỳ chọn
      "messages": [
        {"role": "user",      "content": "RAG là gì?"},
        {"role": "assistant", "content": "RAG là..."},
        {"role": "user",      "content": "Cho ví dụ thực tế đi"}
      ]
    }

    Response:
    {
      "role": "assistant",
      "content": "..."
    }
    """
    llm = get_llm()

    # Ghép context slide vào system prompt nếu có
    context_block = ""
    if req.context.strip():
        context_block = f"""
NỘI DUNG SLIDE HIỆN TẠI (ưu tiên bám vào đây khi trả lời):
---
{req.context.strip()}
---
"""

    system_prompt = f"""Bạn là trợ lý học tập của khóa học AI.20K - Cohort 2.
{context_block}
CHỦ ĐỀ ĐƯỢC PHÉP:
- Các khái niệm AI/ML: LLM, RAG, Agent, LangChain, LangGraph, Embedding, Prompt Engineering...
- Nội dung các ngày học trong khóa (Day 1–6)
- Câu hỏi về bài tập, hackathon, prototype
- Giải thích thuật ngữ kỹ thuật liên quan AI/software

KHI BỊ HỎI NGOÀI CHỦ ĐỀ:
Từ chối lịch sự, nhắc lại phạm vi, gợi ý câu hỏi liên quan AI mà user có thể hỏi thay thế.

PHONG CÁCH:
- Thân thiện, ngắn gọn, dùng ví dụ thực tế
- Trả lời tiếng Việt trừ khi user hỏi tiếng Anh
- Không quá 200 chữ mỗi lượt trừ khi được yêu cầu giải thích chi tiết"""

    lc_messages = [("system", system_prompt)]
    for m in req.messages:
        lc_messages.append((m.role, m.content))

    logger.info(
        "POST /api/chat turns=%d last_msg=%r context_len=%d",
        len(req.messages),
        req.messages[-1].content[:80],
        len(req.context),
    )

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(llm.invoke, lc_messages),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="LLM xử lý quá lâu, thử lại nhé.")
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {e}")

    reply = response.content.strip()
    logger.info("chat reply: %r", reply[:80])

    return {"role": "assistant", "content": reply}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("MODEL")}


# ── Static files — mount sau cùng ─────────────────────────────────────────────
app.mount("/public", StaticFiles(directory="public"), name="public")
app.mount("/", StaticFiles(directory="static", html=True), name="static")