# AI.20K Hackathon Day 06 - Project Context

**Ngày tạo:** 04/06/2026  
**Dự án:** AI Product Hackathon - Batch 02 · Day 06  
**Deadline:** 23:59 ngày 04/06/2026

---

## 📋 Tổng quan dự án

### Tên sản phẩm
**"Thư ký Kim"** — Máy dịch khái niệm AI sang ngôn ngữ đời thường qua analogy

### Mục tiêu
Giúp người học hiểu các khái niệm AI/ML phức tạp bằng cách giải thích chúng thông qua những ví dụ/analogy liên quan đến sở thích cá nhân của họ.

**Ví dụ quy trình:**
- User chọn khái niệm: "RAG" (Retrieval-Augmented Generation)
- User nhập sở thích: "bóng đá"
- AI tạo: 
  - Analogy narrative: "RAG như một cầu thủ bóng đá..."
  - Bảng mapping: Thành phần kỹ thuật ↔ Yếu tố trong bóng đá

---

## 🗂️ Cấu trúc Thư mục

```
hackathon1/
├── README.md                          # Hướng dẫn chung hackathon
├── hackathon-rules.md                 # Luật chơi, cách chấm điểm, schedule demo
├── pyproject.toml                     # Cấu hình Python dependencies
├── main.py                            # Entry point CLI (chạy local)
├── server.py                          # FastAPI backend + LangGraph
├── .env.example                       # Ví dụ biến môi trường
├── .env                               # Biến môi trường thực (không commit)
├── uv.lock                            # Lock file cho dependencies
├── codebase/
│   └── README.md                      # Hướng dẫn nộp mã nguồn
├── spec/
│   └── README.md                      # Hướng dẫn viết SPEC sản phẩm
├── static/
│   ├── index.html                     # Frontend giao diện chính (PDF.js)
│   └── index-old.html                 # Phiên bản cũ (không dùng)
└── public/
    ├── 1-AICB_Ngay_1.pdf              # Tài liệu PDF ngày 1
    └── day03-tu-chatbot-den-agentic-agent-react-v7.pdf
```

---

## 🔧 Tech Stack

### Backend
- **Framework:** FastAPI (Python)
- **AI Orchestration:** LangGraph
- **LLM Support:**
  - OpenAI (GPT-4o mặc định)
  - Google Gemini
  - Ollama (local models)
- **Config:** python-dotenv

### Frontend
- **Framework:** HTML5 + CSS3 + Vanilla JavaScript (không framework)
- **PDF Viewer:** PDF.js (v3.11.174) - hỗ trợ bôi đen text trong PDF
- **UI Library:** Tabler Icons (CDN)
- **Features:**
  - Sidebar navigation (chọn bài học)
  - PDF viewer với text layer overlay
  - Popup modal cho nhập concept + interest
  - Real-time history (localStorage)
  - Showing work (progress animation)

### Deployment
- **Dev Server:** Uvicorn (port 8000)
- **Tunneling:** ngrok (để share URL public)
- **Static Files:** Mounted ở `/` và `/public`

---

## 📝 Dependencies

### Python
```
langchain>=1.0.0
langgraph>=0.6.0
langchain-core>=0.3.0
langchain-google-genai>=2.1.0
langchain-ollama>=0.3.0
pydantic>=2.8.0
python-dotenv>=1.0.1
langchain-openai>=1.2.2
fastapi
uvicorn
```

### Environment Variables
```
GOOGLE_API_KEY=                    # Google API key cho Gemini
LLM_MODEL=gemini-2.5-flash        # Model LLM mặc định
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:0.8b
```

---

## 🔄 Luồng xử lý (LangGraph)

### State Definition
```python
class AgentState(TypedDict):
    concept: str                       # Khái niệm AI user nhập
    interest: str                      # Sở thích cá nhân
    clarified_interest: str           # Sở thích sau khi làm rõ
    confidence: float                 # Độ tin cậy của analogy (0.0-1.0)
    narrative: str                    # Câu chuyện giải thích
    mapping_table: list[dict]         # Bảng mapping tech ↔ analogy
    is_fallback: bool                 # Có phải fallback case?
    fallback_reason: str              # Lý do fallback
    suggestions: list[str]            # Gợi ý sửa (nếu fallback)
    needs_clarification: bool         # Cần hỏi lại sở thích?
    clarification_question: str       # Câu hỏi làm rõ
```

### Nodes

| Node | Mô tả | Input | Output |
|------|-------|-------|--------|
| `check_input` | Kiểm tra sở thích có mơ hồ không (thể thao, âm nhạc, game, phim) | `concept`, `interest` | `needs_clarification`, `clarification_question` |
| `ask_clarify` | Nếu mơ hồ, dừng và hỏi lại | (from check_input) | END |
| `evaluate_confidence` | Gọi LLM đánh giá độ tin cậy của khái niệm | `concept` | `confidence`, `is_fallback`, `suggestions` |
| `generate_analogy` | Sinh narrative + mapping table | `concept`, `clarified_interest` | `narrative`, `mapping_table` |
| `fallback` | Trả về lỗi và gợi ý (khi confidence < 0.6) | (from evaluate_confidence) | END |

### Routing Logic

```
check_input
  ├─ needs_clarification = True → ask_clarify → END
  └─ needs_clarification = False → evaluate_confidence
                                     ├─ is_fallback = True → fallback → END
                                     └─ is_fallback = False → generate_analogy → END
```

---

## 🌐 API Endpoints

### `POST /api/analogy`
**Yêu cầu:**
```json
{
  "concept": "RAG",
  "interest": "bóng đá"
}
```

**Phản hồi (Success):**
```json
{
  "concept": "RAG",
  "interest": "bóng đá",
  "needs_clarification": false,
  "is_fallback": false,
  "confidence": 0.92,
  "narrative": "RAG như một cầu thủ bóng đá...",
  "mapping_table": [
    {
      "technical_component": "Retrieval",
      "analogy_element": "Nhặt bóng"
    }
  ]
}
```

**Phản hồi (Clarification needed):**
```json
{
  "needs_clarification": true,
  "clarification_question": "Bạn thích 'thể thao' — bạn có thể nói cụ thể hơn không?"
}
```

**Phản hồi (Fallback - low confidence):**
```json
{
  "is_fallback": true,
  "confidence": 0.45,
  "fallback_reason": "Khái niệm không rõ hoặc sở thích quá mơ hồ",
  "suggestions": ["Nhập khái niệm chính xác hơn...", "Thử sở thích cụ thể hơn..."]
}
```

### `GET /health`
Simple health check endpoint

### Static Files
- `/` → Serve `static/index.html`
- `/public/*` → Serve PDF files từ thư mục `public/`

---

## 💻 Frontend Architecture

### HTML Structure
```
<div class="app">
  <aside class="sidebar">
    - Header (course title, progress)
    - Tabs (Path, Learners, Discuss)
    - Nav items (PDF files)
  </aside>
  
  <main class="main">
    - Topbar (navigation, zoom)
    - Timeline (progress bar)
    - Content (PDF viewer - PDF.js)
    - Ask bar (hint + button)
  </main>
  
  <div class="overlay">
    <div class="popup">
      - Input view (concept, interest, plan)
      - Result view (narrative, mapping, actions)
    </div>
  </div>
</div>
```

### Key JavaScript Features

#### PDF Rendering (PDF.js)
- Load PDF từ URL
- Render từng trang thành canvas
- Overlay text layer để cho phép bôi đen
- Support multi-page PDF

#### Text Selection
```javascript
document.addEventListener('mouseup', () => {
  const text = window.getSelection().toString().trim();
  if (text.length > 1 && text.length < 80) {
    selectedConcept = text;
    // Pre-fill concept input
  }
});
```

#### History Management
- Lưu lịch sử tìm kiếm vào localStorage
- Key: `thu_ky_kim_history`
- Max 50 entries
- Load entry từ history → show result luôn

#### Showing Work Animation
- Cycle qua 5 bước công việc (mỗi 900ms)
- Hiện spinner khi bước active
- Mark checkmark khi bước done

#### Timeout Handling
- 25 giây timeout cho mỗi request
- Abort request nếu hết thời gian
- Show error message + retry button

---

## 🎮 User Flows

### Happy Path
```
1. User mở app
2. Click vào nav item (chọn bài PDF)
3. PDF load và hiện lên
4. User bôi đen một từ (concept)
5. Ask hint cập nhật → "Khái niệm đã chọn: [word]"
6. User click "Thư ký Kim" button
7. Modal mở → concept được pre-fill
8. User nhập sở thích
9. Click "Gửi"
10. API call → nhận analogy narrative + mapping
11. Result hiện trong modal
12. User có thể edit, accept, reject, hoặc report
```

### Clarification Flow
```
1. User nhập sở thích mơ hồ (thể thao, âm nhạc, etc.)
2. check_input node phát hiện → return clarification_question
3. Frontend show modal hỏi lại
4. User nhập cụ thể hơn
5. Re-trigger call với interest mới
```

### Fallback Flow
```
1. Khái niệm không rõ (confidence < 0.6)
2. evaluate_confidence node return is_fallback=true
3. Frontend show error + gợi ý
4. User có thể "Nhập lại khái niệm" hoặc "Đổi sở thích"
```

---

## 🚀 Chạy Dự Án

### Dev Mode (Local)

#### 1. Setup Environment
```bash
# Copy .env.example → .env
cp .env.example .env

# Edit .env và điền API keys (nếu dùng OpenAI/Google)
```

#### 2. Install Dependencies
```bash
# Dùng uv (nhanh hơn pip)
uv pip install -e .

# Hoặc dùng pip thường
pip install -e .
```

#### 3. Run Backend
```bash
# Terminal 1: Start server
uvicorn server:app --reload --port 8000
```

Backend sẽ chạy ở `http://localhost:8000`

#### 4. Run Frontend
```bash
# Terminal 2: Open browser
# Frontend là static HTML, access qua:
http://localhost:8000
```

#### 5. (Optional) Expose Public URL
```bash
# Terminal 3: Tunnel qua ngrok
ngrok http 8000
# Sẽ có URL public dạng: https://xxxx-xxxx-xxxx.ngrok.io
```

### CLI Mode
```bash
python main.py
# → Prompt: Nhập khái niệm AI cần hiểu
# → Prompt: Nhập sở thích cá nhân của bạn
# → In ra narrative + mapping table + confidence
```

---

## 🎯 Luật Chơi Hackathon

### Timeline (04/06/2026)
| Giờ | Mốc | Yêu cầu |
|-----|-----|---------|
| 9:00 | Build | Bắt đầu |
| **11:00** | Checkpoint 1 | **Mockup/prototype chạy được** |
| **13:00** | Checkpoint 2 | **AI lắp vào ≥ 1 flow** |
| **15:30** | Checkpoint 3 | **Tài liệu demo + slide ready** |
| **16:00** | Demo round | **10 phút/nhóm (5 trình bày + 5 Q&A)** |
| **17:00+** | Tổng kết | Top zone present cả lớp |

### Cách Chấm (4 tiêu chí, mỗi 1–5 điểm)
1. **Demo quality** — Chạy mượt, Q&A sắc bén
2. **Presentation** — Cuốn hút, mạch lạc
3. **Problem–solution fit** — Insight sắc, muốn dùng ngay
4. **AI product thinking** — AI là lợi thế cốt lõi

### Điểm Tổng (Day 5 + Day 6 = 100)
- SPEC: 25 điểm
- Prototype: 15 điểm
- Demo Day: 25 điểm
- Bài tập UX (Day 5): 10 điểm
- Phản ánh cá nhân: 25 điểm

### Điều kiện chặn
- Prototype không có lời gọi AI thật → max 4/10
- Không commit → mất điểm cá nhân
- Không giải thích được phần mình → 0 điểm demo cá nhân

---

## 📚 Course Concepts (Domain Context)

Danh sách khái niệm AI mà hệ thống hỗ trợ tốt:

```
LangGraph, LangChain, LlamaIndex, AutoGen, CrewAI,
RAG, ReAct, Chain-of-Thought,
Prompt Engineering, Tool Use, Function Calling,
Agent, Multi-agent,
Embedding, Vector DB, Context Window,
Fine-tuning, RLHF,
Transformer, Attention, Token,
LLM, GPT, Claude, Gemini,
FastAPI, Streamlit, LangServe, LangSmith, MCP,
Semantic Search, Chunking, Reranking,
Hallucination, Grounding,
System Prompt, Few-shot, Zero-shot,
Temperature, Top-p
```

Confidence scoring:
- Khái niệm có trong list → confidence = 0.9
- Thuật ngữ AI/ML phổ biến ngoài list → confidence >= 0.8
- Khái niệm không rõ/không liên quan tech → confidence < 0.6 (fallback)

---

## 🔍 Request Phases

### 1. Input Validation (`check_input`)
- Kiểm tra interest có mơ hồ?
- Set: `vague = {"thể thao", "âm nhạc", "nghệ thuật", "sport", "music", "game", "phim"}`
- Nếu mơ hồ → `needs_clarification = true` + END
- Nếu rõ → route tới `evaluate_confidence`

### 2. Confidence Evaluation (`evaluate_confidence`)
- LLM đánh giá: khái niệm có trong domain AI không?
- Trả về JSON: `{"confidence": <0.0-1.0>, "reason": "..."}`
- Nếu confidence < 0.6 → `is_fallback = true` + route to fallback
- Nếu >= 0.6 → route to `generate_analogy`

### 3. Analogy Generation (`generate_analogy`)
- LLM sinh narrative (~120 từ tiếng Việt)
- LLM sinh mapping table (tech ↔ analogy)
- Return JSON: `{"narrative": "...", "mapping": [...]}`

### 4. Response Formats

#### Success
```json
{
  "is_fallback": false,
  "needs_clarification": false,
  "confidence": 0.92,
  "narrative": "...",
  "mapping_table": [...]
}
```

#### Clarification
```json
{
  "needs_clarification": true,
  "clarification_question": "..."
}
```

#### Fallback
```json
{
  "is_fallback": true,
  "fallback_reason": "...",
  "suggestions": ["...", "..."]
}
```

---

## 📖 PDF Features

### Current Implementation
- Load PDF từ `/public/` thư mục
- Render tất cả pages (multi-page)
- Text layer overlay cho selection
- Scale: 1.3x (có thể chỉnh)

### PDF Files
```
public/
├── 1-AICB_Ngay_1.pdf
└── day03-tu-chatbot-den-agentic-agent-react-v7.pdf
```

### Sidebar Nav
Mỗi item có `data-pdf` attribute với đường dẫn tới PDF

```html
<div class="nav-item" data-pdf="/public/1-AICB_Ngay_1.pdf">
  <i class="ti ti-file-type-pdf"></i>Day_1_AI_LLM_Foundation.pdf
</div>
```

---

## 🎨 UI/UX Details

### Color Scheme
- Sidebar: `#1e3a5f` (dark blue)
- Accent: `#4a9eff` (light blue)
- Success: `#dcfce7` (light green)
- Danger: `#fee2e2` (light red)
- Warning: `#fff3cd` (light yellow)

### Layout
- Fixed height: 680px (app container)
- Sidebar: 248px width
- Content: flex 1 (responsive)
- Responsive: max-width 1100px

### States
- **Success:** Green bg + checkmark icon
- **Error:** Red bg + alert icon
- **Warning:** Yellow bg + warning icon
- **Loading:** Spinner animation

### Interactions
- Hover effects on buttons/chips
- Shake animation on validation error
- Smooth transitions (0.15s default)
- Modal overlay backdrop (45% opacity)

---

## 📝 Frontend Modal Workflow

### Input View
1. Concept input (pre-fill nếu user bôi đen)
2. Interest input (user nhập sở thích)
3. Plan section (5 bước công việc user có thể chỉnh)
4. History section (collapsible, localStorage)
5. Buttons: Cancel, Send

### Result View
1. Spinner + status label (loading)
2. Timeout progress bar (25s countdown)
3. Showing work (5 steps animation)
4. Result content:
   - Confidence bar (high/low color)
   - Narrative text
   - Mapping table
   - Action buttons (Accept, Edit, Reject, Report)
   - Follow-up chips (Ví dụ khác, So sánh, etc.)

### Error View
1. Error icon + title
2. Detail message
3. Retry button

---

## 🔐 Security Notes

### API Keys
- Store trong `.env` (không commit)
- Load via `dotenv.load_dotenv()`
- Support: OpenAI, Google Gemini, Ollama

### CORS
- Allow origins: `*` (open for demo)
- Production: Limit to specific domains

### Validation
- Concept: min 2 chars, max 80 chars
- Interest: min 2 chars
- No SQL injection (Pydantic validates)

---

## 🛠️ Debugging Tips

### Backend Issues
```bash
# Check if server running
curl http://localhost:8000/health
→ {"status": "ok"}

# View LLM logs
# Check [DEBUG evaluate] lines trong terminal

# Test API manually
curl -X POST http://localhost:8000/api/analogy \
  -H "Content-Type: application/json" \
  -d '{"concept": "RAG", "interest": "bóng đá"}'
```

### Frontend Issues
- Open DevTools (F12)
- Check Console tab cho JS errors
- Check Network tab cho API calls
- Check localStorage: `localStorage.getItem('thu_ky_kim_history')`
- PDF not loading? Check browser console + ngrok logs

### Common Issues
1. **"Đang tải PDF..."** không biến mất
   - Check `/public/` path tồn tại
   - Check PDF.js CDN link (CORS)
   - Check browser console

2. **API returns 500**
   - Check `.env` có API keys?
   - Check LLM provider running (OpenAI/Gemini/Ollama)
   - View server logs

3. **"Không thể kết nối"**
   - Backend chưa start?
   - CORS issue? Check network tab
   - Firewall blocking port 8000?

---

## 📦 Nộp Bài

### Yêu cầu
1. **Repo nhóm** đặt tên: `Day06-Lop-NhomXX` (e.g., `Day06-C401-Nhom03`)
2. **README** liệt kê đủ thành viên (mã HV + họ tên) + mô tả sản phẩm
3. **SPEC** trong `spec/spec.md` + slides (nếu có)
4. **Codebase** trong `codebase/` hoặc link URL
5. **Mỗi người ≥ 1 commit** thực chất (không "Merge branch..." count)

### Cấu trúc Repo
```
Day06-Lop-NhomXX/
├── README.md        # Danh sách thành viên + mô tả
├── spec/
│   └── spec.md      # SPEC chi tiết
└── codebase/        # Source code + hướng dẫn chạy
```

### Hạn nộp
- **Link repo → LMS trước 23:59 ngày 04/06/2026**
- Demo: 16:00 cùng ngày (10 phút/nhóm)

---

## 📚 Tài liệu Liên Quan

- `README.md` — Hướng dẫn chung hackathon
- `hackathon-rules.md` — Luật chơi chi tiết, cách chấm
- `spec/README.md` — Hướng dẫn viết SPEC sản phẩm
- `codebase/README.md` — Hướng dẫn nộp mã nguồn

---

## 🎓 Learning Outcomes

Sau hackathon này, nhóm sẽ:
1. Hiểu cách xây dựng AI product từ SPEC → Prototype
2. Biết khi nào dùng AI, khi nào không
3. Thực hành design thinking: Problem → Solution → Demo
4. Tích hợp LLM vào ứng dụng thực
5. Communicate sản phẩm cho stakeholders

---

**Chúc nhóm bạn thành công! 🚀**
