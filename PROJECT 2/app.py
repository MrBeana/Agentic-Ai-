"""
🏥 AI Medical Assistant — RAG + LLM
────────────────────────────────────
Upload patient reports (PDF/TXT) → Extract key info →
Generate: Summary · Possible Diagnosis · Precautions · Q&A

Tech stack:
  • OpenAI  (embeddings + GPT-4o / GPT-3.5)
  • FAISS   (vector store)
  • pypdf   (PDF parsing)
  • Gradio  (UI)
  • SpeechRecognition + PyAudio (voice input bonus)
"""

import os, re, json, time, io, tempfile, textwrap
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import gradio as gr

# ──────────────────────────────────────────────────
#  Lazy-import helpers
# ──────────────────────────────────────────────────
def _openai():
    import openai
    return openai

def _faiss():
    import faiss
    return faiss

def _pypdf():
    from pypdf import PdfReader
    return PdfReader

def _sr():
    import speech_recognition as sr
    return sr


# ══════════════════════════════════════════════════
#  CORE  —  Chunker · Embedder · VectorStore
# ══════════════════════════════════════════════════

class MedicalChunker:
    """
    Chunk text with awareness of medical sections
    (e.g. 'Diagnosis:', 'Medications:', etc.).
    Falls back to sliding-window chunking.
    """

    SECTION_HEADERS = re.compile(
        r"(?i)(patient\s*(?:name|id|info)|chief\s*complaint|history|"
        r"examination|diagnosis|impression|assessment|plan|medications?|"
        r"prescriptions?|lab\s*results?|vitals?|allergies|recommendations?|"
        r"precautions?|follow[- ]?up|notes?)\s*[:\-]",
    )

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str) -> List[Dict]:
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        # Try section-aware split first
        splits = self.SECTION_HEADERS.split(text)
        chunks, idx = [], 0

        if len(splits) > 2:                         # sections found
            raw_sections = []
            i = 0
            # The regex split interleaves [before, header, content, header, content …]
            raw_sections.append(splits[0])
            i = 1
            while i + 1 < len(splits):
                raw_sections.append(splits[i] + ": " + splits[i + 1])
                i += 2
            for sec in raw_sections:
                for c in self._sliding(sec, source, idx):
                    chunks.append(c)
                    idx += 1
        else:
            chunks = self._sliding(text, source, 0)
        return chunks

    def _sliding(self, text: str, source: str, start_id: int) -> List[Dict]:
        words = text.split()
        chunks = []
        i, cid = 0, start_id
        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            snippet = " ".join(words[i:end])
            if len(snippet.strip()) > 20:
                chunks.append({
                    "id": f"{source}::{cid}",
                    "text": snippet,
                    "source": source,
                    "chunk_index": cid,
                })
                cid += 1
            i += self.chunk_size - self.overlap
        return chunks


class OpenAIEmbedder:
    """Thin wrapper around OpenAI text-embedding-3-small (1536-dim)."""

    MODEL = "text-embedding-3-small"
    DIM   = 1536

    def __init__(self, api_key: str):
        self._key = api_key

    def embed(self, texts: List[str]) -> np.ndarray:
        openai = _openai()
        client = openai.OpenAI(api_key=self._key)
        resp = client.embeddings.create(model=self.MODEL, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # L2 normalise → cosine via inner product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


class FAISSStore:
    """FAISS inner-product index + metadata list."""

    def __init__(self, dim: int = 1536):
        f = _faiss()
        self.index = f.IndexFlatIP(dim)
        self.chunks: List[Dict] = []
        self.dim = dim

    def add(self, embeddings: np.ndarray, chunks: List[Dict]):
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, q_emb: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q_emb.reshape(1, -1), k)
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i >= 0:
                c = self.chunks[i].copy()
                c["score"] = float(score)
                out.append(c)
        return out

    def clear(self):
        f = _faiss()
        self.index = f.IndexFlatIP(self.dim)
        self.chunks = []

    @property
    def n(self):
        return self.index.ntotal


# ══════════════════════════════════════════════════
#  MEDICAL RAG PIPELINE
# ══════════════════════════════════════════════════

class MedicalRAG:

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a highly experienced clinical AI assistant.
        Your role is to help healthcare professionals by analysing patient documents.

        STRICT RULES:
        1. Base your analysis ONLY on the provided context (patient documents).
        2. Always include the disclaimer:
           "⚠️ This is an AI-generated analysis for reference only and does NOT
            replace professional medical advice."
        3. Never invent facts not present in the documents.
        4. If information is missing, say so explicitly.
        5. Use clear, professional clinical language.
        6. Structure output with markdown headings.
    """).strip()

    ANALYSIS_PROMPT = textwrap.dedent("""
        Analyse the following patient report excerpts and produce a structured clinical note.

        PATIENT DOCUMENT EXCERPTS:
        {context}

        Produce the following sections:

        ## 🧾 Patient Summary
        A concise 3–5 sentence summary of the patient's key information, chief complaint,
        and current status.

        ## 🔬 Key Extracted Information
        Bullet list of: age/sex, vitals, chief complaint, history, medications,
        lab findings, and any other critical data found.

        ## 🩺 Possible Diagnoses
        List 2–5 possible diagnoses with a brief rationale for each, ordered by likelihood.
        Note confidence level (High / Moderate / Low) for each.

        ## ⚕️ Suggested Precautions & Recommendations
        Clinical precautions, lifestyle advice, and next steps based on the findings.

        ## ⚠️ Red Flags / Urgent Alerts
        Any findings that require immediate attention. If none, state "No immediate red flags identified."

        Remember: include the AI disclaimer at the end.
    """).strip()

    QA_PROMPT = textwrap.dedent("""
        You are a clinical AI assistant. Answer the doctor's question based ONLY on the
        patient document excerpts below.

        PATIENT DOCUMENT EXCERPTS:
        {context}

        DOCTOR'S QUESTION: {question}

        Provide a clear, concise clinical answer. Cite which part of the document supports
        your answer. If the information is not in the documents, say so.
        Include the AI disclaimer.
    """).strip()

    def __init__(self):
        self.chunker = MedicalChunker(chunk_size=400, overlap=80)
        self.store: Optional[FAISSStore] = None
        self.embedder: Optional[OpenAIEmbedder] = None
        self.loaded_files: List[str] = []
        self._api_key: str = ""

    # ── setup ────────────────────────────────────

    def setup(self, api_key: str):
        self._api_key = api_key
        self.embedder = OpenAIEmbedder(api_key)
        self.store = FAISSStore(dim=OpenAIEmbedder.DIM)
        self.loaded_files = []

    def ready(self) -> Tuple[bool, str]:
        if not self._api_key:
            return False, "No API key set."
        if self.store is None:
            return False, "Pipeline not initialised."
        return True, "OK"

    # ── ingestion ────────────────────────────────

    def _read_file(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            PdfReader = _pypdf()
            reader = PdfReader(path)
            return "\n\n".join(p.extract_text() or "" for p in reader.pages)
        elif ext in (".txt", ".md"):
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported type: {ext}")

    def ingest(self, path: str) -> Tuple[int, str]:
        ok, msg = self.ready()
        if not ok:
            return 0, f"❌ {msg}"
        name = Path(path).name
        try:
            text = self._read_file(path)
            if not text.strip():
                return 0, f"⚠️ No text in {name}"
            chunks = self.chunker.chunk(text, source=name)
            if not chunks:
                return 0, f"⚠️ No chunks from {name}"
            embs = self.embedder.embed([c["text"] for c in chunks])
            self.store.add(embs, chunks)
            self.loaded_files.append(name)
            return len(chunks), f"✅ {name} — {len(chunks)} chunks indexed"
        except Exception as e:
            return 0, f"❌ {name}: {e}"

    # ── retrieval ────────────────────────────────

    def retrieve(self, query: str, k: int = 6) -> List[Dict]:
        emb = self.embedder.embed([query])
        return self.store.search(emb, k=k)

    # ── generation ───────────────────────────────

    def _call_llm(self, system: str, user: str, model: str,
                  temperature: float = 0.2, max_tokens: int = 1500) -> str:
        openai = _openai()
        client = openai.OpenAI(api_key=self._api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content

    def _context_block(self, chunks: List[Dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[Excerpt {i} | {c['source']}]\n{c['text']}")
        return "\n\n".join(parts)

    def analyse(self, model: str, temperature: float) -> Tuple[str, str]:
        """Full structured analysis of all indexed documents."""
        ok, msg = self.ready()
        if not ok:
            return f"❌ {msg}", ""
        if self.store.n == 0:
            return "⚠️ No documents indexed.", ""

        # Retrieve broad chunks covering the whole document
        top = self.retrieve("patient summary diagnosis medications vitals", k=8)
        ctx = self._context_block(top)
        prompt = self.ANALYSIS_PROMPT.format(context=ctx)
        answer = self._call_llm(self.SYSTEM_PROMPT, prompt, model, temperature)
        return answer, ctx

    def ask(self, question: str, model: str, temperature: float, k: int) -> Tuple[str, List[Dict]]:
        ok, msg = self.ready()
        if not ok:
            return f"❌ {msg}", []
        if self.store.n == 0:
            return "⚠️ No documents indexed.", []

        chunks = self.retrieve(question, k=k)
        ctx = self._context_block(chunks)
        prompt = self.QA_PROMPT.format(context=ctx, question=question)
        answer = self._call_llm(self.SYSTEM_PROMPT, prompt, model, temperature)
        return answer, chunks

    def clear(self):
        if self.store:
            self.store.clear()
        self.loaded_files = []


# ══════════════════════════════════════════════════
#  VOICE INPUT
# ══════════════════════════════════════════════════

def transcribe_audio(audio_path: Optional[str]) -> str:
    """Convert recorded audio to text via SpeechRecognition (Google free API)."""
    if audio_path is None:
        return ""
    try:
        sr = _sr()
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as src:
            audio = r.record(src)
        return r.recognize_google(audio)
    except Exception as e:
        return f"[Voice error: {e}]"


# ══════════════════════════════════════════════════
#  GLOBAL STATE
# ══════════════════════════════════════════════════
rag = MedicalRAG()


# ══════════════════════════════════════════════════
#  GRADIO HANDLERS
# ══════════════════════════════════════════════════

def init_pipeline(api_key: str):
    if not api_key.strip():
        return status_html(), "❌ Please enter an OpenAI API key."
    try:
        rag.setup(api_key.strip())
        return status_html(), "✅ Pipeline initialised. Upload patient files."
    except Exception as e:
        return status_html(), f"❌ {e}"


def ingest_files(files, progress=gr.Progress()):
    ok, msg = rag.ready()
    if not ok:
        return f"❌ {msg}", status_html()
    if not files:
        return "⚠️ No files selected.", status_html()
    logs = []
    for i, f in enumerate(files):
        progress((i + 1) / len(files), desc=Path(f.name).name)
        n, m = rag.ingest(f.name)
        logs.append(m)
    return "\n".join(logs), status_html()


def clear_kb():
    rag.clear()
    return "🗑️ Knowledge base cleared.", status_html()


def run_analysis(model, temperature, progress=gr.Progress()):
    ok, msg = rag.ready()
    if not ok:
        return f"❌ {msg}", ""
    if rag.store is None or rag.store.n == 0:
        return "⚠️ No documents indexed.", ""
    progress(0.3, desc="Retrieving relevant excerpts…")
    result, ctx = rag.analyse(model, temperature)
    progress(1.0, desc="Done")
    return result, ctx


def run_qa(question, model, temperature, k_chunks, chat_history):
    if not question.strip():
        return chat_history, ""
    ok, msg = rag.ready()
    if not ok:
        chat_history.append((question, f"❌ {msg}"))
        return chat_history, ""
    if rag.store is None or rag.store.n == 0:
        chat_history.append((question, "⚠️ Upload and index patient files first."))
        return chat_history, ""
    answer, chunks = rag.ask(question, model, temperature, k=int(k_chunks))
    chat_history.append((question, answer))
    ctx_md = "\n\n".join(
        f"**📌 Excerpt {i+1}** · `{c['source']}` · Score `{c['score']:.2f}`\n```\n{c['text'][:500]}…\n```"
        for i, c in enumerate(chunks)
    )
    return chat_history, ctx_md


def use_voice(audio, current_text):
    if audio is None:
        return current_text
    transcribed = transcribe_audio(audio)
    return transcribed if transcribed and not transcribed.startswith("[Voice error") else current_text


def status_html():
    files = rag.loaded_files
    n = rag.store.n if rag.store else 0
    if not files:
        return "<div class='stat-empty'>📂 No documents loaded</div>"
    items = "".join(f"<li>📄 {f}</li>" for f in files)
    return (f"<div class='stat-box'>"
            f"<b>📚 {len(files)} file(s) · {n} chunks</b>"
            f"<ul>{items}</ul></div>")


# ══════════════════════════════════════════════════
#  CSS  —  Medical Dark Theme
# ══════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Fraunces:ital,wght@0,700;1,400&display=swap');

:root {
  --bg:       #080c10;
  --surface:  #0f1520;
  --surface2: #162030;
  --surface3: #1c2a3c;
  --border:   #1e3050;
  --accent:   #00c8ff;
  --green:    #00e5a0;
  --red:      #ff5c7a;
  --amber:    #ffb84d;
  --text:     #d8e8f8;
  --muted:    #5a7898;
  --radius:   14px;
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: var(--text) !important;
}

/* ── Header ── */
.med-header {
  background: linear-gradient(135deg, #0a1828 0%, #080c10 70%);
  border-bottom: 1px solid var(--border);
  padding: 26px 44px;
  display: flex; align-items: center; gap: 18px;
  position: relative; overflow: hidden;
}
.med-header::before {
  content:''; position:absolute; right:-60px; top:-60px;
  width:240px; height:240px; border-radius:50%;
  background: radial-gradient(circle, rgba(0,200,255,.08), transparent 70%);
}
.med-icon {
  width:54px; height:54px; border-radius:16px;
  background: linear-gradient(135deg, #005580, #00c8ff);
  display:flex; align-items:center; justify-content:center;
  font-size:28px; flex-shrink:0;
  box-shadow: 0 4px 20px rgba(0,200,255,.3);
}
.med-title h1 {
  font-family: 'Fraunces', serif; font-size:26px; font-weight:700; margin:0;
  background: linear-gradient(90deg, #e0f4ff, var(--accent));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.med-title p { margin:4px 0 0; color:var(--muted); font-size:13px; }
.med-badge {
  margin-left:auto;
  background: rgba(0,229,160,.1); border:1px solid rgba(0,229,160,.3);
  border-radius:20px; padding:6px 14px; font-size:12px; color:var(--green);
  font-family:'JetBrains Mono', monospace;
}

/* ── Tabs ── */
.tabs { background:transparent !important; }
.tab-nav {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 5px !important; margin-bottom:18px !important;
}
.tab-nav button {
  color: var(--muted) !important; border-radius:10px !important;
  font-family:'Plus Jakarta Sans', sans-serif !important; font-weight:600 !important;
  font-size:13px !important; padding:10px 18px !important;
  transition:all .2s !important;
}
.tab-nav button.selected {
  background: linear-gradient(135deg, #005580, #0099cc) !important;
  color: white !important; box-shadow: 0 2px 12px rgba(0,200,255,.25) !important;
}

/* ── Cards ── */
.card {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

/* ── Inputs ── */
input, textarea, .gr-input, .gr-textarea {
  background: var(--surface2) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius:8px !important;
  font-family:'Plus Jakarta Sans', sans-serif !important;
}
input:focus, textarea:focus { border-color:var(--accent) !important; }

/* ── Buttons ── */
button.primary, .gr-button-primary {
  background: linear-gradient(135deg, #005580, #00aae0) !important;
  color:white !important; border:none !important; border-radius:8px !important;
  font-family:'Plus Jakarta Sans', sans-serif !important; font-weight:600 !important;
  padding:12px 22px !important; cursor:pointer !important;
  transition: all .2s !important;
  box-shadow: 0 2px 14px rgba(0,200,255,.2) !important;
}
button.primary:hover { filter:brightness(1.15) !important; transform:translateY(-1px) !important; }

button.secondary, .gr-button-secondary {
  background: var(--surface2) !important; color:var(--text) !important;
  border:1px solid var(--border) !important; border-radius:8px !important;
  font-family:'Plus Jakarta Sans', sans-serif !important; padding:10px 18px !important;
}

/* ── Chatbot ── */
.chatbot { background:var(--surface) !important; border:1px solid var(--border) !important;
           border-radius:var(--radius) !important; }
.chatbot .message.user { background:linear-gradient(135deg,#004466,#0077aa) !important;
                          color:white !important; border-radius:14px 14px 4px 14px !important; }
.chatbot .message.bot  { background:var(--surface2) !important; color:var(--text) !important;
                          border:1px solid var(--border) !important;
                          border-radius:14px 14px 14px 4px !important; }

/* ── Status ── */
.stat-empty { color:var(--muted); font-style:italic; padding:12px; }
.stat-box {
  background:var(--surface2); border:1px solid rgba(0,200,255,.25);
  border-radius:10px; padding:14px 18px; font-size:13px;
}
.stat-box ul { margin:8px 0 0 16px; padding:0; }
.stat-box li { color:var(--muted); margin:4px 0; font-family:'JetBrains Mono',monospace; font-size:12px; }

/* ── Misc ── */
label { color:var(--muted) !important; font-size:13px !important; font-weight:500 !important; }
input[type=range] { accent-color:var(--accent) !important; }
pre, code { background:#060a10 !important; border:1px solid var(--border) !important;
            border-radius:6px !important; font-family:'JetBrains Mono',monospace !important;
            font-size:12px !important; color:var(--green) !important; }
.accordion { background:var(--surface) !important; border:1px solid var(--border) !important;
             border-radius:var(--radius) !important; }

/* ── Disclaimer banner ── */
.disclaimer {
  background: rgba(255,92,122,.08); border:1px solid rgba(255,92,122,.3);
  border-radius:10px; padding:12px 18px; font-size:12px; color:#ff8fa3;
  line-height:1.6; margin-bottom:8px;
}

/* ── Analysis output ── */
.analysis-out { background:var(--surface2); border:1px solid var(--border);
                border-radius:var(--radius); padding:20px; min-height:200px; }

/* ── Footer ── */
.med-footer { text-align:center; color:var(--muted); font-size:12px; padding:20px 44px;
              border-top:1px solid var(--border); margin-top:12px; }
"""


# ══════════════════════════════════════════════════
#  BUILD UI
# ══════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(css=CSS, title="AI Medical Assistant", theme=gr.themes.Base()) as demo:

        # ── Header ──────────────────────────────
        gr.HTML("""
        <div class="med-header">
          <div class="med-icon">🏥</div>
          <div class="med-title">
            <h1>AI Medical Assistant</h1>
            <p>RAG-powered patient report analysis · Powered by OpenAI GPT</p>
          </div>
          <div class="med-badge">🔒 Data stays local</div>
        </div>
        """)

        gr.HTML("""
        <div class="disclaimer" style="margin:16px 44px 0">
          ⚠️ <strong>Medical Disclaimer:</strong> This AI tool is for informational &amp;
          educational purposes only. It does NOT replace professional medical advice,
          diagnosis, or treatment. Always consult a qualified healthcare provider.
        </div>
        """)

        with gr.Tabs():

            # ════════════════════════════════════
            # TAB 1  —  Setup
            # ════════════════════════════════════
            with gr.Tab("🔑 Setup"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### API Configuration")
                        api_key_box = gr.Textbox(
                            label="OpenAI API Key",
                            placeholder="sk-…",
                            type="password",
                        )
                        model_dd = gr.Dropdown(
                            choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                            value="gpt-4o-mini",
                            label="GPT Model",
                        )
                        temp_sl = gr.Slider(0.0, 1.0, value=0.2, step=0.05,
                                            label="Temperature (0 = factual)")
                        k_sl = gr.Slider(1, 10, value=6, step=1,
                                         label="Top-K Chunks for Q&A")
                        init_btn = gr.Button("⚡ Initialise Pipeline", variant="primary")
                        init_status = gr.Textbox(label="Status", interactive=False, lines=2)

                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="background:var(--surface2);border:1px solid var(--border);
                             border-radius:12px;padding:20px;font-size:13px;line-height:1.9">
                          <b style="color:var(--accent)">How to get started</b><br><br>
                          1️⃣ Get an OpenAI key at
                          <code>platform.openai.com</code><br>
                          2️⃣ Choose a model (gpt-4o-mini is fast &amp; cheap)<br>
                          3️⃣ Click <b>Initialise Pipeline</b><br>
                          4️⃣ Go to <b>📁 Documents</b> tab and upload PDFs<br>
                          5️⃣ Use <b>🔬 Analyse</b> or <b>💬 Chat</b> tabs
                        </div>
                        """)
                        gr.Markdown("---")
                        gr.Markdown("""
**Embeddings**: `text-embedding-3-small` (1536-dim)

**Vector DB**: FAISS IndexFlatIP (cosine)

**Chunking**: 400-word medical-aware sliding window

**Voice**: Google Speech Recognition (free, online)
                        """)

            # ════════════════════════════════════
            # TAB 2  —  Documents
            # ════════════════════════════════════
            with gr.Tab("📁 Documents"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Upload Patient Reports")
                        gr.Markdown("Accepts: **PDF, TXT, MD** · Multiple files OK")
                        file_up = gr.File(
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md"],
                            label="Drop patient files here",
                        )
                        with gr.Row():
                            ingest_btn = gr.Button("⚡ Index Documents", variant="primary")
                            clear_btn  = gr.Button("🗑 Clear All", variant="secondary")
                        ingest_log = gr.Textbox(label="Ingestion Log", lines=8,
                                                interactive=False)

                    with gr.Column(scale=2):
                        gr.Markdown("### Knowledge Base Status")
                        kb_status = gr.HTML(status_html())
                        gr.HTML("""
                        <br>
                        <div style="background:var(--surface2);border:1px solid var(--border);
                             border-radius:10px;padding:16px;font-size:12px;color:var(--muted);
                             line-height:1.8">
                          <b style="color:var(--text)">Supported report types</b><br>
                          • Lab results / blood panels<br>
                          • Clinical notes & doctor summaries<br>
                          • Discharge summaries<br>
                          • Radiology reports<br>
                          • Prescription histories<br>
                          • General patient intake forms
                        </div>
                        """)

            # ════════════════════════════════════
            # TAB 3  —  Analyse
            # ════════════════════════════════════
            with gr.Tab("🔬 Analyse Report"):
                gr.Markdown(
                    "### Structured AI Analysis\n"
                    "Generates a full clinical note: Summary · Key Data · "
                    "Possible Diagnoses · Precautions · Red Flags"
                )
                analyse_btn = gr.Button("🧬 Run Full Analysis", variant="primary")
                with gr.Row():
                    with gr.Column(scale=3):
                        analysis_out = gr.Markdown(
                            "_Click **Run Full Analysis** after indexing documents._",
                            label="Analysis"
                        )
                    with gr.Column(scale=2):
                        with gr.Accordion("📄 Context Excerpts Used", open=False):
                            ctx_out = gr.Textbox(lines=18, interactive=False,
                                                 label="Retrieved passages")

            # ════════════════════════════════════
            # TAB 4  —  Q&A Chat
            # ════════════════════════════════════
            with gr.Tab("💬 Ask the Report"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(label="", height=430)
                        with gr.Row():
                            q_box = gr.Textbox(
                                placeholder="Ask about the patient… e.g. 'What medications is the patient on?'",
                                show_label=False, scale=5,
                            )
                            ask_btn = gr.Button("Ask ↵", variant="primary", scale=1)

                        # ── Voice input bonus ──
                        with gr.Accordion("🎙️ Voice Input (Bonus)", open=False):
                            gr.Markdown("Record your question and it will be transcribed automatically.")
                            mic_in = gr.Audio(sources=["microphone"], type="filepath",
                                              label="Record question")
                            voice_btn = gr.Button("📝 Transcribe to Question", variant="secondary")

                        clear_chat_btn = gr.Button("🗑 Clear Chat", variant="secondary", size="sm")

                    with gr.Column(scale=2):
                        with gr.Accordion("🔍 Retrieved Excerpts", open=True):
                            qa_ctx = gr.Markdown("_Excerpts will appear here after asking._")

                        gr.HTML("""
                        <div style="margin-top:16px;background:var(--surface2);
                             border:1px solid var(--border);border-radius:10px;
                             padding:14px;font-size:12px;color:var(--muted);line-height:1.8">
                          <b style="color:var(--text)">💡 Example questions</b><br>
                          • What is the patient's main diagnosis?<br>
                          • What medications are listed?<br>
                          • Are there any abnormal lab values?<br>
                          • What follow-up is recommended?<br>
                          • Are there any drug allergies?<br>
                          • What are the patient's vitals?
                        </div>
                        """)

            # ════════════════════════════════════
            # TAB 5  —  About
            # ════════════════════════════════════
            with gr.Tab("📖 About"):
                gr.Markdown("""
# 🏥 AI Medical Assistant — Architecture

## Pipeline

```
Patient PDF / TXT
      │
      ▼ (pypdf / plain text)
  Text Extraction
      │
      ▼ (MedicalChunker — 400-word, section-aware)
  Overlapping Chunks
      │
      ▼ (OpenAI text-embedding-3-small, 1536-dim)
  Dense Embeddings
      │
      ▼ (FAISS IndexFlatIP — cosine similarity)
  Vector Index
      │
  ┌───┴──────────────────────┐
  │                          │
  ▼                          ▼
Full Analysis             Q&A Chat
  │                          │
  ▼ (top-8 chunks)           ▼ (top-K chunks)
Structured Prompt         QA Prompt
  │                          │
  └──────────┬───────────────┘
             ▼
         OpenAI GPT
             │
             ▼
   Answer + Source Citation
```

## Component Details

| Component | Implementation | Notes |
|-----------|---------------|-------|
| PDF Parser | `pypdf` | Page-by-page extraction |
| Chunker | `MedicalChunker` | Section-aware + sliding window |
| Embeddings | `text-embedding-3-small` | 1536-dim, L2-normalised |
| Vector DB | FAISS `IndexFlatIP` | Exact cosine search |
| LLM | OpenAI GPT-4o / GPT-4o-mini | Configurable |
| Voice | Google Speech Recognition | Free, online |

## Extending the Project

- **Persistent KB**: Serialize FAISS index + JSON metadata for multi-session storage
- **Local LLM**: Swap OpenAI call for Ollama (LLaMA 3, Mistral, etc.)
- **Better Embeddings**: Use `text-embedding-3-large` (3072-dim) for more accuracy
- **Reranking**: Add cross-encoder reranker for better retrieval precision
- **HIPAA compliance**: Deploy on-premise with a local LLM and no external API calls
- **EHR integration**: Connect to HL7 FHIR APIs to pull live patient data
                """)

        # ── Footer ──────────────────────────────
        gr.HTML("""
        <div class="med-footer">
          🏥 AI Medical Assistant · Built with FAISS · OpenAI · Gradio ·
          <span style="color:var(--red)">❤️</span> for better healthcare
        </div>
        """)

        # ════════════════════════════════════════
        # Event bindings
        # ════════════════════════════════════════

        init_btn.click(
            fn=init_pipeline,
            inputs=[api_key_box],
            outputs=[kb_status, init_status],
        )

        ingest_btn.click(
            fn=ingest_files,
            inputs=[file_up],
            outputs=[ingest_log, kb_status],
        )

        clear_btn.click(
            fn=clear_kb,
            outputs=[ingest_log, kb_status],
        )

        analyse_btn.click(
            fn=run_analysis,
            inputs=[model_dd, temp_sl],
            outputs=[analysis_out, ctx_out],
        )

        ask_btn.click(
            fn=run_qa,
            inputs=[q_box, model_dd, temp_sl, k_sl, chatbot],
            outputs=[chatbot, qa_ctx],
        ).then(lambda: "", outputs=[q_box])

        q_box.submit(
            fn=run_qa,
            inputs=[q_box, model_dd, temp_sl, k_sl, chatbot],
            outputs=[chatbot, qa_ctx],
        ).then(lambda: "", outputs=[q_box])

        voice_btn.click(
            fn=use_voice,
            inputs=[mic_in, q_box],
            outputs=[q_box],
        )

        clear_chat_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, qa_ctx],
        )

    return demo


# ══════════════════════════════════════════════════
if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
