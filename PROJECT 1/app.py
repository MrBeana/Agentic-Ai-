"""
RAG (Retrieval-Augmented Generation) System
End-to-End Gradio Application
"""

import os
import re
import time
import json
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import gradio as gr
import numpy as np

# ──────────────────────────────────────────────
#  Lazy imports for heavy libraries
# ──────────────────────────────────────────────
def _import_sentence_transformers():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

def _import_faiss():
    import faiss
    return faiss

def _import_openai():
    from openai import OpenAI
    return OpenAI

def _import_pypdf():
    from pypdf import PdfReader
    return PdfReader

def _import_docx():
    import docx
    return docx

# ──────────────────────────────────────────────
#  RAG Core
# ──────────────────────────────────────────────

class DocumentChunker:
    """Split documents into overlapping chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str) -> List[dict]:
        text = re.sub(r'\n{3,}', '\n\n', text.strip())
        words = text.split()
        chunks, i = [], 0
        chunk_id = 0
        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            chunk_words = words[i:end]
            chunk_text = ' '.join(chunk_words)
            if len(chunk_text.strip()) > 30:
                chunks.append({
                    "id": f"{source}_{chunk_id}",
                    "text": chunk_text,
                    "source": source,
                    "chunk_index": chunk_id,
                    "word_count": len(chunk_words),
                })
                chunk_id += 1
            i += self.chunk_size - self.chunk_overlap
        return chunks


class EmbeddingEngine:
    """Manages sentence-transformer embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            SentenceTransformer = _import_sentence_transformers()
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True,
                                       show_progress_bar=False,
                                       normalize_embeddings=True)
        return embeddings.astype(np.float32)


class VectorStore:
    """FAISS-backed vector store with metadata."""

    def __init__(self, dim: int = 384):
        faiss = _import_faiss()
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine (normalised)
        self.chunks: List[dict] = []

    def add(self, embeddings: np.ndarray, chunks: List[dict]):
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[dict]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                results.append(chunk)
        return results

    @property
    def total_chunks(self):
        return self.index.ntotal

    def clear(self):
        faiss = _import_faiss()
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []


class RAGPipeline:
    """Orchestrates the full RAG workflow."""

    def __init__(self):
        self.chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        self.embedder = EmbeddingEngine()
        self.store = VectorStore(dim=384)
        self.loaded_files: List[str] = []
        self._api_key: Optional[str] = None

    # ── Document ingestion ──────────────────────

    def _read_txt(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    def _read_pdf(self, path: str) -> str:
        PdfReader = _import_pypdf()
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    def _read_docx(self, path: str) -> str:
        docx = _import_docx()
        doc = docx.Document(path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def ingest_file(self, path: str) -> Tuple[int, str]:
        ext = Path(path).suffix.lower()
        name = Path(path).name
        try:
            if ext == ".pdf":
                text = self._read_pdf(path)
            elif ext in (".doc", ".docx"):
                text = self._read_docx(path)
            elif ext in (".txt", ".md"):
                text = self._read_txt(path)
            else:
                return 0, f"❌ Unsupported file type: {ext}"

            if not text.strip():
                return 0, f"⚠️ No text extracted from {name}"

            chunks = self.chunker.chunk_text(text, source=name)
            if not chunks:
                return 0, f"⚠️ No chunks produced from {name}"

            embeddings = self.embedder.embed([c["text"] for c in chunks])
            self.store.add(embeddings, chunks)
            self.loaded_files.append(name)
            return len(chunks), f"✅ {name} → {len(chunks)} chunks indexed"
        except Exception as e:
            return 0, f"❌ Error processing {name}: {e}"

    # ── Retrieval ────────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        q_emb = self.embedder.embed([query])
        return self.store.search(q_emb, k=k)

    # ── Generation ───────────────────────────────

    def build_prompt(self, query: str, contexts: List[dict]) -> str:
        ctx_text = ""
        for i, c in enumerate(contexts, 1):
            ctx_text += f"\n[Source {i}: {c['source']}]\n{c['text']}\n"
        return f"""You are a helpful AI assistant. Answer the user's question using ONLY the provided context below.
If the answer is not in the context, say "I don't have enough information in the provided documents to answer this."
Always cite which source(s) you used.

CONTEXT:{ctx_text}

QUESTION: {query}

ANSWER:"""

    def generate(self, query: str, contexts: List[dict],
             api_key: str, model: str = "gpt-5",
             max_tokens: int = 1024, temperature: float = 0.3) -> str:

        OpenAI = _import_openai()
        client = OpenAI(api_key=api_key)

        prompt = self.build_prompt(query, contexts)

        response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
        return response.choices[0].message.content


    def query(self, question: str, api_key: str, k: int = 5,
              model: str = "gpt-5",
              temperature: float = 0.3) -> Tuple[str, List[dict], str]:
        if self.store.total_chunks == 0:
            return "⚠️ No documents indexed yet. Please upload documents first.", [], ""

        contexts = self.retrieve(question, k=k)
        if not contexts:
            return "⚠️ No relevant context found.", [], ""

        answer = self.generate(question, contexts, api_key, model=model,
                               temperature=temperature)
        prompt = self.build_prompt(question, contexts)
        return answer, contexts, prompt

    def clear_knowledge_base(self):
        self.store.clear()
        self.loaded_files = []


# ──────────────────────────────────────────────
#  Global pipeline instance
# ──────────────────────────────────────────────
pipeline = RAGPipeline()


# ──────────────────────────────────────────────
#  Gradio helpers
# ──────────────────────────────────────────────

def ingest_documents(files, progress=gr.Progress()):
    if not files:
        return "⚠️ No files selected.", status_html()

    messages = []
    total_chunks = 0
    for i, f in enumerate(files):
        progress((i + 1) / len(files), desc=f"Processing {Path(f.name).name}…")
        n, msg = pipeline.ingest_file(f.name)
        total_chunks += n
        messages.append(msg)

    summary = (f"\n\n📚 **Knowledge Base Summary**\n"
               f"- Files loaded: {len(pipeline.loaded_files)}\n"
               f"- Total chunks: {pipeline.store.total_chunks}")
    return "\n".join(messages) + summary, status_html()


def status_html():
    files = pipeline.loaded_files
    total = pipeline.store.total_chunks
    if not files:
        return "<div class='status-empty'>📂 Knowledge base is empty</div>"
    items = "".join(f"<li>📄 {f}</li>" for f in files)
    return (f"<div class='status-box'>"
            f"<strong>📚 {len(files)} file(s) · {total} chunks</strong>"
            f"<ul>{items}</ul></div>")


def clear_kb():
    pipeline.clear_knowledge_base()
    return "🗑️ Knowledge base cleared.", status_html()


def ask_question(question, api_key, k_chunks, model, temperature, chat_history):
    if not question.strip():
        return chat_history, "", ""

    if not api_key.strip():
        chat_history.append({"role": "user", "content": question})

        chat_history.append({"role": "assistant","content": "❌ Please enter your OpenAI API key in the Settings tab."})        
        return chat_history, "", ""

    if pipeline.store.total_chunks == 0:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant","content": "⚠️ No documents indexed. Please upload files first."})
        return chat_history, "", ""

    try:
        answer, contexts, prompt = pipeline.query(
            question, api_key, k=int(k_chunks), model=model, temperature=temperature
        )
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        ctx_md = _contexts_to_md(contexts)
        return chat_history, ctx_md, prompt
    except Exception as e:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({
            "role": "assistant",
            "content": f"❌ Error: {e}"
            })
        return chat_history, "", ""


def _contexts_to_md(contexts: List[dict]) -> str:
    if not contexts:
        return "_No contexts retrieved._"
    lines = []
    for i, c in enumerate(contexts, 1):
        score_pct = f"{c['score']*100:.1f}%"
        lines.append(f"### 📌 Chunk {i} · {c['source']} · Score: {score_pct}")
        lines.append(f"```\n{c['text'][:600]}{'...' if len(c['text'])>600 else ''}\n```")
    return "\n\n".join(lines)


def clear_chat(history):
    return [], "", ""


# ──────────────────────────────────────────────
#  Custom CSS
# ──────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #161922;
    --surface2: #1e2230;
    --border: #2a2f3f;
    --accent: #6c63ff;
    --accent2: #00d4aa;
    --text: #e8eaf0;
    --text-muted: #7a8099;
    --danger: #ff5757;
    --radius: 12px;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* Header */
.rag-header {
    background: linear-gradient(135deg, #1a1d2e 0%, #0d0f14 60%);
    border-bottom: 1px solid var(--border);
    padding: 32px 48px;
    display: flex;
    align-items: center;
    gap: 20px;
}

.rag-logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
}

.rag-title { font-family: 'Syne', sans-serif; }
.rag-title h1 {
    font-size: 28px; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #fff, var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.rag-title p { margin: 4px 0 0; color: var(--text-muted); font-size: 14px; }

/* Tabs */
.tabs { background: transparent !important; }
.tab-nav { background: var(--surface) !important; border-radius: var(--radius) !important;
           padding: 6px !important; border: 1px solid var(--border) !important; margin-bottom: 20px !important; }
.tab-nav button { color: var(--text-muted) !important; border-radius: 8px !important;
                  font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
                  padding: 10px 20px !important; transition: all 0.2s !important; }
.tab-nav button.selected { background: var(--accent) !important; color: white !important; }

/* Panels */
.panel-card {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
}

/* Inputs */
input, textarea, .gr-input, .gr-textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
input:focus, textarea:focus { border-color: var(--accent) !important; outline: none !important; }

/* Buttons */
button.primary, .gr-button.primary {
    background: linear-gradient(135deg, var(--accent), #8b83ff) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; padding: 12px 24px !important;
    transition: all 0.2s !important; cursor: pointer !important;
}
button.primary:hover { transform: translateY(-1px) !important; filter: brightness(1.15) !important; }

button.secondary, .gr-button.secondary {
    background: var(--surface2) !important;
    color: var(--text) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important;
    padding: 10px 20px !important; cursor: pointer !important;
}

/* Chatbot */
.chatbot { background: var(--surface) !important; border: 1px solid var(--border) !important;
           border-radius: var(--radius) !important; }
.chatbot .message.user { background: var(--accent) !important; color: white !important;
                          border-radius: 12px 12px 4px 12px !important; }
.chatbot .message.bot { background: var(--surface2) !important; color: var(--text) !important;
                         border-radius: 12px 12px 12px 4px !important;
                         border: 1px solid var(--border) !important; }

/* Status boxes */
.status-empty { color: var(--text-muted); font-style: italic; padding: 12px; }
.status-box {
    background: var(--surface2); border: 1px solid var(--accent);
    border-radius: 8px; padding: 14px 18px; font-size: 14px;
}
.status-box ul { margin: 8px 0 0 16px; padding: 0; }
.status-box li { color: var(--text-muted); margin: 4px 0; font-family: 'DM Mono', monospace; font-size: 13px; }

/* Upload zone */
.upload-zone { border: 2px dashed var(--border) !important; border-radius: var(--radius) !important;
               background: var(--surface) !important; }
.upload-zone:hover { border-color: var(--accent) !important; }

/* Labels */
label { color: var(--text-muted) !important; font-size: 13px !important;
        font-weight: 500 !important; margin-bottom: 6px !important; }

/* Sliders */
input[type=range] { accent-color: var(--accent) !important; }

/* Accordion */
.accordion { background: var(--surface) !important; border: 1px solid var(--border) !important;
             border-radius: var(--radius) !important; }

/* Code blocks inside markdown */
pre, code { background: #0a0c12 !important; border: 1px solid var(--border) !important;
            border-radius: 6px !important; font-family: 'DM Mono', monospace !important;
            font-size: 13px !important; color: var(--accent2) !important; }

/* Metrics row */
.metric-chip {
    display: inline-block; background: var(--surface2);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 4px 12px; font-size: 12px; font-family: 'DM Mono', monospace;
    color: var(--accent2); margin: 4px;
}

/* Footer */
.rag-footer {
    text-align: center; color: var(--text-muted);
    font-size: 12px; padding: 24px; margin-top: 24px;
    border-top: 1px solid var(--border);
}
"""

# ──────────────────────────────────────────────
#  Build UI
# ──────────────────────────────────────────────

def build_app():
    with gr.Blocks(css=CUSTOM_CSS, title="RAG System", theme=gr.themes.Base()) as app:

        # ── Header ──────────────────────────────
        gr.HTML("""
        <div class="rag-header">
          <div class="rag-logo">🔍</div>
          <div class="rag-title">
            <h1>RAG Intelligence</h1>
            <p>Retrieval-Augmented Generation · Powered by OpenAI + FAISS</p>
          </div>
        </div>
        """)

        with gr.Tabs():

            # ════════════════════════════════════════
            # TAB 1 – Chat
            # ════════════════════════════════════════
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot( 
                            type="messages",
                            label="", height=480,
                            show_label=False, elem_classes=["chatbot"]
                        )
                        with gr.Row():
                            question_box = gr.Textbox(
                                placeholder="Ask anything about your documents…",
                                show_label=False, scale=5,
                                elem_classes=["gr-input"]
                            )
                            ask_btn = gr.Button("Ask ↵", variant="primary", scale=1)
                        clear_btn = gr.Button("🗑 Clear Chat", variant="secondary", size="sm")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📂 Knowledge Base")
                        kb_status = gr.HTML(status_html())

                        with gr.Accordion("🔍 Retrieved Contexts", open=False):
                            ctx_display = gr.Markdown("_Ask a question to see retrieved chunks._")

                        with gr.Accordion("🔧 Debug: Prompt", open=False):
                            prompt_display = gr.Textbox(
                                label="Full prompt sent to LLM",
                                lines=10, interactive=False,
                                elem_classes=["gr-textarea"]
                            )

            # ════════════════════════════════════════
            # TAB 2 – Documents
            # ════════════════════════════════════════
            with gr.Tab("📁 Documents"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Upload Documents")
                        gr.Markdown("Supported: **PDF, TXT, MD, DOCX**")
                        file_upload = gr.File(
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md", ".docx", ".doc"],
                            label="Drop files here or click to browse",
                            elem_classes=["upload-zone"]
                        )
                        with gr.Row():
                            ingest_btn = gr.Button("⚡ Index Documents", variant="primary")
                            clear_kb_btn = gr.Button("🗑 Clear KB", variant="secondary")

                        ingest_log = gr.Textbox(
                            label="Ingestion Log", lines=10,
                            interactive=False, elem_classes=["gr-textarea"]
                        )

                    with gr.Column():
                        gr.Markdown("### Knowledge Base Status")
                        kb_status2 = gr.HTML(status_html())

                        gr.Markdown("---")
                        gr.Markdown("### 💡 Tips")
                        gr.HTML("""
                        <div style='font-size:13px; color:#7a8099; line-height:1.8'>
                        <b style='color:#e8eaf0'>Chunk size</b>: 500 words per chunk<br>
                        <b style='color:#e8eaf0'>Overlap</b>: 100 words between chunks<br>
                        <b style='color:#e8eaf0'>Embeddings</b>: all-MiniLM-L6-v2 (384d)<br>
                        <b style='color:#e8eaf0'>Vector DB</b>: FAISS IndexFlatIP (cosine)<br>
                        <b style='color:#e8eaf0'>LLM</b>: OpenAI GPT-5
                        </div>
                        """)

            # ════════════════════════════════════════
            # TAB 3 – Settings
            # ════════════════════════════════════════
            with gr.Tab("⚙️ Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔑 API Configuration")
                        api_key_input = gr.Textbox(
                            label="OpenAI API Key",
                            placeholder="sk-...",
                            type="password",
                            elem_classes=["gr-input"]
                        )
                        model_dropdown = gr.Dropdown(
                        choices=[
                            "gpt-5",
                            "gpt-5-mini",
                            "gpt-4o",
                            "gpt-4.1"
                        ],
                        value="gpt-5",
                        label="OpenAI Model"
                        )

                    with gr.Column():
                        gr.Markdown("### 🎛 Retrieval Parameters")
                        k_slider = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Top-K Chunks to Retrieve"
                        )
                        temp_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                            label="LLM Temperature"
                        )

            # ════════════════════════════════════════
            # TAB 4 – About
            # ════════════════════════════════════════
            with gr.Tab("📖 About"):
                gr.Markdown("""
# RAG Intelligence — How It Works

## Architecture

```
User Query
    │
    ▼
Embedding Model (all-MiniLM-L6-v2)
    │  384-dim vector
    ▼
FAISS Vector Index  ──── IndexFlatIP (cosine similarity)
    │  Top-K chunks
    ▼
Prompt Builder  ──── Injects context + question
    │
    ▼
OpenAI GPT  ──── Generates grounded answer
    │
    ▼
Answer + Sources
```

## Pipeline Steps

| Step | Component | Detail |
|------|-----------|--------|
| 1. Ingest | DocumentChunker | 500-word chunks, 100-word overlap |
| 2. Embed  | SentenceTransformer | all-MiniLM-L6-v2, 384 dims |
| 3. Index  | FAISS | IndexFlatIP with L2-normalised vectors |
| 4. Retrieve | VectorStore.search | Cosine similarity, top-K |
| 5. Generate | Chatgpt | Context-grounded generation |

## Supported File Types
- 📄 PDF (via pypdf)
- 📝 TXT / Markdown
- 📃 DOCX (via python-docx)

## Key Design Decisions
- **Overlap chunking** preserves context at chunk boundaries
- **Cosine similarity** (via normalised inner product) is robust to length variation
- **Source citation** is enforced in the prompt so the LLM always references its sources
- **Temperature 0.3** balances factuality vs. fluency
                """)

        # ── Footer ──────────────────────────────
        gr.HTML("""
        <div class='rag-footer'>
          Built with 🔍 FAISS · 🤗 sentence-transformers · 🎨 Gradio · 🤖 OpenAI GPT 
        </div>
        """)

        # ════════════════════════════════════════
        # Event handlers
        # ════════════════════════════════════════

        ingest_btn.click(
            fn=ingest_documents,
            inputs=[file_upload],
            outputs=[ingest_log, kb_status2],
        ).then(fn=status_html, outputs=[kb_status])

        clear_kb_btn.click(
            fn=clear_kb,
            outputs=[ingest_log, kb_status2]
        ).then(fn=status_html, outputs=[kb_status])

        ask_btn.click(
            fn=ask_question,
            inputs=[question_box, api_key_input, k_slider, model_dropdown, temp_slider,
                    chatbot],
            outputs=[chatbot, ctx_display, prompt_display],
        ).then(lambda: "", outputs=[question_box])

        question_box.submit(
            fn=ask_question,
            inputs=[question_box, api_key_input, k_slider, model_dropdown, temp_slider,
                    chatbot],
            outputs=[chatbot, ctx_display, prompt_display],
        ).then(lambda: "", outputs=[question_box])

        clear_btn.click(
            fn=clear_chat,
            inputs=[chatbot],
            outputs=[chatbot, ctx_display, prompt_display]
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True,
    )
