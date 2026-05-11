# 🏥 AI Medical Assistant — RAG + LLM

**End-to-end Retrieval-Augmented Generation system for patient report analysis.**

Upload patient PDFs → Extract key info → Generate:
- 🧾 Patient Summary
- 🔬 Key Extracted Data (vitals, labs, medications)
- 🩺 Possible Diagnoses (with confidence levels)
- ⚕️ Precautions & Recommendations
- ⚠️ Red Flags / Urgent Alerts

**+ Bonus: 🎙️ Voice Input — speak your question, get an answer.**

> ⚠️ **Disclaimer**: For informational/educational use only. Does NOT replace professional medical advice.

---

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run
python app.py

# 3. Open http://localhost:7860
```

### Setup Steps in the App
1. **🔑 Setup tab** → Paste your OpenAI API key → Click **Initialise Pipeline**
2. **📁 Documents tab** → Upload patient PDFs/TXT files → Click **Index Documents**
3. **🔬 Analyse tab** → Click **Run Full Analysis** → Get structured clinical note
4. **💬 Ask tab** → Type (or speak) any question about the patient report

---

## 📂 Project Structure

```
medical_rag/
├── app.py                          # Main app: RAG pipeline + Gradio UI
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── demo_screenshots/
│   └── analyse_tab_demo.html       # Interactive UI preview
└── sample_data/
    ├── patient_john_matthews_report.txt    # Adult with T2DM, HTN, anaemia
    ├── patient_emily_rodriguez_pediatric.txt  # Paediatric asthma case
    └── lab_report_sarah_kim.txt           # Full lab panel with iron deficiency
```

---

## 🏗️ Architecture

```
Patient PDF / TXT
      │
      ▼  pypdf / plain-text
  Text Extraction
      │
      ▼  MedicalChunker (400-word, section-aware)
  Overlapping Chunks
      │
      ▼  OpenAI text-embedding-3-small (1536-dim, L2-normalised)
  Dense Embeddings
      │
      ▼  FAISS IndexFlatIP (exact cosine similarity)
  Vector Index
      │
   ┌──┴─────────────────────┐
   ▼                         ▼
Full Analysis             Q&A Chat
(top-8 chunks)           (top-K chunks)
   │                         │
   └──────────┬──────────────┘
              ▼
          OpenAI GPT
              │
              ▼
   Answer + Source Citation + Disclaimer
```

---

## 🧩 Core Components

### `MedicalChunker`
Section-aware chunker that recognises clinical headings (`Diagnosis:`, `Medications:`, `Vitals:`, etc.) before falling back to 400-word sliding-window chunks with 80-word overlap. Prevents important medical sections from being split mid-thought.

### `OpenAIEmbedder`
Wraps `text-embedding-3-small` (1536-dim). Vectors are L2-normalised so cosine similarity equals inner product, compatible with FAISS `IndexFlatIP`.

### `FAISSStore`
Exact nearest-neighbour search. Stores chunk metadata alongside the index. Supports add, search, and clear. No persistence between sessions (see [Extending](#extending)).

### `MedicalRAG`
Orchestrates the full pipeline:
- `setup()` — initialise embedder + store
- `ingest()` — parse file, chunk, embed, index
- `retrieve()` — embed query, FAISS search, return top-K chunks
- `analyse()` — retrieve broad context, build structured clinical prompt, call GPT
- `ask()` — retrieve question-specific context, QA prompt, call GPT

### Voice Input
Uses Google Speech Recognition (`SpeechRecognition` library) — free, no API key needed. Records via microphone, transcribes, and fills the question box automatically.

---

## 🎛️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 400 words | Words per chunk |
| `overlap` | 80 words | Overlap between adjacent chunks |
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model |
| `top_k` (analysis) | 8 | Chunks for full analysis |
| `top_k` (Q&A) | 6 | Chunks for Q&A retrieval |
| `temperature` | 0.2 | LLM temperature |
| `max_tokens` | 1500 | Max LLM output tokens |

---

## 📄 Supported File Types

| Format | Parser | Notes |
|--------|--------|-------|
| `.pdf` | `pypdf` | Text-based PDFs; scanned PDFs need OCR |
| `.txt` | Built-in | Plain text reports |
| `.md` | Built-in | Markdown-formatted notes |

---

## 💰 Estimated API Costs

| Operation | Model | Approx. cost |
|-----------|-------|-------------|
| Embed 10-page PDF | `text-embedding-3-small` | ~$0.002 |
| Full analysis | `gpt-4o-mini` | ~$0.01–0.03 |
| Q&A per question | `gpt-4o-mini` | ~$0.005–0.01 |
| Full analysis | `gpt-4o` | ~$0.10–0.20 |

Use `gpt-4o-mini` for development; `gpt-4o` for production/critical use.

---

## 🛠️ Extending the Project <a name="extending"></a>

### Persistent Knowledge Base
```python
import faiss, json

# Save
faiss.write_index(store.index, "medical_kb.index")
with open("medical_chunks.json", "w") as f:
    json.dump(store.chunks, f)

# Load
store.index = faiss.read_index("medical_kb.index")
with open("medical_chunks.json") as f:
    store.chunks = json.load(f)
```

### Use a Local LLM (HIPAA-compliant, zero data egress)
```python
# pip install ollama
import ollama
response = ollama.chat(model="llama3", messages=[...])
```

### Use a Better Embedding Model
```python
# In OpenAIEmbedder:
MODEL = "text-embedding-3-large"  # 3072-dim, higher accuracy
DIM   = 3072
```

### Add Reranking
```python
# pip install sentence-transformers
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, c["text"]) for c in chunks])
chunks = [c for _, c in sorted(zip(scores, chunks), reverse=True)]
```

### OCR for Scanned PDFs
```bash
pip install pytesseract pdf2image
# Convert each page to image, then OCR
```

---

## 🔒 Privacy & Compliance Notes

- Patient data is processed **in memory only** — nothing is saved to disk beyond your session
- For production/HIPAA environments: use a **local LLM** (Ollama + LLaMA 3) to avoid sending data to OpenAI
- For cloud deployment: ensure your OpenAI data processing agreement covers PHI

---

## 📦 Dependencies

```
openai>=1.35.0          # LLM + Embeddings
faiss-cpu>=1.8.0        # Vector search
pypdf>=4.3.0            # PDF parsing
gradio>=4.40.0          # UI
numpy>=1.26.0           # Array ops
SpeechRecognition>=3.10.0  # Voice input
```

---

## 💡 Sample Questions to Try

After uploading the sample patient files:

**Patient John Matthews:**
- *"What medications is the patient currently on?"*
- *"What are the abnormal lab values?"*
- *"Is the patient's diabetes well controlled?"*
- *"What is the plan for the patient's anaemia?"*

**Patient Emily Rodriguez:**
- *"What triggered Emily's asthma episodes?"*
- *"What is the prescribed treatment plan?"*
- *"What should the school nurse know?"*
- *"Does Emily have any drug allergies?"*

**Lab Report Sarah Kim:**
- *"What is causing Sarah's anaemia?"*
- *"Which tests are critically abnormal?"*
- *"What follow-up investigations are recommended?"*

---

*🏥 Built to support healthcare professionals with AI-powered document analysis.*
*Always verify AI outputs with qualified medical expertise.*
