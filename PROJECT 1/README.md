# 🔍 RAG Intelligence

**An end-to-end Retrieval-Augmented Generation (RAG) system built with Gradio, FAISS, sentence-transformers, and OpenAI gpt.**

---

## 📸 Screenshots

The app features four tabs:
- **💬 Chat** — Ask questions, see answers with source citations
- **📁 Documents** — Upload & index your files
- **⚙️ Settings** — API key, model selection, retrieval params
- **📖 About** — Architecture docs

---

## 🏗️ Architecture

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
Claude LLM  ──── Generates grounded answer
    │
    ▼
Answer + Sources
```

### Pipeline Steps

| Step | Component | Detail |
|------|-----------|--------|
| 1. Ingest | `DocumentChunker` | 500-word chunks, 100-word overlap |
| 2. Embed | `SentenceTransformer` | `all-MiniLM-L6-v2`, 384 dims |
| 3. Index | `FAISS` | `IndexFlatIP` with L2-normalised vectors |
| 4. Retrieve | `VectorStore.search` | Cosine similarity, top-K |
| 5. Generate | `Claude` | Context-grounded generation with source citation |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo>
cd rag_project
pip install -r requirements.txt
```

### 2. Set API Key

Get your key from https://console.anthropic.com and paste it into the **Settings** tab (or set env var):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run

```bash
python app.py
```

Open http://localhost:7860 in your browser.

---

## 📂 Project Structure

```
rag_project/
├── app.py                  # Main Gradio application + RAG pipeline
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── demo_screenshots/       # UI screenshots
└── sample_data/            # Example documents to test with
    ├── artificial_intelligence_overview.txt
    ├── climate_change_overview.txt
    └── space_exploration.txt
```

---

## 🧩 Components

### `DocumentChunker`
Splits documents into overlapping text windows. Configurable `chunk_size` (words) and `chunk_overlap`. Overlapping chunks prevent information from being lost at boundaries.

### `EmbeddingEngine`
Wraps `sentence-transformers` for dense vector encoding. Uses `all-MiniLM-L6-v2` by default — a fast, high-quality 384-dimensional embedding model. Vectors are L2-normalised so cosine similarity equals inner product.

### `VectorStore`
Wraps FAISS `IndexFlatIP` (exact inner-product / cosine search). Each chunk is stored alongside its metadata (source filename, position, word count). Supports search, add, and clear operations.

### `RAGPipeline`
Orchestrates the full workflow: ingest files → embed chunks → store in FAISS → retrieve on query → build prompt → call Claude API → return answer + contexts.

---

## 🎛️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 500 words | Words per chunk |
| `chunk_overlap` | 100 words | Overlap between adjacent chunks |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `top_k` | 5 | Number of chunks to retrieve |
| `temperature` | 0.3 | LLM sampling temperature |
| `max_tokens` | 1024 | Max LLM response tokens |
| `llm_model` | `claude-sonnet-4-20250514` | Anthropic model |

---

## 📄 Supported File Types

| Format | Parser |
|--------|--------|
| `.txt`, `.md` | Built-in Python |
| `.pdf` | `pypdf` |
| `.docx`, `.doc` | `python-docx` |

---

## 💡 Example Questions (with sample data)

After uploading the sample documents:

- *"What is machine learning and what are its three main types?"*
- *"How much have global temperatures risen since the pre-industrial era?"*
- *"Who were the first humans to walk on the Moon?"*
- *"What ethical concerns surround AI development?"*
- *"What is the International Space Station?"*

---

## 🛠️ Extending the Project

### Swap the Vector DB
Replace `VectorStore` with ChromaDB, Pinecone, Weaviate, or Qdrant by implementing the same `.add()` / `.search()` interface.

### Add Persistent Storage
Serialize the FAISS index with `faiss.write_index()` and save chunk metadata as JSON to persist the knowledge base between sessions.

### Use a Different LLM
Replace the Anthropic call in `RAGPipeline.generate()` with OpenAI, Cohere, or a local model via Ollama.

### Add Reranking
After initial retrieval, run a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to improve precision.

### Hybrid Search
Combine dense vector search with BM25 keyword search for hybrid retrieval.

---

## 📦 Dependencies

```
sentence-transformers>=2.7.0   # Embeddings
faiss-cpu>=1.8.0               # Vector search
anthropic>=0.30.0              # LLM (Claude)
pypdf>=4.3.0                   # PDF parsing
python-docx>=1.1.2             # DOCX parsing
gradio>=4.40.0                 # UI
numpy>=1.26.0                  # Array ops
```

---

## 📝 License

MIT — free to use, modify, and distribute.

---

*Built with ❤️ using FAISS · sentence-transformers · Gradio · Anthropic Claude*
