# 🔮 AI Jathakam Reader

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Gradio](https://img.shields.io/badge/UI-Gradio-20B2AA.svg)](#)
[![Model](https://img.shields.io/badge/LLM-GPT--4o%20(Vision)-6C63FF.svg)](#)
[![RAG](https://img.shields.io/badge/RAG-FAISS-FFB84D.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](#contributing)

An end-to-end assistant to read, structure, interpret, and converse with an Indian astrology birth chart (Jathakam). Upload a PDF/image/screenshot, let the system run OCR with GPT‑4o Vision, extract structured chart data (Lagna, Nakshatra, planets), generate human-readable interpretations, index everything with FAISS, and chat with your chart in Malayalam/Sanskrit/English. Includes a South Indian chart SVG renderer and a clean, modern Gradio UI.

---

## Why this exists

Reading a Jathakam often requires manual transcription, specialized tools, and scattered references. This project brings the full pipeline together:
- Automate OCR and structure extraction
- Provide contextual, multilingual explanations
- Enable retrieval‑augmented chat grounded in your own chart
- Visualize the South Indian chart clearly in the browser

---

## Key features

- Upload a Jathakam as PDF/Image/Screenshot
- OCR via GPT‑4o Vision to extract raw text
- Structured JSON extraction (Lagna, Nakshatra, planets, houses…)
- Interpretation modules for personality, career, doshas, and yogas
- RAG with FAISS + curated knowledge base for grounded chat
- Multilingual (Malayalam / Sanskrit / English)
- South Indian chart SVG renderer (beautiful and responsive)
- Detailed pipeline progress and logs
- Term explainer (“What is Lagna?”, “What is Rahu?”) with multilingual output

---

## Tech stack and prerequisites

- Python 3.10+
- Gradio (web UI)
- OpenAI API access with GPT‑4o (Vision) or GPT‑4o‑mini (Vision)
- FAISS (faiss-cpu) for the RAG index
- Standard Python tooling (venv, pip)

You’ll need:
- An OpenAI API key (enter it in the app’s Settings when prompted)

---

## Installation

1) Clone the repository
- git clone https://github.com/your-org/ai-jathakam-reader.git
- cd ai-jathakam-reader

2) (Recommended) Create a virtual environment
- python -m venv .venv
- source .venv/bin/activate  # Windows: .venv\Scripts\activate

3) Install dependencies
- If a requirements.txt is provided:
  - pip install -r requirements.txt
- Otherwise, install the essentials:
  - pip install gradio openai faiss-cpu

Note: On some platforms, installing faiss-cpu may require specific Python versions or wheels. If you encounter issues, consult FAISS installation notes or try a different Python minor version.

---

## Quick start

Run the Gradio app locally:
- python app.py

Then:
- Open the local URL shown in your terminal (e.g., http://127.0.0.1:7860/)
- Go to Settings and paste your OpenAI API key
- Upload a Jathakam (PDF/Image/Screenshot)
- Click Process
- Explore the extracted JSON, interpretations, South Indian chart, and start chatting

Sample questions you can ask:
- What is my Lagna and what does it mean?
- What career fields suit my jathakam?
- Do I have Mangal Dosha?
- ലഗ്നം എന്നാൽ എന്താണ്? (What is Lagna?)

---

## Programmatic usage example

If you prefer running the pipeline from Python (outside the UI), you can call the agents directly:

```python
from agents import OCRAgent, ExtractionAgent, InterpretationAgent, LanguageAgent
from rag import JathakamRAG
from visualization import render_south_indian_chart, parse_planet_positions

API_KEY = "sk-..."
MODEL   = "gpt-4o-mini"  # or "gpt-4o" for higher quality

# 1) OCR
ocr = OCRAgent(API_KEY, MODEL)
raw_text, ocr_status = ocr.run("path/to/jathakam.pdf")
if not raw_text:
    raise RuntimeError(f"OCR failed: {ocr_status}")

# 2) Structured extraction
ex_agent = ExtractionAgent(API_KEY, MODEL)
extracted, ex_status = ex_agent.run(raw_text)

# 3) Language detection (optional)
lang_agent = LanguageAgent(API_KEY, MODEL)
lang_info  = lang_agent.run(raw_text)

# 4) Interpretations
interp_agent   = InterpretationAgent(API_KEY, MODEL)
interpretation = interp_agent.run(extracted, raw_text)

# 5) Build a RAG index and chat
rag = JathakamRAG(API_KEY, MODEL)
rag.build_index(extracted, raw_text, interpretation)
answer = rag.chat("What yogas are present in my jathakam?", history=[])
print(answer)

# 6) South Indian chart SVG
planet_positions = parse_planet_positions(extracted)
lagna_house = extracted.get("lagna", {}).get("rasi_number") or 1
svg = render_south_indian_chart(
    planet_positions,
    lagna_house=lagna_house,
    title=extracted.get("personal", {}).get("name", "ജാതകം") or "ജാതകം"
)
with open("chart.svg", "w", encoding="utf-8") as f:
    f.write(svg)
```

---

## Configuration

- OpenAI API key
  - Provided at runtime via the UI Settings. The key is only used client‑side to call the agents and is not stored by the app.
- Model selection
  - Default: gpt‑4o‑mini
  - You can switch to gpt‑4o for higher OCR/LLM quality (higher cost).
- RAG
  - Uses FAISS in-memory index. Internal chunking and retrieval parameters are defined in rag.py (adjust as needed).
- Languages
  - LanguageAgent attempts to detect and adapt output for Malayalam / Sanskrit / English automatically.
- Chart rendering
  - render_south_indian_chart produces an SVG string suitable for embedding or saving.

---

## Project structure

```
ai-jathakam-reader/
├─ app.py                         # Gradio app entrypoint (UI, pipeline wiring, CSS)
├─ agents.py                      # OCRAgent, ExtractionAgent, InterpretationAgent, LanguageAgent
├─ rag.py                         # Jath