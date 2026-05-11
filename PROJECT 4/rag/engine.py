"""
rag/engine.py
══════════════════════════════════════════════════
RAG Chat Engine — Chat with your Jathakam
Combines: Extracted chart data + Astrology knowledge base
══════════════════════════════════════════════════
"""

from __future__ import annotations
import json, textwrap
from typing import List, Dict, Tuple, Optional
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.knowledge import ASTROLOGY_KNOWLEDGE_CORPUS, RASHIS, GRAHAS, HOUSES


def _openai():
    import openai; return openai
def _faiss():
    import faiss; return faiss


class JathakamRAG:
    """
    FAISS-backed RAG chat for jathakam Q&A.
    Indexes: (1) extracted chart data, (2) astrology knowledge corpus.
    Retrieves relevant context, then answers with GPT.
    """

    DIM   = 1536
    MODEL_EMB = "text-embedding-3-small"

    SYSTEM = textwrap.dedent("""
        You are a wise, warm Vedic astrology guide helping someone understand their jathakam.
        You have access to:
        1. Their personal jathakam (horoscope chart data)
        2. A knowledge base of Vedic astrology principles

        Answer their questions by:
        - Referring specifically to THEIR chart (use actual planet/house positions)
        - Explaining in simple, jargon-free language
        - Being warm, encouraging, and balanced
        - Mentioning classical principles when relevant
        - Being honest if you don't have enough chart data

        Always note: "This is based on Vedic astrology tradition and should be used
        as a reflective guide, not as definitive prediction."

        If asked in Malayalam, respond in English but acknowledge the Malayalam terms.
    """).strip()

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._key   = api_key
        self.model  = model
        faiss = _faiss()
        self.index  = faiss.IndexFlatIP(self.DIM)
        self.chunks: List[Dict] = []
        self.chart_summary = ""

    def _embed(self, texts: List[str]) -> np.ndarray:
        client = _openai().OpenAI(api_key=self._key)
        resp   = client.embeddings.create(model=self.MODEL_EMB, input=texts)
        vecs   = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    def build_index(self, extracted: Dict, raw_text: str, interpretations: Dict):
        """Index all chart data + knowledge base for RAG."""
        documents = []

        # 1. Personal chart summary
        lagna   = extracted.get("lagna", {})
        rasi    = extracted.get("rasi_moon_sign", {})
        naksha  = extracted.get("nakshatra", {})
        planets = extracted.get("planets", {})
        personal= extracted.get("personal", {})
        dasha   = extracted.get("dasha", {})

        chart_text = f"""
PERSONAL JATHAKAM DATA:
Name: {personal.get('name','Not provided')}
Date of Birth: {personal.get('date_of_birth','?')}
Time of Birth: {personal.get('time_of_birth','?')}
Place of Birth: {personal.get('place_of_birth','?')}

LAGNA (Ascendant): {lagna.get('rasi','?')} (House {lagna.get('rasi_number','?')})
RASI (Moon Sign): {rasi.get('rasi','?')}
NAKSHATRA: {naksha.get('name','?')}, Pada {naksha.get('pada','?')}, Lord: {naksha.get('lord','?')}

CURRENT DASHA: {dasha.get('current_mahadasha','?')} Mahadasha / {dasha.get('current_antardasha','?')} Antardasha

PLANET POSITIONS:
{self._fmt_planets(planets)}

SPECIAL NOTES: {extracted.get('special_notes','')}
        """.strip()

        self.chart_summary = chart_text
        documents.append({"text": chart_text, "source": "Personal Chart"})

        # 2. Interpretation sections
        for section, content in interpretations.items():
            if content and not content.startswith("_"):
                documents.append({
                    "text": f"[{section.upper()} ANALYSIS]\n{content}",
                    "source": f"AI Analysis: {section}"
                })

        # 3. Raw OCR text (chunked)
        if raw_text:
            for i in range(0, len(raw_text), 500):
                chunk = raw_text[i:i+500].strip()
                if len(chunk) > 30:
                    documents.append({"text": chunk, "source": "Raw Jathakam Text"})

        # 4. Astrology knowledge base (chunked)
        kb_chunks = ASTROLOGY_KNOWLEDGE_CORPUS.split("\n\n")
        for chunk in kb_chunks:
            if len(chunk.strip()) > 50:
                documents.append({"text": chunk.strip(), "source": "Astrology Knowledge Base"})

        # Embed all documents
        texts = [d["text"] for d in documents]
        if texts:
            # Batch embed (max 100 at a time)
            all_embs = []
            for i in range(0, len(texts), 50):
                batch = texts[i:i+50]
                all_embs.append(self._embed(batch))
            embeddings = np.vstack(all_embs)
            self.index.add(embeddings)
            self.chunks = documents

    def _fmt_planets(self, planets: dict) -> str:
        lines = []
        for p, info in planets.items():
            if isinstance(info, dict) and info.get("house"):
                retro = " (Retrograde)" if info.get("retrograde") else ""
                lines.append(f"  {p} ({GRAHAS.get(p, {}).get('english', p)}): "
                             f"House {info['house']} in {info.get('rasi', '?')}{retro}")
        return "\n".join(lines) or "  (Planet positions not fully available)"

    def retrieve(self, query: str, k: int = 6) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        emb = self._embed([query])
        k   = min(k, self.index.ntotal)
        scores, idxs = self.index.search(emb, k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i >= 0:
                c = self.chunks[i].copy()
                c["score"] = float(score)
                results.append(c)
        return results

    def chat(self, question: str, history: List[Tuple[str, str]]) -> str:
        if self.index.ntotal == 0:
            return "⚠️ Please upload and process a jathakam first, then ask questions."

        # Retrieve relevant context
        contexts = self.retrieve(question, k=6)
        ctx_text = "\n\n".join(
            f"[{c['source']}]\n{c['text']}"
            for c in contexts
        )

        # Build conversation history
        msgs = [{"role": "system", "content": self.SYSTEM}]
        for user_msg, bot_msg in history[-4:]:   # keep last 4 turns
            msgs.append({"role": "user",      "content": user_msg})
            msgs.append({"role": "assistant", "content": bot_msg})

        msgs.append({"role": "user", "content": f"""
Context from jathakam and knowledge base:
{ctx_text}

Question: {question}
        """.strip()})

        try:
            client = _openai().OpenAI(api_key=self._key)
            resp   = client.chat.completions.create(
                model=self.model, max_tokens=800, temperature=0.4,
                messages=msgs
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"❌ Chat error: {e}"

    @property
    def ready(self) -> bool:
        return self.index.ntotal > 0

    @property
    def chunk_count(self) -> int:
        return self.index.ntotal