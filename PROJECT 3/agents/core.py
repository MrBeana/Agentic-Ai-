"""
agents/core.py
══════════════════════════════════════════════════════════════════
Multi-Agent System — Core Agent Definitions
══════════════════════════════════════════════════════════════════

Agents:
  OrchestratorAgent  — Master planner; breaks query into subtasks
  ResearchAgent      — Deep product / topic research
  ShoppingAgent      — Price analysis, deal scoring, recommendations
  ComparisonAgent    — Side-by-side structured comparison tables
  CriticAgent        — Reviews & improves other agents' outputs
  MemoryStore        — FAISS-based shared agent memory (RAG)
"""

from __future__ import annotations
import json, time, re, hashlib, textwrap
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

# ──────────────────────────────────────────────────────────────
#  Lazy imports
# ──────────────────────────────────────────────────────────────
def _openai():
    import openai
    return openai

def _faiss():
    import faiss
    return faiss


# ══════════════════════════════════════════════════════════════
#  AGENT MEMORY  (FAISS + metadata)
# ══════════════════════════════════════════════════════════════

class AgentMemory:
    """
    Shared short-term memory across all agents within one session.
    Each 'memory' is a text fragment embedded and stored in FAISS.
    Agents can write memories and search for relevant past context.
    """

    DIM   = 1536
    MODEL = "text-embedding-3-small"

    def __init__(self, api_key: str):
        self._key = api_key
        faiss = _faiss()
        self.index = faiss.IndexFlatIP(self.DIM)
        self.records: List[Dict] = []   # {text, agent, timestamp, tags}

    def _embed(self, texts: List[str]) -> np.ndarray:
        openai = _openai()
        client = openai.OpenAI(api_key=self._key)
        resp = client.embeddings.create(model=self.MODEL, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    def remember(self, text: str, agent: str, tags: List[str] = []):
        emb = self._embed([text])
        self.index.add(emb)
        self.records.append({
            "text": text, "agent": agent,
            "timestamp": datetime.now().isoformat(),
            "tags": tags,
        })

    def recall(self, query: str, k: int = 4, agent_filter: str = "") -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        emb = self._embed([query])
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(emb, k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i >= 0:
                r = self.records[i].copy()
                r["score"] = float(score)
                if agent_filter and r["agent"] != agent_filter:
                    continue
                results.append(r)
        return results

    def dump(self) -> str:
        if not self.records:
            return "_No memories stored yet._"
        lines = []
        for r in self.records[-15:]:   # last 15
            lines.append(f"[{r['agent']}] {r['text'][:120]}…")
        return "\n".join(lines)

    @property
    def size(self):
        return len(self.records)


# ══════════════════════════════════════════════════════════════
#  BASE AGENT
# ══════════════════════════════════════════════════════════════

class BaseAgent:
    NAME        = "BaseAgent"
    EMOJI       = "🤖"
    COLOR       = "#6c63ff"
    DESCRIPTION = "Generic agent"

    def __init__(self, api_key: str, model: str, memory: AgentMemory):
        self._key    = api_key
        self.model   = model
        self.memory  = memory
        self.log: List[Dict] = []   # per-run logs

    def _chat(self, system: str, user: str,
              temperature: float = 0.3,
              max_tokens: int = 1800,
              response_format: Optional[Dict] = None) -> str:
        openai = _openai()
        client = openai.OpenAI(api_key=self._key)
        kwargs = dict(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        if response_format:
            kwargs["response_format"] = response_format
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def _log(self, action: str, detail: str = ""):
        entry = {
            "agent": self.NAME, "emoji": self.EMOJI,
            "action": action, "detail": detail,
            "ts": datetime.now().strftime("%H:%M:%S"),
        }
        self.log.append(entry)
        return entry

    def run(self, task: str, context: Dict) -> Dict:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════
#  ORCHESTRATOR AGENT
# ══════════════════════════════════════════════════════════════

class OrchestratorAgent(BaseAgent):
    NAME        = "Orchestrator"
    EMOJI       = "🧠"
    COLOR       = "#00c8ff"
    DESCRIPTION = "Master planner — decomposes queries & routes to specialist agents"

    SYSTEM = textwrap.dedent("""
        You are a master orchestrator for a multi-agent shopping and research system.
        Your job is to analyse the user's query and produce a structured execution plan
        that routes subtasks to the correct specialist agents.

        Available specialist agents:
          - ResearchAgent   : Researches products, brands, technologies, market context
          - ShoppingAgent   : Analyses pricing, deals, value-for-money, recommendations
          - ComparisonAgent : Creates detailed side-by-side comparisons between options
          - CriticAgent     : Reviews and validates other agents' outputs for accuracy

        You MUST respond with ONLY valid JSON (no markdown, no explanation):
        {
          "query_type": "shopping" | "research" | "comparison" | "general",
          "intent": "<one sentence — what does the user really want>",
          "entities": ["<product/topic 1>", "<product/topic 2>", ...],
          "budget": "<extracted budget string or null>",
          "plan": [
            {"agent": "<AgentName>", "task": "<specific task for this agent>", "depends_on": []},
            ...
          ],
          "final_synthesis": "<what the synthesised output should look like>"
        }

        Rules:
        - Always include ResearchAgent first to gather context
        - Include ComparisonAgent only when user is comparing 2+ things
        - Always end with CriticAgent to review the final answer
        - Keep plan steps to 3–5 agents maximum
    """).strip()

    def plan(self, query: str) -> Dict:
        self._log("Planning", query[:80])
        raw = self._chat(self.SYSTEM, f"User query: {query}", temperature=0.1)
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            plan = json.loads(clean)
        except Exception:
            plan = {
                "query_type": "general",
                "intent": query,
                "entities": [query],
                "budget": None,
                "plan": [
                    {"agent": "ResearchAgent",   "task": query, "depends_on": []},
                    {"agent": "ShoppingAgent",   "task": query, "depends_on": ["ResearchAgent"]},
                    {"agent": "CriticAgent",     "task": "Review and summarise findings", "depends_on": ["ShoppingAgent"]},
                ],
                "final_synthesis": "Comprehensive answer",
            }
        self.memory.remember(
            f"PLAN for '{query[:60]}': {json.dumps(plan.get('plan', []))}",
            agent=self.NAME, tags=["plan"]
        )
        return plan

    def run(self, task: str, context: Dict) -> Dict:
        return {"plan": self.plan(task)}


# ══════════════════════════════════════════════════════════════
#  RESEARCH AGENT
# ══════════════════════════════════════════════════════════════

class ResearchAgent(BaseAgent):
    NAME        = "ResearchAgent"
    EMOJI       = "🔍"
    COLOR       = "#00e5a0"
    DESCRIPTION = "Deep product/topic researcher — gathers market context & specifications"

    SYSTEM = textwrap.dedent("""
        You are an expert research analyst specialising in consumer products, technology,
        and market intelligence. You have deep knowledge of products, brands, specifications,
        market trends, and consumer sentiment.

        Your research output must be thorough, factual, and well-structured. Always include:
        - Market overview
        - Key players / brands
        - Important specifications or factors to consider
        - Recent trends or developments
        - Expert consensus and common pitfalls

        Format: Use markdown with clear sections. Be specific with numbers and facts.
        Be comprehensive — this research will feed other agents.
    """).strip()

    def run(self, task: str, context: Dict) -> Dict:
        self._log("Researching", task[:80])

        # Pull relevant memories
        memories = self.memory.recall(task, k=3)
        mem_ctx  = "\n".join(m["text"] for m in memories) if memories else "None"

        entities = context.get("entities", [task])
        budget   = context.get("budget")

        prompt = f"""
Research Task: {task}

Key entities to research: {', '.join(entities)}
User budget context: {budget or 'Not specified'}
Relevant prior context: {mem_ctx}

Provide comprehensive research covering:
1. Market overview and key brands
2. Product categories and what to look for
3. Price ranges and value tiers
4. Key specifications/features that matter
5. Common pitfalls and red flags
6. Current trends and best options in 2024-2025
        """.strip()

        result = self._chat(self.SYSTEM, prompt, temperature=0.3, max_tokens=1600)

        self.memory.remember(
            f"RESEARCH on '{task[:60]}': {result[:300]}",
            agent=self.NAME, tags=["research"] + entities[:2]
        )
        self._log("Research complete", f"{len(result)} chars")
        return {"research": result, "entities": entities}


# ══════════════════════════════════════════════════════════════
#  SHOPPING AGENT
# ══════════════════════════════════════════════════════════════

class ShoppingAgent(BaseAgent):
    NAME        = "ShoppingAgent"
    EMOJI       = "🛒"
    COLOR       = "#ffb84d"
    DESCRIPTION = "Deal finder & recommendation engine — scores value, finds best buys"

    SYSTEM = textwrap.dedent("""
        You are a world-class personal shopping advisor and deal analyst.
        Your expertise is in finding the best value products for any budget,
        identifying hidden gems, and warning against overpriced or poor-quality items.

        For every recommendation you MUST:
        1. Score each product on: Value (1-10), Quality (1-10), Deal Rating (⭐ 1-5)
        2. Include approximate price range (USD)
        3. State clearly: Best Budget / Best Mid-range / Best Premium pick
        4. List 2-3 specific product recommendations per tier with rationale
        5. Add a "Smart Shopper Tips" section with actionable advice

        Be opinionated and specific — vague advice is useless to shoppers.
        Format output as clean markdown with tables where helpful.
    """).strip()

    def run(self, task: str, context: Dict) -> Dict:
        self._log("Analysing deals", task[:80])

        research = context.get("research", "")
        budget   = context.get("budget", "any budget")
        entities = context.get("entities", [])

        # Recall what we know
        memories = self.memory.recall(f"shopping {task}", k=3)
        mem_ctx  = "\n".join(m["text"] for m in memories) if memories else ""

        prompt = f"""
Shopping Task: {task}
User Budget: {budget or 'Not specified — give options across all tiers'}
Products/Topics: {', '.join(entities)}

Research Context from ResearchAgent:
{research[:1200] if research else 'Not available — use your own knowledge'}

Prior memory context:
{mem_ctx[:400] if mem_ctx else 'None'}

Provide:
1. Best Budget Pick (under $50 or entry-level)
2. Best Mid-Range Pick ($50–$200 or mid-tier)
3. Best Premium Pick ($200+ or top-tier)
4. Deal Score table for top 5 products
5. Smart Shopper Tips (timing, platforms, negotiation, alternatives)
6. Red flags to avoid

Be SPECIFIC with product names, model numbers, and prices.
        """.strip()

        result = self._chat(self.SYSTEM, prompt, temperature=0.4, max_tokens=1800)

        self.memory.remember(
            f"SHOPPING RECS for '{task[:60]}': {result[:300]}",
            agent=self.NAME, tags=["shopping", "recommendations"] + entities[:2]
        )
        self._log("Recommendations ready", f"{len(result)} chars")
        return {"shopping": result}


# ══════════════════════════════════════════════════════════════
#  COMPARISON AGENT
# ══════════════════════════════════════════════════════════════

class ComparisonAgent(BaseAgent):
    NAME        = "ComparisonAgent"
    EMOJI       = "⚖️"
    COLOR       = "#c084fc"
    DESCRIPTION = "Head-to-head analyst — structured comparisons with winner verdicts"

    SYSTEM = textwrap.dedent("""
        You are an expert product comparison analyst. Your comparisons are objective,
        data-driven, and always end with a clear verdict.

        For every comparison you MUST produce:
        1. A structured comparison table (markdown) covering all key dimensions
        2. Category-by-category winner declarations
        3. Use-case based recommendations ("Best for X: Product A because…")
        4. Overall verdict with percentage recommendation split
           (e.g., "85% of users should choose A, 15% should choose B")
        5. A "Switcher's Guide" — when does it make sense to switch from one to another?

        Use emojis for winners: ✅ winner, 🟡 tie, ❌ loser
        Be decisive. Wishy-washy comparisons are useless.
    """).strip()

    def run(self, task: str, context: Dict) -> Dict:
        self._log("Comparing", task[:80])

        entities = context.get("entities", [])
        research = context.get("research", "")
        shopping = context.get("shopping", "")

        if len(entities) < 2:
            self._log("Skipped", "Need 2+ entities to compare")
            return {"comparison": "_Comparison skipped: fewer than 2 items to compare._"}

        prompt = f"""
Comparison Task: {task}
Items to compare: {' vs '.join(entities)}

Research context: {research[:800] if research else 'Use your own knowledge'}
Shopping context: {shopping[:600] if shopping else 'Not available'}

Produce a complete head-to-head comparison:
1. Overview table (specs, price, target user)
2. Detailed comparison across: Performance, Value, Build Quality,
   Features, Ecosystem/Support, Longevity
3. Use-case routing (who should pick which)
4. Final verdict with recommendation split
5. Switcher's guide
        """.strip()

        result = self._chat(self.SYSTEM, prompt, temperature=0.2, max_tokens=1800)

        self.memory.remember(
            f"COMPARISON {' vs '.join(entities[:3])}: {result[:300]}",
            agent=self.NAME, tags=["comparison"] + entities[:2]
        )
        self._log("Comparison done", f"{len(entities)} items")
        return {"comparison": result}


# ══════════════════════════════════════════════════════════════
#  CRITIC AGENT
# ══════════════════════════════════════════════════════════════

class CriticAgent(BaseAgent):
    NAME        = "CriticAgent"
    EMOJI       = "🔎"
    COLOR       = "#ff5c7a"
    DESCRIPTION = "Quality controller — validates accuracy, adds caveats, synthesises final answer"

    SYSTEM = textwrap.dedent("""
        You are a senior editor and fact-checker for a consumer advice platform.
        You review outputs from other AI agents and:
        1. Identify any factual errors, outdated info, or unsupported claims
        2. Check for bias or missing perspectives
        3. Add important caveats or disclaimers the user should know
        4. Synthesise all agent outputs into a single, polished final answer

        Your output IS the final answer the user sees. Make it:
        - Well-structured with clear sections
        - Actionable (user knows exactly what to do next)
        - Honest about uncertainty
        - Concise but complete

        Always end with a "⚡ Quick Action Plan" — 3-5 bullet points
        the user should do RIGHT NOW.
    """).strip()

    def run(self, task: str, context: Dict) -> Dict:
        self._log("Reviewing & synthesising", task[:80])

        research   = context.get("research",   "")
        shopping   = context.get("shopping",   "")
        comparison = context.get("comparison", "")
        intent     = context.get("intent",     task)
        query_type = context.get("query_type", "general")

        prompt = f"""
Original user query: {task}
User intent: {intent}
Query type: {query_type}

=== RESEARCH AGENT OUTPUT ===
{research[:1000] if research else 'Not available'}

=== SHOPPING AGENT OUTPUT ===
{shopping[:1000] if shopping else 'Not available'}

=== COMPARISON AGENT OUTPUT ===
{comparison[:1000] if comparison else 'Not applicable'}

Your job:
1. Verify and validate the above outputs
2. Fix any obvious errors or gaps
3. Synthesise into a single definitive answer
4. Add any critical caveats or warnings
5. End with "⚡ Quick Action Plan" — 3–5 steps the user should take NOW
        """.strip()

        result = self._chat(self.SYSTEM, prompt, temperature=0.2, max_tokens=2000)

        self.memory.remember(
            f"FINAL ANSWER for '{task[:60]}': {result[:400]}",
            agent=self.NAME, tags=["final", "reviewed"]
        )
        self._log("Synthesis complete", "✅")
        return {"final_answer": result}


# ══════════════════════════════════════════════════════════════
#  AGENT REGISTRY
# ══════════════════════════════════════════════════════════════

AGENT_MAP = {
    "ResearchAgent":    ResearchAgent,
    "ShoppingAgent":    ShoppingAgent,
    "ComparisonAgent":  ComparisonAgent,
    "CriticAgent":      CriticAgent,
}

ALL_AGENT_INFO = [
    {"name": a.NAME, "emoji": a.EMOJI, "color": a.COLOR, "desc": a.DESCRIPTION}
    for a in [OrchestratorAgent, ResearchAgent, ShoppingAgent, ComparisonAgent, CriticAgent]
]
