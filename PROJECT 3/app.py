"""
🤖 Multi-Agent Shopping & Research System
══════════════════════════════════════════════════════════════════════
Architecture:
  OrchestratorAgent  ─── plans & routes
        │
   ┌────┼────┬──────────┐
   ▼    ▼    ▼          ▼
 Research  Shopping  Comparison  Critic
  Agent    Agent      Agent      Agent
        │                         │
        └──────── Memory ──────────┘
                 (FAISS RAG)
══════════════════════════════════════════════════════════════════════
"""

import os, sys, json, time, threading
from pathlib import Path
from typing import Generator, List, Dict, Tuple, Optional

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))
from agents import (
    AgentMemory, OrchestratorAgent,
    AGENT_MAP, ALL_AGENT_INFO,
)

# ──────────────────────────────────────────────────────────────────────
#  PIPELINE RUNNER
# ──────────────────────────────────────────────────────────────────────

class MultiAgentRunner:
    """Execute the full multi-agent pipeline for a given query."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model   = model
        self.memory  = AgentMemory(api_key)
        self.orch    = OrchestratorAgent(api_key, model, self.memory)

    def run(self, query: str) -> Generator[Tuple[List[Dict], Dict, str, str], None, None]:
        """
        Yields tuples of (timeline_events, outputs, memory_dump, final_answer)
        progressively as each agent completes.
        """
        timeline: List[Dict] = []
        outputs:  Dict       = {}

        def emit(agent, action, detail="", status="running"):
            ev = {
                "agent": agent.NAME if hasattr(agent, "NAME") else agent,
                "emoji": agent.EMOJI if hasattr(agent, "EMOJI") else "🤖",
                "color": agent.COLOR if hasattr(agent, "COLOR") else "#6c63ff",
                "action": action, "detail": detail, "status": status,
                "ts": time.strftime("%H:%M:%S"),
            }
            timeline.append(ev)
            return ev

        # ── Step 1: Orchestrator plans ─────────────────────────
        emit(self.orch, "Received query", query[:60])
        yield timeline.copy(), outputs.copy(), self.memory.dump(), ""
        time.sleep(0.1)

        emit(self.orch, "Planning execution…")
        yield timeline.copy(), outputs.copy(), self.memory.dump(), ""

        plan_result = self.orch.run(query, {})
        plan = plan_result["plan"]

        emit(self.orch, "Plan ready",
             f"{len(plan['plan'])} agents → {plan.get('query_type','?')} query",
             status="done")
        outputs["plan"] = plan
        yield timeline.copy(), outputs.copy(), self.memory.dump(), ""

        # Build shared context that grows as agents complete
        shared_ctx = {
            "entities":   plan.get("entities", [query]),
            "budget":     plan.get("budget"),
            "intent":     plan.get("intent", query),
            "query_type": plan.get("query_type", "general"),
        }

        # ── Step 2: Execute planned agents ─────────────────────
        agent_steps = plan.get("plan", [])
        for step in agent_steps:
            agent_name = step["agent"]
            task       = step.get("task", query)
            AgentClass = AGENT_MAP.get(agent_name)

            if AgentClass is None:
                continue

            agent = AgentClass(self.api_key, self.model, self.memory)
            emit(agent, f"Starting: {task[:60]}")
            yield timeline.copy(), outputs.copy(), self.memory.dump(), ""

            try:
                result = agent.run(task, {**shared_ctx, **outputs})
                shared_ctx.update(result)
                outputs.update(result)

                # Record agent logs in timeline
                for log_entry in agent.log:
                    emit(agent, log_entry["action"], log_entry.get("detail", ""), status="done")

                emit(agent, "✅ Completed", status="done")
            except Exception as e:
                emit(agent, f"❌ Error: {str(e)[:80]}", status="error")

            yield timeline.copy(), outputs.copy(), self.memory.dump(), \
                  outputs.get("final_answer", "")

        # ── Step 3: Final yield ────────────────────────────────
        final = outputs.get("final_answer", "_No final answer generated._")
        emit(self.orch, "🎯 Pipeline complete", f"Memory: {self.memory.size} entries", status="done")
        yield timeline.copy(), outputs.copy(), self.memory.dump(), final


# ──────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────

def timeline_to_html(events: List[Dict]) -> str:
    if not events:
        return "<div class='tl-empty'>Waiting for query…</div>"
    rows = []
    for ev in events:
        color   = ev.get("color", "#6c63ff")
        emoji   = ev.get("emoji", "🤖")
        agent   = ev.get("agent", "?")
        action  = ev.get("action", "")
        detail  = ev.get("detail", "")
        status  = ev.get("status", "running")
        ts      = ev.get("ts", "")

        dot_color = "#00e5a0" if status == "done" else \
                    "#ff5c7a" if status == "error" else "#ffb84d"

        rows.append(f"""
        <div class="tl-row">
          <div class="tl-dot" style="background:{dot_color}"></div>
          <div class="tl-badge" style="background:{color}22;border-color:{color}55;color:{color}">
            {emoji} {agent}
          </div>
          <div class="tl-text">
            <span class="tl-action">{action}</span>
            {"<span class='tl-detail'>" + detail + "</span>" if detail else ""}
          </div>
          <div class="tl-ts">{ts}</div>
        </div>
        """)
    return "<div class='timeline'>" + "".join(rows) + "</div>"


def outputs_to_tabs(outputs: Dict) -> Tuple[str, str, str, str]:
    return (
        outputs.get("research",    "_No research yet._"),
        outputs.get("shopping",    "_No shopping analysis yet._"),
        outputs.get("comparison",  "_No comparison yet._"),
        outputs.get("final_answer","_No final answer yet._"),
    )


def agent_cards_html() -> str:
    cards = []
    for a in ALL_AGENT_INFO:
        c = a["color"]
        cards.append(f"""
        <div class="agent-card" style="border-color:{c}33">
          <div class="agent-icon" style="background:{c}22;color:{c}">{a['emoji']}</div>
          <div class="agent-info">
            <div class="agent-name" style="color:{c}">{a['name']}</div>
            <div class="agent-desc">{a['desc']}</div>
          </div>
        </div>
        """)
    return "<div class='agent-grid'>" + "".join(cards) + "</div>"


def memory_html(mem_dump: str) -> str:
    if not mem_dump or mem_dump.startswith("_"):
        return "<div class='mem-empty'>Memory is empty</div>"
    lines = mem_dump.strip().split("\n")
    rows = "".join(
        f"<div class='mem-row'>{l}</div>"
        for l in lines
    )
    return f"<div class='mem-list'>{rows}</div>"


# ──────────────────────────────────────────────────────────────────────
#  EXAMPLE QUERIES
# ──────────────────────────────────────────────────────────────────────

EXAMPLES = [
    ["Best laptop for video editing under $1500 — MacBook vs Dell XPS"],
    ["Compare iPhone 16 Pro vs Samsung Galaxy S25 Ultra"],
    ["I want to build a home gym. Budget $800. What should I buy?"],
    ["Best noise-cancelling headphones in 2025 — Sony vs Bose vs Apple AirPods Max"],
    ["Research the best robot vacuum cleaners and recommend one for a family with pets"],
    ["I need a 4K monitor for programming. Under $600. Compare top 3 options"],
    ["What are the best budget mechanical keyboards for typing and gaming?"],
]


# ──────────────────────────────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Clash+Display:wght@600;700&display=swap');

:root {
  --bg:       #07090f;
  --surface:  #0e1218;
  --surface2: #141c26;
  --surface3: #1a2438;
  --border:   #1e2d44;
  --blue:     #00c8ff;
  --green:    #00e5a0;
  --purple:   #c084fc;
  --amber:    #ffb84d;
  --red:      #ff5c7a;
  --text:     #ccdde8;
  --muted:    #4a6880;
  --radius:   14px;
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  color: var(--text) !important;
}

/* ── Header ── */
.mas-header {
  background: linear-gradient(135deg, #0a1020 0%, #07090f 65%);
  border-bottom: 1px solid var(--border);
  padding: 28px 44px; display: flex; align-items: center; gap: 20px;
  position: relative; overflow: hidden;
}
.mas-header::after {
  content:''; position:absolute; right:-80px; top:-80px;
  width:320px; height:320px; border-radius:50%;
  background: radial-gradient(circle, rgba(0,200,255,.06), transparent 70%);
  pointer-events:none;
}
.mas-logo {
  width:56px; height:56px; border-radius:16px; flex-shrink:0;
  background: linear-gradient(135deg, #0055aa, #00c8ff);
  display:flex; align-items:center; justify-content:center; font-size:28px;
  box-shadow: 0 6px 24px rgba(0,200,255,.25);
}
.mas-title h1 {
  font-size:26px; font-weight:700; margin:0;
  background: linear-gradient(90deg, #e0f4ff, var(--blue), var(--green));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  letter-spacing:-.3px;
}
.mas-title p { color:var(--muted); font-size:13px; margin-top:4px; }
.mas-pills { margin-left:auto; display:flex; gap:8px; }
.pill {
  border-radius:20px; padding:5px 13px; font-size:11px;
  font-family:'JetBrains Mono',monospace; border:1px solid; font-weight:500;
}
.pill-blue  { background:rgba(0,200,255,.08); border-color:rgba(0,200,255,.3); color:var(--blue); }
.pill-green { background:rgba(0,229,160,.08); border-color:rgba(0,229,160,.3); color:var(--green); }

/* ── Tabs ── */
.tabs { background:transparent !important; }
.tab-nav {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 5px !important; margin-bottom: 16px !important;
}
.tab-nav button {
  color: var(--muted) !important; border-radius:10px !important;
  font-family:'Space Grotesk',sans-serif !important; font-weight:600 !important;
  font-size:13px !important; padding:10px 18px !important;
  transition:all .2s !important;
}
.tab-nav button.selected {
  background: linear-gradient(135deg, #0055aa, #0088cc) !important;
  color:white !important; box-shadow:0 2px 12px rgba(0,200,255,.2) !important;
}

/* ── Agent grid ── */
.agent-grid { display:flex; flex-wrap:wrap; gap:10px; padding:12px 0; }
.agent-card {
  display:flex; align-items:center; gap:10px;
  background:var(--surface2); border:1px solid; border-radius:10px;
  padding:10px 14px; flex:1; min-width:220px;
}
.agent-icon {
  width:36px; height:36px; border-radius:10px;
  display:flex; align-items:center; justify-content:center;
  font-size:18px; flex-shrink:0;
}
.agent-name { font-weight:600; font-size:13px; }
.agent-desc { color:var(--muted); font-size:11px; margin-top:2px; }

/* ── Query input ── */
.query-area textarea {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important; color: var(--text) !important;
  font-family:'Space Grotesk',sans-serif !important;
  font-size:14px !important; resize:none !important;
}
.query-area textarea:focus { border-color: var(--blue) !important; }

/* ── Buttons ── */
button.primary {
  background: linear-gradient(135deg, #0055aa, #00aae0) !important;
  color:white !important; border:none !important; border-radius:10px !important;
  font-family:'Space Grotesk',sans-serif !important; font-weight:700 !important;
  font-size:15px !important; padding:14px 28px !important;
  box-shadow:0 4px 20px rgba(0,170,224,.3) !important;
  transition:all .2s !important; cursor:pointer !important;
}
button.primary:hover { filter:brightness(1.2) !important; transform:translateY(-1px) !important; }
button.secondary {
  background: var(--surface2) !important; color:var(--text) !important;
  border:1px solid var(--border) !important; border-radius:10px !important;
  font-family:'Space Grotesk',sans-serif !important; font-size:13px !important;
  padding:10px 18px !important; cursor:pointer !important;
}

/* ── Timeline ── */
.timeline { display:flex; flex-direction:column; gap:6px; padding:10px 0; }
.tl-empty { color:var(--muted); font-style:italic; padding:18px; text-align:center; }
.tl-row {
  display:flex; align-items:flex-start; gap:10px;
  padding:8px 12px; background:var(--surface2);
  border-radius:8px; border:1px solid var(--border);
  font-size:12px;
}
.tl-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; margin-top:5px; }
.tl-badge {
  border-radius:6px; padding:2px 10px; font-size:11px;
  font-family:'JetBrains Mono',monospace; border:1px solid;
  white-space:nowrap; flex-shrink:0; font-weight:600;
}
.tl-text { flex:1; line-height:1.5; }
.tl-action { color:var(--text); font-weight:500; }
.tl-detail { color:var(--muted); margin-left:8px; }
.tl-ts { color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:10px; flex-shrink:0; margin-top:2px; }

/* ── Memory ── */
.mem-empty { color:var(--muted); font-style:italic; padding:12px; }
.mem-list { display:flex; flex-direction:column; gap:4px; }
.mem-row {
  background:var(--surface2); border:1px solid var(--border);
  border-radius:6px; padding:6px 10px; font-size:11px;
  color:var(--muted); font-family:'JetBrains Mono',monospace;
  line-height:1.5;
}

/* ── Inputs ── */
input, .gr-input {
  background:var(--surface2) !important; color:var(--text) !important;
  border:1px solid var(--border) !important; border-radius:8px !important;
}
input:focus { border-color:var(--blue) !important; }
label { color:var(--muted) !important; font-size:13px !important; font-weight:500 !important; }
input[type=range] { accent-color:var(--blue) !important; }

/* ── Output markdown ── */
.output-md { background:var(--surface); border:1px solid var(--border);
             border-radius:var(--radius); padding:20px; min-height:160px; }

/* ── Chatbot ── */
.chatbot { background:var(--surface) !important; border:1px solid var(--border) !important;
           border-radius:var(--radius) !important; }
.chatbot .message.user { background:linear-gradient(135deg,#004488,#0077bb) !important;
                          color:white !important; border-radius:14px 14px 4px 14px !important; }
.chatbot .message.bot  { background:var(--surface2) !important; color:var(--text) !important;
                          border:1px solid var(--border) !important;
                          border-radius:14px 14px 14px 4px !important; }

/* ── Accordion ── */
.accordion { background:var(--surface) !important; border:1px solid var(--border) !important;
             border-radius:var(--radius) !important; }
pre, code { background:#04060a !important; border:1px solid var(--border) !important;
            border-radius:6px !important; font-family:'JetBrains Mono',monospace !important;
            font-size:12px !important; color:var(--green) !important; }

/* ── Plan badge ── */
.plan-box {
  background:rgba(0,200,255,.05); border:1px solid rgba(0,200,255,.2);
  border-radius:10px; padding:14px 18px; font-size:12px;
  font-family:'JetBrains Mono',monospace; color:var(--text); line-height:2;
}

/* ── Footer ── */
.mas-footer { text-align:center; color:var(--muted); font-size:11px;
              padding:18px; border-top:1px solid var(--border); margin-top:12px; }

/* Dropdown */
select, .gr-dropdown {
  background:var(--surface2) !important; color:var(--text) !important;
  border:1px solid var(--border) !important; border-radius:8px !important;
}
"""


# ──────────────────────────────────────────────────────────────────────
#  BUILD UI
# ──────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(css=CSS, title="Multi-Agent AI System", theme=gr.themes.Base()) as demo:

        # ── State ──────────────────────────────────────────────
        runner_state = gr.State(None)

        # ── Header ─────────────────────────────────────────────
        gr.HTML("""
        <div class="mas-header">
          <div class="mas-logo">🤖</div>
          <div class="mas-title">
            <h1>Multi-Agent AI System</h1>
            <p>Shopping &amp; Research · Powered by OpenAI GPT + FAISS Memory</p>
          </div>
          <div class="mas-pills">
            <div class="pill pill-blue">5 Specialist Agents</div>
            <div class="pill pill-green">FAISS Memory</div>
          </div>
        </div>
        """)

        with gr.Tabs():

            # ════════════════════════════════════════
            # TAB 1 — Run
            # ════════════════════════════════════════
            with gr.Tab("🚀 Run Agent"):
                with gr.Row():
                    # ── Left: Input + Timeline ──────────────
                    with gr.Column(scale=5):

                        gr.HTML("<br>")
                        gr.HTML(agent_cards_html())

                        gr.Markdown("---")
                        gr.Markdown("### 💬 Your Query")

                        query_box = gr.Textbox(
                            placeholder="e.g. Compare iPhone 16 Pro vs Samsung Galaxy S25 Ultra, budget $1200",
                            lines=3, show_label=False, elem_classes=["query-area"],
                        )

                        with gr.Row():
                            run_btn   = gr.Button("⚡ Launch Agents", variant="primary", scale=3)
                            clear_btn = gr.Button("🗑 Clear", variant="secondary", scale=1)

                        gr.Markdown("**🎯 Example Queries — click to load:**")
                        example_btns = []
                        for ex in EXAMPLES:
                            b = gr.Button(ex[0], variant="secondary", size="sm")
                            example_btns.append((b, ex[0]))

                        gr.Markdown("---")
                        gr.Markdown("### 🕐 Agent Timeline")
                        timeline_html = gr.HTML("<div class='tl-empty'>Launch a query to see agents in action…</div>")

                    # ── Right: Config ───────────────────────
                    with gr.Column(scale=2):
                        gr.Markdown("### ⚙️ Configuration")
                        api_key_box = gr.Textbox(
                            label="OpenAI API Key",
                            placeholder="sk-…", type="password",
                        )
                        model_dd = gr.Dropdown(
                            choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                            value="gpt-4o-mini",
                            label="GPT Model",
                        )
                        gr.HTML("""
                        <div style="background:var(--surface2);border:1px solid var(--border);
                             border-radius:10px;padding:14px;font-size:12px;color:var(--muted);
                             line-height:1.9;margin-top:8px">
                          <b style="color:var(--text)">Agent Flow</b><br>
                          🧠 Orchestrator → plans route<br>
                          🔍 ResearchAgent → context &amp; specs<br>
                          🛒 ShoppingAgent → prices &amp; deals<br>
                          ⚖️ ComparisonAgent → head-to-head<br>
                          🔎 CriticAgent → validate &amp; synthesise<br><br>
                          <b style="color:var(--text)">Shared Memory</b><br>
                          FAISS vector store lets agents<br>
                          recall each other's findings.
                        </div>
                        """)

                        gr.Markdown("### 🧠 Agent Memory")
                        memory_display = gr.HTML(
                            "<div class='mem-empty'>Memory fills as agents run…</div>"
                        )

            # ════════════════════════════════════════
            # TAB 2 — Results
            # ════════════════════════════════════════
            with gr.Tab("📊 Results"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔍 Research Report")
                        research_out = gr.Markdown(
                            "_Run a query to see research output._",
                            elem_classes=["output-md"]
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### 🛒 Shopping Analysis")
                        shopping_out = gr.Markdown(
                            "_Run a query to see shopping recommendations._",
                            elem_classes=["output-md"]
                        )

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚖️ Comparison Report")
                        comparison_out = gr.Markdown(
                            "_Comparison shown when 2+ items detected._",
                            elem_classes=["output-md"]
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 Final Synthesised Answer")
                        final_out = gr.Markdown(
                            "_CriticAgent's final answer will appear here._",
                            elem_classes=["output-md"]
                        )

            # ════════════════════════════════════════
            # TAB 3 — Execution Plan
            # ════════════════════════════════════════
            with gr.Tab("🗺️ Execution Plan"):
                plan_json  = gr.JSON(label="Orchestrator Plan (raw)")
                plan_html  = gr.HTML("<div class='plan-box'>Plan loads after first run…</div>")

            # ════════════════════════════════════════
            # TAB 4 — Memory Explorer
            # ════════════════════════════════════════
            with gr.Tab("🧠 Memory Explorer"):
                gr.Markdown("""
### FAISS Agent Memory
Each agent writes its findings to a shared FAISS vector store.
Agents can retrieve relevant memories from prior steps — enabling genuine collaboration.
                """)
                mem_full_html = gr.HTML(
                    "<div class='mem-empty'>Memory is empty. Run a query first.</div>"
                )
                mem_size_txt  = gr.Markdown("**Memories stored:** 0")

            # ════════════════════════════════════════
            # TAB 5 — Architecture
            # ════════════════════════════════════════
            with gr.Tab("📐 Architecture"):
                gr.Markdown("""
# Multi-Agent System — Architecture

## Agent Communication Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│           OrchestratorAgent 🧠               │
│  • Parses query type (shopping/comparison)  │
│  • Extracts entities, budget, intent        │
│  • Produces ordered execution plan (JSON)   │
└────────────────────┬────────────────────────┘
                     │
        ┌────────────┼────────────┬───────────┐
        ▼            ▼            ▼           ▼
  ResearchAgent  ShoppingAgent ComparisonAgent CriticAgent
  🔍              🛒             ⚖️             🔎
  │              │              │              │
  └──────────────┴──────────────┴──────────────┘
                           │
                 ┌─────────▼─────────┐
                 │   AgentMemory     │
                 │  (FAISS + numpy)  │
                 │  text-embedding   │
                 │  -3-small 1536d   │
                 └───────────────────┘
```

## Orchestrator Plan Schema

```json
{
  "query_type": "comparison",
  "intent": "Compare iPhone 16 Pro vs Galaxy S25 for photography",
  "entities": ["iPhone 16 Pro", "Samsung Galaxy S25 Ultra"],
  "budget": "$1200",
  "plan": [
    {"agent": "ResearchAgent",   "task": "Research both phones specs and camera systems", "depends_on": []},
    {"agent": "ShoppingAgent",   "task": "Find best prices and deals for both phones",   "depends_on": ["ResearchAgent"]},
    {"agent": "ComparisonAgent", "task": "Head-to-head comparison: camera, performance, value", "depends_on": ["ShoppingAgent"]},
    {"agent": "CriticAgent",     "task": "Validate and synthesise final recommendation", "depends_on": ["ComparisonAgent"]}
  ]
}
```

## Agent Specialisations

| Agent | Role | Output |
|-------|------|--------|
| 🧠 Orchestrator | Query parsing + routing | JSON execution plan |
| 🔍 ResearchAgent | Market context + specs | Markdown research report |
| 🛒 ShoppingAgent | Prices + deal scoring | Tiered recommendations |
| ⚖️ ComparisonAgent | Side-by-side analysis | Comparison tables + verdict |
| 🔎 CriticAgent | Validation + synthesis | Final polished answer |

## Shared Memory (FAISS RAG)

Each agent:
1. **Writes** its key findings to the FAISS memory store
2. **Reads** relevant past findings before generating its output

This creates genuine agent collaboration — ShoppingAgent can recall ResearchAgent's
findings about specs to make smarter price-per-feature analysis.

## Why This Architecture?

- **Specialisation**: Each agent has a focused system prompt → better quality per task
- **Parallelism-ready**: Steps without `depends_on` can run in parallel
- **Shared Memory**: FAISS enables agents to build on each other's work
- **Critic pattern**: Final validation step catches errors and improves quality
- **Observable**: Full timeline shows exactly what each agent did

## Extending

```python
# Add a new agent
class PriceTrackerAgent(BaseAgent):
    NAME = "PriceTrackerAgent"
    EMOJI = "📈"

    def run(self, task, context):
        # Use OpenAI to analyse historical price trends
        ...

# Register it
AGENT_MAP["PriceTrackerAgent"] = PriceTrackerAgent
```
                """)

        # ── Footer ─────────────────────────────────────────────
        gr.HTML("""
        <div class="mas-footer">
          🤖 Multi-Agent System · OpenAI GPT · FAISS Vector Memory · Gradio UI
          · OrchestratorAgent · ResearchAgent · ShoppingAgent · ComparisonAgent · CriticAgent
        </div>
        """)

        # ════════════════════════════════════════════════════════
        # EVENT HANDLERS
        # ════════════════════════════════════════════════════════

        def load_example(text):
            return text

        for btn, text in example_btns:
            btn.click(fn=lambda t=text: t, outputs=[query_box])

        def clear_all():
            return (
                "",
                "<div class='tl-empty'>Launch a query to see agents in action…</div>",
                "_Run a query to see research output._",
                "_Run a query to see shopping recommendations._",
                "_Comparison shown when 2+ items detected._",
                "_CriticAgent's final answer will appear here._",
                None, "<div class='plan-box'>Plan loads after first run…</div>",
                "<div class='mem-empty'>Memory is empty. Run a query first.</div>",
                "**Memories stored:** 0",
                "<div class='mem-empty'>Memory fills as agents run…</div>",
            )

        clear_btn.click(
            fn=clear_all,
            outputs=[
                query_box, timeline_html,
                research_out, shopping_out, comparison_out, final_out,
                plan_json, plan_html,
                mem_full_html, mem_size_txt,
                memory_display,
            ],
        )

        def run_pipeline(query: str, api_key: str, model: str):
            """Generator that yields UI updates as agents complete."""

            if not query.strip():
                yield (
                    "<div class='tl-empty'>Enter a query first.</div>",
                    "_No query._", "_No query._", "_No query._", "_No query._",
                    None, "<div class='plan-box'>No plan yet.</div>",
                    "<div class='mem-empty'>No memories.</div>",
                    "**Memories stored:** 0",
                    "<div class='mem-empty'>No memories.</div>",
                )
                return

            if not api_key.strip():
                yield (
                    "<div class='tl-empty' style='color:#ff5c7a'>❌ Please enter your OpenAI API key in ⚙️ Configuration</div>",
                    "_No API key._", "_No API key._", "_No API key._", "_No API key._",
                    None, "<div class='plan-box'>No plan.</div>",
                    "<div class='mem-empty'>No memories.</div>",
                    "**Memories stored:** 0",
                    "<div class='mem-empty'>No memories.</div>",
                )
                return

            try:
                runner = MultiAgentRunner(api_key.strip(), model)

                for timeline_evs, outputs, mem_dump, final in runner.run(query):
                    plan    = outputs.get("plan", {})
                    n_mem   = runner.memory.size

                    yield (
                        timeline_to_html(timeline_evs),
                        outputs.get("research",    "_Researching…_"),
                        outputs.get("shopping",    "_Analysing deals…_"),
                        outputs.get("comparison",  "_Comparing…_"),
                        outputs.get("final_answer","_Synthesising…_"),
                        plan if plan else None,
                        _plan_html(plan),
                        memory_html(mem_dump),
                        f"**Memories stored:** {n_mem}",
                        memory_html(mem_dump),
                    )

            except Exception as e:
                yield (
                    f"<div class='tl-empty' style='color:#ff5c7a'>❌ Error: {e}</div>",
                    f"❌ Error: {e}", "", "", "",
                    None, "<div class='plan-box'>Error.</div>",
                    "<div class='mem-empty'>Error.</div>",
                    "**Memories stored:** 0",
                    "<div class='mem-empty'>Error.</div>",
                )

        def _plan_html(plan: dict) -> str:
            if not plan or "plan" not in plan:
                return "<div class='plan-box'>Plan loads after first run…</div>"
            steps  = plan.get("plan", [])
            intent = plan.get("intent", "")
            qtype  = plan.get("query_type", "")
            budget = plan.get("budget", "—")
            rows = "".join(
                f"<div>{'→' if i else '①②③④⑤'[i]} <b>{s['agent']}</b>: {s['task'][:80]}</div>"
                for i, s in enumerate(steps)
            )
            return (f"<div class='plan-box'>"
                    f"<b>🎯 Intent:</b> {intent}<br>"
                    f"<b>📋 Type:</b> {qtype} | <b>💰 Budget:</b> {budget}<br><br>"
                    f"{rows}</div>")

        run_btn.click(
            fn=run_pipeline,
            inputs=[query_box, api_key_box, model_dd],
            outputs=[
                timeline_html,
                research_out, shopping_out, comparison_out, final_out,
                plan_json, plan_html,
                mem_full_html, mem_size_txt,
                memory_display,
            ],
        )

    return demo


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
