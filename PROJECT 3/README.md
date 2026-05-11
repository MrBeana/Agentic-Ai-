# 🤖 Multi-Agent Shopping & Research System

**An advanced multi-agent AI system where 5 specialist agents collaborate to answer shopping and research queries — with shared FAISS vector memory.**

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│           OrchestratorAgent 🧠                   │
│  • Parses query (shopping / comparison / research)│
│  • Extracts entities, budget, intent             │
│  • Produces ordered JSON execution plan          │
└────────────────────┬────────────────────────────┘
                     │
      ┌──────────────┼────────────┬────────────────┐
      ▼              ▼            ▼                ▼
ResearchAgent  ShoppingAgent  ComparisonAgent  CriticAgent
    🔍              🛒             ⚖️               🔎
    │               │              │                │
    └───────────────┴──────────────┴────────────────┘
                          │
              ┌───────────▼───────────┐
              │     AgentMemory       │
              │  FAISS IndexFlatIP    │
              │  text-embedding-3-    │
              │  small  (1536-dim)    │
              └───────────────────────┘
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:7860
```

**Steps:**
1. Enter your **OpenAI API key** in the ⚙️ Config panel
2. Pick a **model** (gpt-4o-mini is fast & cheap)
3. Type or pick an **example query**
4. Click **⚡ Launch Agents**
5. Watch the 🕐 **timeline** as agents collaborate in real-time

---

## 📂 Project Structure

```
multi_agent/
├── app.py                          # Main Gradio app + pipeline runner
├── agents/
│   ├── __init__.py                 # Public exports
│   └── core.py                     # All 5 agents + memory
├── requirements.txt
├── README.md
├── demo_screenshots/
│   └── system_demo.html            # Interactive UI preview
└── sample_queries/
    └── example_queries.json        # 8 test queries with expected agent routes
```

---

## 🤖 Agent Specifications

### 🧠 OrchestratorAgent
- **Role**: Master planner — parses the query and routes to specialist agents
- **Output**: JSON execution plan with `agent`, `task`, `depends_on` per step
- **Key skill**: Query classification (shopping / comparison / research / general)

### 🔍 ResearchAgent
- **Role**: Deep domain research — market context, specs, trends, key players
- **Output**: Structured markdown research report
- **Memory**: Writes key findings; reads prior context before generating

### 🛒 ShoppingAgent
- **Role**: Price analysis, deal scoring, tiered recommendations
- **Output**: Budget / Mid-range / Premium picks with scores and smart tips
- **Memory**: Reads ResearchAgent's findings to make better price-per-feature decisions

### ⚖️ ComparisonAgent
- **Role**: Head-to-head comparison — triggered automatically for 2+ entities
- **Output**: Comparison tables, category winners, verdict with % recommendation split
- **Memory**: Reads research + shopping outputs for comprehensive comparison

### 🔎 CriticAgent
- **Role**: Validation + synthesis — reviews all agent outputs, fixes errors, synthesises
- **Output**: Final polished answer with "⚡ Quick Action Plan"
- **Always runs**: Last step in every plan

---

## 🧠 Shared FAISS Memory

```python
# Every agent writes its findings:
memory.remember(
    "RESEARCH: iPhone 16 Pro has 48MP main sensor, ProRes video...",
    agent="ResearchAgent",
    tags=["iPhone", "camera"]
)

# Every agent reads relevant context before running:
context = memory.recall("iPhone camera specs", k=4)
```

This creates **genuine agent collaboration** — agents build on each other's work
rather than operating in isolation.

---

## 💡 Example Queries

| Query | Agents Triggered |
|-------|-----------------|
| `Best laptop under $700 for college` | Research → Shopping → Critic |
| `iPhone 16 Pro vs Galaxy S25 Ultra` | Research → Shopping → **Comparison** → Critic |
| `Build a home gym for $800` | Research → Shopping → Critic |
| `Best noise-cancelling: Sony vs Bose vs Apple` | Research → Shopping → **Comparison** → Critic |
| `4K monitor for programming under $600` | Research → Shopping → Critic |

---

## 🎛️ Configuration

| Parameter | Default | Options |
|-----------|---------|---------|
| Model | `gpt-4o-mini` | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo |
| Embedding | `text-embedding-3-small` | Fixed (1536-dim) |
| Memory recall k | 3–4 | Per-agent setting in `core.py` |
| Max output tokens | 1800 | Per-agent setting |

---

## 💰 Cost Estimates (gpt-4o-mini)

| Query Type | API Calls | Approx. Cost |
|-----------|-----------|-------------|
| Simple shopping | 4 agents | ~$0.02–0.05 |
| 3-way comparison | 5 agents | ~$0.04–0.08 |
| Deep research | 4 agents | ~$0.03–0.06 |

---

## 🛠️ Extending the System

### Add a New Agent

```python
# In agents/core.py:
class PriceHistoryAgent(BaseAgent):
    NAME        = "PriceHistoryAgent"
    EMOJI       = "📈"
    COLOR       = "#00e5a0"
    DESCRIPTION = "Analyses historical price trends and predicts best time to buy"

    SYSTEM = "You are a price analytics expert..."

    def run(self, task: str, context: Dict) -> Dict:
        result = self._chat(self.SYSTEM, task)
        self.memory.remember(result[:300], agent=self.NAME)
        return {"price_history": result}

# Register it:
AGENT_MAP["PriceHistoryAgent"] = PriceHistoryAgent
```

### Add Parallel Execution

```python
import concurrent.futures

# In MultiAgentRunner.run() — run agents without dependencies in parallel:
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(agent.run, task, context): agent for agent in parallel_agents}
    for future in concurrent.futures.as_completed(futures):
        outputs.update(future.result())
```

### Persistent Memory

```python
import faiss, json

# Save
faiss.write_index(memory.index, "agent_memory.faiss")
json.dump(memory.records, open("agent_records.json", "w"))

# Load
memory.index = faiss.read_index("agent_memory.faiss")
memory.records = json.load(open("agent_records.json"))
```

### Use a Local LLM

```python
# Replace _chat() in BaseAgent:
import ollama
def _chat(self, system, user, **kwargs):
    resp = ollama.chat(model="llama3", messages=[
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ])
    return resp["message"]["content"]
```

---

## 📦 Dependencies

```
openai>=1.35.0      # LLM + Embeddings
faiss-cpu>=1.8.0    # Shared vector memory
numpy>=1.26.0       # Vector math
gradio>=4.40.0      # UI
```

---

*🤖 Multi-Agent AI — where specialist agents collaborate smarter than any single model.*
