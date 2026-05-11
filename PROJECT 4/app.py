"""
🔮 AI Jathakam Reader — End-to-End System
══════════════════════════════════════════════════════════════════════
Pipeline:
  Upload Jathakam (PDF/Image/Screenshot)
       ↓ OCRAgent (GPT-4o Vision)
  Raw Extracted Text
       ↓ ExtractionAgent
  Structured JSON (Lagna, Nakshatra, Planets...)
       ↓ InterpretationAgent
  Personality · Career · Doshas · Yogas
       ↓ RAG (FAISS + Knowledge Base)
  Conversational Chat with Jathakam
       ↓ LanguageAgent
  Multi-language support (Malayalam/Sanskrit/English)
══════════════════════════════════════════════════════════════════════
"""

import os, sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from agents import OCRAgent, ExtractionAgent, InterpretationAgent, LanguageAgent, AGENT_INFO
from rag import JathakamRAG
from visualization import render_south_indian_chart, parse_planet_positions, demo_chart
from data.knowledge import RASHIS, GRAHAS, NAKSHATRAS


# ──────────────────────────────────────────────────────────────
#  Session state
# ──────────────────────────────────────────────────────────────
class Session:
    def __init__(self):
        self.raw_text      = ""
        self.extracted     = {}
        self.interpretations = {}
        self.lang_info     = {}
        self.rag: JathakamRAG = None
        self.chart_svg     = ""
        self.pipeline_logs = []
        self.api_key       = ""
        self.model         = "gpt-4o-mini"

session = Session()


# ──────────────────────────────────────────────────────────────
#  Pipeline runner
# ──────────────────────────────────────────────────────────────
def run_pipeline(file, api_key, model, progress=gr.Progress()):
    """Full pipeline: OCR → Extract → Interpret → RAG index"""
    global session
    if file is None:
        return (status_pill("⚠️ Upload a file first","#ffb84d"),)*1 + ("",)*6 + (demo_chart(),)
    if not api_key.strip():
        return (status_pill("❌ Enter API key in Settings","#ff5c7a"),)*1 + ("",)*6 + (demo_chart(),)

    session.api_key = api_key.strip()
    session.model   = model
    session.pipeline_logs = []

    def log(agent, action, detail=""):
        session.pipeline_logs.append({"agent":agent,"action":action,"detail":detail,"ts":time.strftime("%H:%M:%S")})

    # ── Step 1: OCR ──────────────────────────────────────────
    progress(0.1, desc="👁️ Extracting text from document…")
    log("OCRAgent","Starting","")
    ocr = OCRAgent(api_key.strip(), model)
    raw_text, ocr_status = ocr.run(file.name)
    session.raw_text = raw_text
    for l in ocr.logs: session.pipeline_logs.append(l)
    log("OCRAgent", ocr_status)

    if not raw_text:
        return (
            status_pill(f"❌ OCR failed: {ocr_status}","#ff5c7a"),
            "","","","","","",demo_chart()
        )

    # ── Step 2: Extract ──────────────────────────────────────
    progress(0.3, desc="🔬 Extracting structured data…")
    ex_agent = ExtractionAgent(api_key.strip(), model)
    extracted, ex_status = ex_agent.run(raw_text)
    session.extracted = extracted
    for l in ex_agent.logs: session.pipeline_logs.append(l)
    log("ExtractionAgent", ex_status)

    # ── Step 3: Language ─────────────────────────────────────
    progress(0.4, desc="🌐 Detecting language…")
    lang_agent = LanguageAgent(api_key.strip(), model)
    lang_info  = lang_agent.run(raw_text)
    session.lang_info = lang_info
    for l in lang_agent.logs: session.pipeline_logs.append(l)

    # ── Step 4: Interpret ────────────────────────────────────
    progress(0.55, desc="🔮 Generating interpretations…")
    interp_agent = InterpretationAgent(api_key.strip(), model)
    interpretations = interp_agent.run(extracted, raw_text)
    session.interpretations = interpretations
    for l in interp_agent.logs: session.pipeline_logs.append(l)
    log("InterpretationAgent","All sections done","✅")

    # ── Step 5: Build RAG ────────────────────────────────────
    progress(0.8, desc="📚 Building RAG knowledge index…")
    rag = JathakamRAG(api_key.strip(), model)
    rag.build_index(extracted, raw_text, interpretations)
    session.rag = rag
    log("RAGEngine",f"Index built",f"{rag.chunk_count} chunks")

    # ── Step 6: Render Chart ─────────────────────────────────
    progress(0.9, desc="🎨 Rendering chart…")
    planet_positions = parse_planet_positions(extracted)
    lagna_house = extracted.get("lagna",{}).get("rasi_number") or 1
    chart_svg = render_south_indian_chart(
        planet_positions, lagna_house=lagna_house,
        title=extracted.get("personal",{}).get("name","ജാതകം") or "ജാതകം"
    )
    session.chart_svg = chart_svg

    progress(1.0, desc="✅ Complete!")
    log("Pipeline","✅ Complete",f"RAG: {rag.chunk_count} chunks indexed")

    return (
        status_pill("✅ Jathakam processed successfully","#00e5a0"),
        raw_text[:2000] + ("…" if len(raw_text)>2000 else ""),
        json.dumps(extracted, indent=2, ensure_ascii=False)[:3000],
        interpretations.get("personality",""),
        interpretations.get("career",""),
        interpretations.get("doshas",""),
        interpretations.get("yogas",""),
        chart_svg,
    )

def chat_with_jathakam(question, history):

    if history is None:
        history = []

    if not session.rag or not session.rag.ready:

        history.append({
            "role": "user",
            "content": question
        })

        history.append({
            "role": "assistant",
            "content": "⚠️ Please upload and process a jathakam first."
        })

        return history, ""

    if not question.strip():
        return history, ""

    answer = session.rag.chat(question, history)

    history.append({
        "role": "user",
        "content": question
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    return history, ""


"""def chat_with_jathakam(question, history):
    if not session.rag or not session.rag.ready:
        history.append((question, "⚠️ Please upload and process a jathakam first."))
        return history, ""
    if not question.strip():
        return history, ""
    answer = session.rag.chat(question, history)
    history.append((question, answer))
    return history, ""
"""

def explain_term(term):
    if not session.api_key:
        return "⚠️ Set your API key in Settings first."
    if not term.strip():
        return "Enter a term to explain."
    agent = LanguageAgent(session.api_key, session.model)
    return agent.explain_term(term)


def get_section(section_key):
    return session.interpretations.get(section_key, "_No data — process a jathakam first._")


def status_pill(text, color="#00e5a0"):
    return f'<div style="background:{color}18;border:1px solid {color}44;border-radius:8px;padding:10px 16px;font-size:13px;color:{color};font-weight:600">{text}</div>'


def pipeline_log_html():
    if not session.pipeline_logs:
        return "<div style='color:#4a6880;font-style:italic;padding:12px'>No pipeline run yet.</div>"

    rows = "".join(
        (
            f'<div style="display:flex;gap:10px;padding:7px 12px;background:#141c26;border-radius:7px;'
            f'margin-bottom:5px;font-size:12px;border:1px solid #1e3050">'
            f'<span style="color:{_agent_color(l["agent"])};font-weight:700">'
            f'{_agent_emoji(l["agent"])} {l["agent"]}</span>'
            f'<span style="color:#ccdde8">{l["action"]}</span>'
            + (
                f'<span style="color:#4a6880">{l["detail"]}</span>'
                if l.get("detail")
                else ""
            )
            + f'<span style="color:#4a6880;margin-left:auto;font-family:monospace">'
            f'{l["ts"]}</span>'
            f'</div>'
        )
        for l in session.pipeline_logs
    )

    return rows


def _agent_color(name):
    colors = {"OCRAgent":"#00c8ff","ExtractionAgent":"#00e5a0","InterpretationAgent":"#c084fc",
              "LanguageAgent":"#ffb84d","RAGEngine":"#ff5c7a","Pipeline":"#6c63ff"}
    return colors.get(name,"#ccdde8")

def _agent_emoji(name):
    emojis = {"OCRAgent":"👁️","ExtractionAgent":"🔬","InterpretationAgent":"🔮",
              "LanguageAgent":"🌐","RAGEngine":"📚","Pipeline":"🔧"}
    return emojis.get(name,"🤖")


# ──────────────────────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+Malayalam:wght@400;600&display=swap');

:root {
  --bg:     #07090f; --surface:#0e1218; --surface2:#141c26; --surface3:#1a2438;
  --border: #1e3050; --gold:#c8a84b; --deep:#4a1e7a;
  --blue:   #00c8ff; --green:#00e5a0; --purple:#c084fc;
  --amber:  #ffb84d; --red:#ff5c7a;
  --text:   #ccdde8; --muted:#4a6880; --radius:14px;
}
*,*::before,*::after{box-sizing:border-box}
body,.gradio-container{background:var(--bg)!important;font-family:'Space Grotesk',sans-serif!important;color:var(--text)!important}

/* ── Header ── */
.jh-header{background:linear-gradient(135deg,#0c0820 0%,#07090f 70%);border-bottom:1px solid #2a1860;
  padding:26px 44px;display:flex;align-items:center;gap:20px;position:relative;overflow:hidden}
.jh-header::before{content:'';position:absolute;right:-60px;top:-60px;width:300px;height:300px;border-radius:50%;
  background:radial-gradient(circle,rgba(200,168,75,.06),transparent 70%);pointer-events:none}
.jh-logo{width:54px;height:54px;border-radius:16px;background:linear-gradient(135deg,#4a1e7a,#c8a84b);
  display:flex;align-items:center;justify-content:center;font-size:28px;
  box-shadow:0 6px 24px rgba(200,168,75,.2);flex-shrink:0}
.jh-title h1{font-size:26px;font-weight:700;margin:0;background:linear-gradient(90deg,#f0d890,#c8a84b,#e8c878);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.jh-title p{color:var(--muted);font-size:13px;margin-top:4px}
.jh-pills{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
.jpill{border-radius:20px;padding:4px 12px;font-size:11px;font-family:'JetBrains Mono',monospace;border:1px solid;font-weight:500}
.jpill-gold{background:rgba(200,168,75,.1);border-color:rgba(200,168,75,.35);color:var(--gold)}
.jpill-blue{background:rgba(0,200,255,.08);border-color:rgba(0,200,255,.3);color:var(--blue)}

/* ── Tabs ── */
.tabs{background:transparent!important}
.tab-nav{background:var(--surface)!important;border:1px solid var(--border)!important;
  border-radius:var(--radius)!important;padding:5px!important;margin-bottom:16px!important}
.tab-nav button{color:var(--muted)!important;border-radius:10px!important;
  font-family:'Space Grotesk',sans-serif!important;font-weight:600!important;
  font-size:13px!important;padding:10px 18px!important;transition:all .2s!important}
.tab-nav button.selected{background:linear-gradient(135deg,#4a1e7a,#7a3ea8)!important;
  color:white!important;box-shadow:0 2px 12px rgba(74,30,122,.3)!important}

/* ── Cards ── */
.card{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important}
label{color:var(--muted)!important;font-size:13px!important;font-weight:500!important}

/* ── Inputs ── */
input,textarea,.gr-input,.gr-textarea{background:var(--surface2)!important;color:var(--text)!important;
  border:1px solid var(--border)!important;border-radius:8px!important;font-family:'Space Grotesk',sans-serif!important}
input:focus,textarea:focus{border-color:var(--gold)!important}

/* ── Buttons ── */
button.primary{background:linear-gradient(135deg,#4a1e7a,#c8a84b)!important;
  color:white!important;border:none!important;border-radius:10px!important;
  font-family:'Space Grotesk',sans-serif!important;font-weight:700!important;
  font-size:14px!important;padding:13px 26px!important;
  box-shadow:0 4px 20px rgba(200,168,75,.2)!important;cursor:pointer!important;transition:all .2s!important}
button.primary:hover{filter:brightness(1.18)!important;transform:translateY(-1px)!important}
button.secondary{background:var(--surface2)!important;color:var(--text)!important;
  border:1px solid var(--border)!important;border-radius:10px!important;
  font-family:'Space Grotesk',sans-serif!important;padding:10px 18px!important;cursor:pointer!important}

/* ── Upload area ── */
.upload-area{border:2px dashed #2a1860!important;border-radius:var(--radius)!important;
  background:var(--surface)!important}
.upload-area:hover{border-color:var(--gold)!important}

/* ── Chart container ── */
.chart-wrap{background:transparent;display:flex;justify-content:center;padding:12px 0}

/* ── Chatbot ── */
.chatbot{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important}
.chatbot .message.user{background:linear-gradient(135deg,#3a1060,#6a2090)!important;color:white!important;border-radius:14px 14px 4px 14px!important}
.chatbot .message.bot{background:var(--surface2)!important;color:var(--text)!important;
  border:1px solid var(--border)!important;border-radius:14px 14px 14px 4px!important}

/* ── Accordion ── */
.accordion{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important}
pre,code{background:#04060a!important;border:1px solid var(--border)!important;border-radius:6px!important;
  font-family:'JetBrains Mono',monospace!important;font-size:12px!important;color:var(--green)!important}
input[type=range]{accent-color:var(--gold)!important}
select{background:var(--surface2)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:8px!important}

/* ── Insights cards ── */
.insight-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0}
.insight-card{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:14px;font-size:13px}
.ic-icon{font-size:22px;margin-bottom:6px}
.ic-title{font-weight:700;font-size:12px;color:var(--gold);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
.ic-val{color:var(--text);font-size:14px;font-weight:600}
.ic-sub{color:var(--muted);font-size:11px;margin-top:2px}

/* ── Footer ── */
.jh-footer{text-align:center;color:var(--muted);font-size:11px;padding:18px 44px;border-top:1px solid var(--border);margin-top:10px}
"""


# ──────────────────────────────────────────────────────────────
#  Sample queries for chat
# ──────────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What is my Lagna and what does it mean?",
    "Explain my Nakshatra and how it affects my personality",
    "What career fields suit my jathakam?",
    "Do I have Mangal Dosha? What does it mean?",
    "Explain the current Dasha period and what to expect",
    "What are my strongest planetary placements?",
    "How is my 10th house for career?",
    "Explain what Rahu in my chart means",
    "What yogas are present in my jathakam?",
    "ലഗ്നം എന്നാൽ എന്താണ്? (What is Lagna?)",  # Malayalam question
]


# ──────────────────────────────────────────────────────────────
#  BUILD UI
# ──────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(css=CSS, title="AI Jathakam Reader", theme=gr.themes.Base()) as demo:

        # ── Header ──────────────────────────────────────────
        gr.HTML("""
        <div class="jh-header">
          <div class="jh-logo">🔮</div>
          <div class="jh-title">
            <h1>AI Jathakam Reader</h1>
            <p>ജാതകം · Horoscope AI · OCR → Extraction → RAG → Insights Dashboard</p>
          </div>
          <div class="jh-pills">
            <div class="jpill jpill-gold">GPT-4o Vision OCR</div>
            <div class="jpill jpill-blue">FAISS RAG</div>
            <div class="jpill jpill-gold">Malayalam Support</div>
            <div class="jpill jpill-blue">Multi-Agent</div>
          </div>
        </div>
        """)

        with gr.Tabs():

            # ════════════════════════════════════════
            # TAB 1 — Upload & Process
            # ════════════════════════════════════════
            with gr.Tab("📤 Upload Jathakam"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Upload Your Jathakam")
                        gr.Markdown(
                            "Supports: **PDF · JPG · PNG · Screenshot · Scanned images · Text files**\n\n"
                            "Works with Malayalam, Sanskrit, and English jathakams."
                        )
                        file_upload = gr.File(
                            file_count="single",
                            file_types=[".pdf",".jpg",".jpeg",".png",".webp",".bmp",".txt",".md"],
                            label="Drop jathakam file here",
                            elem_classes=["upload-area"],
                        )
                        process_btn = gr.Button("✨ Process Jathakam", variant="primary")
                        status_out  = gr.HTML(
                            '<div style="color:#4a6880;font-style:italic;padding:10px">Upload a file and click Process.</div>'
                        )

                        gr.Markdown("---")
                        gr.Markdown("### 🗺️ Pipeline Activity Log")
                        log_html = gr.HTML("<div style='color:#4a6880;font-style:italic;padding:12px'>No pipeline run yet.</div>")

                    with gr.Column(scale=2):
                        gr.HTML("""
                        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:20px">
                          <div style="font-size:20px;margin-bottom:12px;text-align:center">🔮</div>
                          <div style="font-size:13px;font-weight:700;color:#c8a84b;margin-bottom:14px;text-align:center">HOW IT WORKS</div>
                          <div style="display:flex;flex-direction:column;gap:10px;font-size:12px">
                            <div style="display:flex;gap:10px;align-items:center">
                              <div style="width:28px;height:28px;border-radius:8px;background:rgba(0,200,255,.15);display:flex;align-items:center;justify-content:center;flex-shrink:0">👁️</div>
                              <div><b style="color:#00c8ff">OCR Agent</b><br><span style="color:#4a6880">GPT-4o Vision reads your jathakam image/PDF</span></div>
                            </div>
                            <div style="display:flex;gap:10px;align-items:center">
                              <div style="width:28px;height:28px;border-radius:8px;background:rgba(0,229,160,.15);display:flex;align-items:center;justify-content:center;flex-shrink:0">🔬</div>
                              <div><b style="color:#00e5a0">Extraction Agent</b><br><span style="color:#4a6880">Extracts Lagna, Rasi, Nakshatra, planet positions</span></div>
                            </div>
                            <div style="display:flex;gap:10px;align-items:center">
                              <div style="width:28px;height:28px;border-radius:8px;background:rgba(192,132,252,.15);display:flex;align-items:center;justify-content:center;flex-shrink:0">🔮</div>
                              <div><b style="color:#c084fc">Interpretation Agent</b><br><span style="color:#4a6880">Personality · Career · Doshas · Yogas</span></div>
                            </div>
                            <div style="display:flex;gap:10px;align-items:center">
                              <div style="width:28px;height:28px;border-radius:8px;background:rgba(255,184,77,.15);display:flex;align-items:center;justify-content:center;flex-shrink:0">🌐</div>
                              <div><b style="color:#ffb84d">Language Agent</b><br><span style="color:#4a6880">Malayalam/Sanskrit terms detected & translated</span></div>
                            </div>
                            <div style="display:flex;gap:10px;align-items:center">
                              <div style="width:28px;height:28px;border-radius:8px;background:rgba(255,92,122,.15);display:flex;align-items:center;justify-content:center;flex-shrink:0">📚</div>
                              <div><b style="color:#ff5c7a">RAG Engine</b><br><span style="color:#4a6880">Builds FAISS index for intelligent Q&A chat</span></div>
                            </div>
                          </div>
                        </div>
                        """)

                        gr.Markdown("### ⚙️ Settings", elem_classes=["card"])
                        api_key_in = gr.Textbox(label="OpenAI API Key", placeholder="sk-…", type="password")
                        model_dd   = gr.Dropdown(
                            choices=["gpt-4o-mini","gpt-4o","gpt-4-turbo","gpt-3.5-turbo"],
                            value="gpt-4o-mini", label="GPT Model"
                        )
                        gr.HTML("""
                        <div style="background:rgba(200,168,75,.06);border:1px solid rgba(200,168,75,.2);
                             border-radius:8px;padding:10px 14px;font-size:11px;color:#8a7050;margin-top:8px">
                          💡 <b>gpt-4o-mini</b> is recommended — fast & cost-effective.<br>
                          Use <b>gpt-4o</b> for scanned/complex images (better vision OCR).
                        </div>
                        """)

            # ════════════════════════════════════════
            # TAB 2 — Insights Dashboard
            # ════════════════════════════════════════
            with gr.Tab("✨ Insights Dashboard"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 🎯 Personality Profile")
                        personality_out = gr.Markdown("_Process a jathakam to see your personality profile._")

                        gr.Markdown("### 💼 Career Tendencies")
                        career_out = gr.Markdown("_Process a jathakam to see career analysis._")

                    with gr.Column(scale=1):
                        gr.Markdown("### 🗺️ South Indian Chart")
                        chart_out = gr.HTML(demo_chart())

                        gr.Markdown("### 🌟 Special Yogas")
                        yogas_out = gr.Markdown("_Yogas appear after processing._")

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔴 Dosha Analysis")
                        doshas_out = gr.Markdown("_Dosha analysis appears after processing._")
                    with gr.Column():
                        with gr.Accordion("📊 More Insights", open=False):
                            health_btn  = gr.Button("🏥 Health Tendencies", variant="secondary")
                            health_out  = gr.Markdown()
                            rel_btn     = gr.Button("💑 Relationships", variant="secondary")
                            rel_out     = gr.Markdown()
                            dasha_btn   = gr.Button("⏳ Current Dasha Period", variant="secondary")
                            dasha_out   = gr.Markdown()

            # ════════════════════════════════════════
            # TAB 3 — Chat with Jathakam
            # ════════════════════════════════════════
            with gr.Tab("💬 Chat with Jathakam"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot( label="", height=450, type="messages")
                        with gr.Row():
                            chat_in  = gr.Textbox(
                                placeholder="Ask anything about your jathakam… e.g. 'What is my Lagna?' or 'ലഗ്നം എന്തൊക്കെ?'",
                                show_label=False, scale=5,
                            )
                            chat_btn = gr.Button("Ask 🔮", variant="primary", scale=1)
                        clear_chat_btn = gr.Button("🗑 Clear", variant="secondary", size="sm")

                    with gr.Column(scale=2):
                        gr.Markdown("### 💡 Sample Questions")
                        for q in SAMPLE_QUESTIONS:
                            btn = gr.Button(q, variant="secondary", size="sm")
                            btn.click(fn=lambda x=q: x, outputs=[chat_in])

            # ════════════════════════════════════════
            # TAB 4 — Raw Data
            # ════════════════════════════════════════
            with gr.Tab("🔬 Raw Data"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 👁️ OCR Extracted Text")
                        ocr_out = gr.Textbox(label="", lines=20, interactive=False)
                    with gr.Column():
                        gr.Markdown("### 🔬 Structured Extraction (JSON)")
                        json_out = gr.Code(language="json", label="", lines=20)

            # ════════════════════════════════════════
            # TAB 5 — Language Tools
            # ════════════════════════════════════════
            with gr.Tab("🌐 Language Tools"):
                gr.Markdown("""
### Malayalam / Sanskrit Term Explainer
Enter any astrological term in Malayalam, Sanskrit, or English to get a clear explanation.
                """)
                with gr.Row():
                    with gr.Column():
                        term_in  = gr.Textbox(label="Enter astrological term", placeholder="e.g. ലഗ്നം / Nakshatra / Gaja Kesari Yoga")
                        explain_btn = gr.Button("🌐 Explain Term", variant="primary")
                        term_out = gr.Markdown("_Enter a term and click Explain._")

                    with gr.Column():
                        gr.Markdown("### Common Terms Quick Reference")
                        gr.HTML("""
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px">
                          """ + "".join(
                            f'<div style="background:#141c26;border:1px solid #1e3050;border-radius:8px;padding:8px 10px">'
                            f'<span style="color:#c8a84b;font-weight:700">{ml}</span><br>'
                            f'<span style="color:#ccdde8">{en}</span></div>'
                            for ml, en in [
                                ("ജാതകം","Jathakam (Horoscope)"),("ലഗ്നം","Lagna (Ascendant)"),
                                ("രാശി","Rasi (Moon sign)"),("നക്ഷത്രം","Nakshatra (Birth star)"),
                                ("ഭാവം","Bhavam (House)"),("ഗ്രഹം","Graham (Planet)"),
                                ("ദോഷം","Dosham (Affliction)"),("ദശ","Dasha (Period)"),
                                ("ഭഗ്യം","Bhagyam (Fortune)"),("യോഗം","Yogam (Combination)"),
                                ("ചൊവ്വ","Chovva (Mars/Kuja)"),("വ്യാഴം","Vyazham (Jupiter/Guru)"),
                            ]
                        ) + """
                        </div>
                        """)

            # ════════════════════════════════════════
            # TAB 6 — Architecture
            # ════════════════════════════════════════
            with gr.Tab("📐 Architecture"):
                gr.Markdown("""
# 🔮 AI Jathakam Reader — Architecture

## Full Pipeline

```
Upload Jathakam (PDF / Image / Screenshot)
          │
          ▼  ─────── OCRAgent (GPT-4o Vision) ───────────────────────────────
          │  • PDF: pypdf text extraction → GPT-4o vision fallback for scans
          │  • Image: GPT-4o vision with specialized astrology OCR prompt
          │  • Handles Malayalam script, Sanskrit terms, grid tables
          ▼
     Raw Extracted Text
          │
          ▼  ─────── ExtractionAgent ──────────────────────────────────────
          │  • Parses Lagna, Rasi, Nakshatra, Pada
          │  • Extracts planet house positions (1-12) for all 9 grahas
          │  • Detects dasha periods, special yogas mentioned
          │  • Outputs structured JSON for downstream agents
          ▼
   Structured JSON
    ┌─────┼──────────────────┐
    ▼     ▼                  ▼
InterpAgent  LanguageAgent  ChartRenderer
 • Personality  • Language     • South Indian
 • Career       • Term detect    SVG chart
 • Doshas      • Translation   • Planet chips
 • Yogas        • Glossary     • Lagna marker
 • Dasha        • Malayalam
 • Health
 • Relations
    │
    ▼
FAISS RAG Index
 • Personal chart data (all extractions)
 • All interpretation sections
 • Raw OCR text (chunked)
 • Vedic astrology knowledge base (2000+ facts)
 • 27 nakshatras, 12 rashis, 9 planets, 12 houses
 • Doshas, yogas, career indicators
    │
    ▼
Conversational Chat (GPT-4o-mini)
 • Retrieves relevant context per question
 • Answers with chart-specific references
 • Maintains conversation history
 • Supports Malayalam questions → English answers
```

## Agent Specifications

| Agent | Role | LLM Call |
|-------|------|----------|
| OCRAgent | PDF/image text extraction | GPT-4o vision |
| ExtractionAgent | JSON structure parsing | gpt-4o-mini |
| InterpretationAgent | 7 analysis sections | gpt-4o-mini |
| LanguageAgent | Term detection + explanation | gpt-4o-mini |
| RAG Chat Engine | Conversational Q&A | gpt-4o-mini |

## Knowledge Base (built-in, no external calls)
- 12 Rashis with elements, lords, qualities
- 9 Grahas with significations and exaltation data
- 27 Nakshatras with lords, deities, padas
- 12 Bhavas with full governance descriptions
- Dosha definitions (Mangal, Kala Sarpa, Pitru, Shani)
- Major yoga combinations (Raja, Dhana, Gaja Kesari, etc.)
- Planet-in-sign effects (exalted, own, debilitated)
- Career indicators by house
- Personality by Lagna (12 descriptions)
- Malayalam ↔ Sanskrit ↔ English glossary
                """)

        # ── Footer ────────────────────────────────────────────
        gr.HTML("""
        <div class="jh-footer">
          🔮 AI Jathakam Reader · GPT-4o Vision OCR · FAISS RAG · South Indian Chart · Multi-Agent Pipeline
          · Malayalam Support · ॐ
        </div>
        """)

        # ════════════════════════════════════════════════════════
        # EVENT BINDINGS
        # ════════════════════════════════════════════════════════

        process_btn.click(
            fn=run_pipeline,
            inputs=[file_upload, api_key_in, model_dd],
            outputs=[status_out, ocr_out, json_out, personality_out, career_out, doshas_out, yogas_out, chart_out],
        ).then(fn=lambda: pipeline_log_html(), outputs=[log_html])

        chat_btn.click(
            fn=chat_with_jathakam,
            inputs=[chat_in, chatbot],
            outputs=[chatbot, chat_in],
        )
        chat_in.submit(
            fn=chat_with_jathakam,
            inputs=[chat_in, chatbot],
            outputs=[chatbot, chat_in],
        )
        clear_chat_btn.click(fn=lambda: ([], ""), outputs=[chatbot, chat_in])

        explain_btn.click(fn=explain_term, inputs=[term_in], outputs=[term_out])

        health_btn.click(fn=lambda: get_section("health"),      outputs=[health_out])
        rel_btn.click(   fn=lambda: get_section("relationships"),outputs=[rel_out])
        dasha_btn.click( fn=lambda: get_section("dasha_analysis"),outputs=[dasha_out])

    return demo


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)