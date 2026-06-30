# ⚡ DocGen AI — AI Documentation Generator

> Instantly generate production-quality API docs, README files, and architecture summaries from any codebase — powered by GPT 4o.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-4.44%2B-orange?style=flat-square)
![OpenAI](https://img.shields.io/badge/GPT-4o-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🚀 Features

- **📄 API Documentation** — Full function/method/class docs with params, return types, exceptions, and examples
- **📖 README Generator and ppt** — Professional README with install steps, usage, structure, and badges
- **🏛 Architecture Summaries** — High-level system overview, component breakdown, data flow, and design patterns
- **⚡ Streaming Output** — See documentation generate in real time
- **📂 File Upload** — Paste code or upload `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, and more
- **⬇️ Download** — Export your docs as `.md` with one click
- **🎨 Dark Cyberpunk UI** — Slick interface built with custom Gradio CSS

---

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| Frontend/UI | Gradio 4.x |
| AI Backend | OpenAI GPT 4o |
| Language | Python 3.9+ |
| Output Format | Markdown |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/docgen-ai.git
cd docgen-ai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

```bash
export OpenAI API key="sk-..."     # macOS/Linux
set OpenAI API key=sk-...          # Windows CMD
$env:OpenAI API key="sk-..."       # Windows PowerShell
```

### 5. Launch the app

```bash
python app.py
```

Open your browser at **http://localhost:7860**

---

## 🖥 Usage

### Option A — Paste Code
1. Click the **"✏️ Paste Code"** tab
2. Paste your source code (any language)
3. Optionally click a sample button (🐍 Python · 🟨 JS · 🔵 Go)

### Option B — Upload a File
1. Click the **"📂 Upload File"** tab
2. Drop or browse to your source file

### Generate Docs
1. Choose a **Documentation Type**:
   - `📄 API Docs` — function/method reference
   - `📖 README.md` — project readme
   - `🏛 Architecture` — system design summary
2. (Optional) Add extra context in the text box
3. Click **⚡ Generate Documentation**
4. View the result in **Preview** or **Raw Markdown** tabs
5. Click **⬇️ Download .md** to save

---

## 📁 Project Structure

```
project/
├── app.py              # Main Gradio app + OpenAI API integration
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── demo_screenshots/   # UI screenshots
│   ├── main_ui.png
│   ├── api_docs_output.png
│   └── architecture_output.png
└── sample_data/        # Example source files
    ├── auth.py
    ├── event_emitter.js
    └── lru_cache.go
```

---

## 🔧 Configuration

| Environment Variable | Description | Required |
|---|---|---|
| `OpenAI API key` | Your OpenAI API key | ✅ Yes |

The app uses **GPT-4o-4-5** by default. To change the model, edit `MODEL` in `app.py`.

---

## 🧩 Supported File Types

`.py` · `.js` · `.ts` · `.go` · `.rs` · `.java` · `.rb` · `.cpp` · `.c` · `.cs` · `.php` · `.kt` · `.swift` · `.txt` · `.md`

---

## 📸 Screenshots

| Input Panel | API Docs Output | Architecture Output |
|---|---|---|
| *(see demo_screenshots/)* | *(see demo_screenshots/)* | *(see demo_screenshots/)* |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit your changes: `git commit -m 'feat: add my feature'`
4. Push to the branch: `git push origin feat/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT © 2025 DocGen AI Contributors
