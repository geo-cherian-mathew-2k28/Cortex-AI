<p align="center">
  <img src="https://img.shields.io/badge/Cortex_AI-Multi--Modal_Intelligence-6366f1?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0MCA0MCIgZmlsbD0ibm9uZSI+PHJlY3Qgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiByeD0iMTAiIGZpbGw9IiM2MzY2ZjEiLz48cGF0aCBkPSJNMjAgOEwzMCAxNHYxMmwtMTAgNi0xMC02VjE0bDEwLTZ6IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjEuOCIvPjxjaXJjbGUgY3g9IjIwIiBjeT0iMjAiIHI9IjUiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMS44Ii8+PC9zdmc+&logoColor=white" alt="Cortex AI" />
</p>

<h1 align="center">⬡ Cortex AI</h1>
<h2 align="center">Enterprise-Grade Multi-Modal Document & Media Intelligence</h2>

<p align="center">
  <em>Developed by <strong>Geo Cherian Mathew</strong></em>
</p>

---

## 🌟 The Future of Document Interaction

**Cortex AI** is an industry-grade, multi-modal reasoning engine that goes beyond simple text. It's your personal intelligent workspace that can **read** your documents, **see** your images, and **hear** your videos.

Built for scale and precision, Cortex AI handles everything from complex Excel spreadsheets and PowerPoint decks to high-resolution images and long-form video content.

### 🚀 Industrial-Strength Capabilities

- **👁️ Visual Intelligence:** Leverages **Llama 3.2 Vision** to describe images, extract handwritten text, and analyze visual data.
- **👂 Audio & Video Intelligence:** Transcribes and analyzes videos and audio recordings using **Whisper**, providing searchable text for every spoken word.
- **📄 Complete Document Support:** Native parsing for PDF, Word, Excel, PowerPoint, CSV, JSON, Markdown, and various Code files.
- **🤖 Autonomous Reasoning:** Uses a "Chain-of-Reasoning" agent that decides when to use tools, calculators, or cross-document comparisons.
- **🛡️ Data Integrity:** Strict retrieval-augmented generation (RAG) ensures answers are 100% grounded in your data—**zero hallucinations**.

---

## ⚡ Quick Start Guide

### 1. 📥 Installation
Clone the repository and install the requirements:
```bash
git clone https://github.com/geo-cherian-mathew-2k28/Cortex-AI.git
cd Cortex-AI
pip install -r requirements.txt
```

### 2. 🔑 API Configuration
Cortex AI requires two API keys for its high-speed reasoning and cloud deployment:
1. **Groq API Key**: Get it at [Groq Console](https://console.groq.com).
2. **HuggingFace Token**: Get a free one at [HuggingFace Settings](https://huggingface.co/settings/tokens).
3. Rename `.env.example` to `.env`.
4. Paste your keys:
   - `GROQ_API_KEY=gsk_...`
   - `HUGGINGFACE_TOKEN=hf_...`

### 3. 🚀 Launch
Start the production-ready server:
```bash
python -m uvicorn backend.server:app --reload
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## ✨ Features

| Feature | Description |
|---|---|
| **🖼️ Image OCR & Analysis** | Drop a photo of a receipt or a diagram; Cortex AI explains it. |
| **🎬 Video Transcription** | Upload an MP4; search for specific quotes or get a visual summary. |
| **📊 Excel/CSV Deep Analytics** | Automatically calculates statistics and extracts tables from huge data sets. |
| **🧠 Intelligent Comparison** | Compare a video's transcript against a PDF contract for discrepancies. |
| **📋 Summary Extraction** | One-click summaries for 50+ page documents or hour-long media. |
| **📱 Mobile Optimized** | Fully responsive UX for seamless tablet/phone interaction. |

---

## 📁 Technical Architecture

- **Engine:** FastAPI (Python 3.9+)
- **Models:** Groq Llama 3.3 (Reasoning), Llama 3.2 (Vision), Whisper (Audio).
- **Storage:** FAISS Vector Indexing with BM25 Hybrid Search.
- **Logic:** Custom autonomous agent with tool-calling and memory.

---

## 🤝 Support & Contribution

Created and maintained by **Geo Cherian Mathew**. 
If this project empowers your workflow, please consider giving it a ⭐ on GitHub!

---

<p align="center">
  <strong>⬡ Cortex AI</strong> — Empowering Minds through Multi-Modal Intelligence.
</p>
