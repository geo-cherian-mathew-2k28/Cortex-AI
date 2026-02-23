"""
Lexi-Sense Configuration Module
Centralized configuration management with environment variable support.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
# On Vercel, the filesystem is read-only except for /tmp
if os.getenv("VERCEL"):
    UPLOAD_DIR = Path("/tmp") / "uploads"
else:
    UPLOAD_DIR = BASE_DIR / "uploads"

# Moved mkdir into the upload logic to prevent import-time errors on certain platforms
def ensure_upload_dir():
    UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# ── API Keys ───────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── LLM Configuration ─────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = "llama-3.2-11b-vision-preview"
TRANSCRIPTION_MODEL = "whisper-large-v3"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096

# ── Embedding Configuration ───────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── Chunking Configuration ────────────────────────────────────────────
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# ── Retrieval Configuration ───────────────────────────────────────────
TOP_K_RESULTS = 8
SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

# ── Session Configuration ─────────────────────────────────────────────
MAX_MEMORY_MESSAGES = 15
SESSION_TIMEOUT_MINUTES = 60

# ── File Limits ────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 4.5 if os.getenv("VERCEL") else 25
# ALLOWED_EXTENSIONS removed – all file types are now handled by the universal parser
