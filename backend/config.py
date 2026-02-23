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
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── LLM Configuration ─────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
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
MAX_FILE_SIZE_MB = 25
# ALLOWED_EXTENSIONS removed – all file types are now handled by the universal parser
