"""
Lexi-Sense FastAPI Server
Main API server handling file uploads, chat, streaming, and document management.
"""
import os
import uuid
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import UPLOAD_DIR, MAX_FILE_SIZE_MB
from utils.file_parser import parse_file
from utils.chunker import chunk_text
from utils.embeddings import VectorStore
from backend.agent import LexiSenseAgent

logger = logging.getLogger("lexi-sense")

# ═══════════════════════════════════════════════════════════════════════
#  APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Lexi-Sense API",
    description="Autonomous Document Intelligence Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ── Global State (session-based) ───────────────────────────────────────
sessions: Dict[str, dict] = {}


def get_or_create_session(session_id: Optional[str] = None) -> dict:
    """Get or create a user session with its own vector store and agent."""
    if session_id and session_id in sessions:
        return sessions[session_id]

    sid = session_id or str(uuid.uuid4())
    vector_store = VectorStore()
    agent = LexiSenseAgent(vector_store)

    sessions[sid] = {
        "id": sid,
        "vector_store": vector_store,
        "agent": agent,
        "files": {},
        "created_at": datetime.utcnow().isoformat(),
    }
    return sessions[sid]


# ═══════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    query: str
    session_id: str = ""
    file_filter: Optional[str] = None
    stream: bool = True


class SummaryRequest(BaseModel):
    filename: str
    session_id: str


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: list = []


# ═══════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Lexi-Sense API is running", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ── File Upload ────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    session_id: str = Query(default=""),
):
    """Upload and process one or more documents."""
    session = get_or_create_session(session_id or None)
    results = []

    for file in files:
        ext = Path(file.filename).suffix.lower()

        # Read and validate size
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"File too large ({size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB",
            })
            continue

        # Save file temporarily
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(content)

        try:
            # Parse the document
            parsed = parse_file(str(file_path))

            # Chunk the content
            chunks = chunk_text(
                parsed.content,
                filename=parsed.filename,
                file_type=parsed.file_type,
            )

            # Add to vector store (embedding happens here)
            session["vector_store"].add_chunks(chunks)

            # Store file metadata
            session["files"][file.filename] = {
                "filename": file.filename,
                "file_type": parsed.file_type,
                "word_count": parsed.word_count,
                "page_count": parsed.page_count,
                "chunk_count": len(chunks),
                "size_mb": round(size_mb, 2),
                "uploaded_at": datetime.utcnow().isoformat(),
                "metadata": parsed.metadata,
            }

            results.append({
                "filename": file.filename,
                "status": "success",
                "file_type": parsed.file_type,
                "word_count": parsed.word_count,
                "page_count": parsed.page_count,
                "chunk_count": len(chunks),
                "size_mb": round(size_mb, 2),
            })

        except Exception as e:
            logger.error(f"Upload error for {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e),
            })
        finally:
            # Clean up temp file
            if file_path.exists():
                os.remove(file_path)

    return {
        "session_id": session["id"],
        "results": results,
        "total_chunks": session["vector_store"].total_chunks,
        "total_files": len(session["files"]),
    }


# ── Chat ───────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send a message to the agent."""
    session = get_or_create_session(request.session_id or None)
    agent: LexiSenseAgent = session["agent"]

    if request.stream:
        async def stream_generator():
            try:
                async for chunk in agent.chat_stream(request.query, file_filter=request.file_filter):
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: {json.dumps({'text': f'⚠️ Error: {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'session_id': session['id']})}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-Id": session["id"],
            },
        )
    else:
        response = agent.chat(request.query, file_filter=request.file_filter)
        return ChatResponse(
            response=response,
            session_id=session["id"],
        )


# ── Summary (POST to avoid filename-in-URL issues) ────────────────────
@app.post("/api/summary")
async def get_summary(request: SummaryRequest):
    """Generate an AI summary of a specific document."""
    session = get_or_create_session(request.session_id)
    agent: LexiSenseAgent = session["agent"]

    if request.filename not in session["files"]:
        raise HTTPException(status_code=404, detail=f"File '{request.filename}' not found in session")

    try:
        summary = agent.generate_summary(request.filename)
        return {
            "filename": request.filename,
            "summary": summary,
            "session_id": session["id"],
        }
    except Exception as e:
        logger.error(f"Summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Session Info ───────────────────────────────────────────────────────
@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information including uploaded files."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    return {
        "session_id": session["id"],
        "files": list(session["files"].values()),
        "total_chunks": session["vector_store"].total_chunks,
        "memory_length": len(session["agent"].memory),
        "created_at": session["created_at"],
    }


# ── Chat History ───────────────────────────────────────────────────────
@app.get("/api/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get conversation history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    agent: LexiSenseAgent = sessions[session_id]["agent"]
    return {"history": agent.get_memory_context(), "session_id": session_id}


# ── Clear Session ──────────────────────────────────────────────────────
@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session and all its data."""
    if session_id in sessions:
        sessions[session_id]["vector_store"].clear()
        sessions[session_id]["agent"].clear_memory()
        del sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


# ── File List ──────────────────────────────────────────────────────────
@app.get("/api/files/{session_id}")
async def get_files(session_id: str):
    """Get list of uploaded files in a session."""
    if session_id not in sessions:
        return {"files": []}

    return {"files": list(sessions[session_id]["files"].values())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
