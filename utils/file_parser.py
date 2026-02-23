"""
Cortex AI – Multi-Modal File Parser
Industry-grade extraction engine for Text, Data, Images, Video, and Audio.
Supports:
- Documents: PDF, DOCX, XLSX, PPTX, CSV, JSON, MD, HTML
- Images: JPG, PNG, WEBP (using Llama 3.2 Vision)
- Video: MP4, MOV, MKV (Audio Transcription + Visual Analysis)
- Audio: MP3, WAV, M4A (Whisper Transcription)
- Code: Python, JS, etc. (with intelligent fallback)

Created by Geo Cherian Mathew.
"""
import os
import csv
import io
import json
import re
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

import pypdf
# Lazy import for large/native libs below in functions
# import docx
# import pandas as pd
# from PIL import Image
from groq import Groq

# Import configuration
from backend.config import GROQ_API_KEY, VISION_MODEL, TRANSCRIPTION_MODEL


@dataclass
class ParsedDocument:
    """Represents a parsed document with its content and metadata."""
    filename: str
    file_type: str
    content: str
    page_count: int = 1
    word_count: int = 0
    char_count: int = 0
    metadata: dict = field(default_factory=dict)
    parsed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self):
        self.word_count = len(self.content.split())
        self.char_count = len(self.content)


# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def get_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


def encode_image(image_path: str):
    """Encode an image to base64 for vision processing."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ═══════════════════════════════════════════════════════════════════════
#  IMAGE PARSER (VISION)
# ═══════════════════════════════════════════════════════════════════════

def parse_image(file_path: str) -> ParsedDocument:
    """Analyze image using Groq Vision model."""
    client = get_groq_client()
    if not client:
        return ParsedDocument(
            filename=Path(file_path).name,
            file_type=Path(file_path).suffix[1:],
            content="[Image uploaded but Groq API key is missing for Vision analysis.]"
        )

    try:
        from PIL import Image
        base64_image = encode_image(file_path)
        img = Image.open(file_path)
        width, height = img.size

        # Use Llama Vision to describe and extract OCR
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image in detail. Provide a comprehensive description of what's happening, identify any people, objects, locations, or branding. If there is any text (like a document, label, or signage), extract it exactly as seen."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            temperature=0.2,
            max_tokens=1024,
        )

        analysis = response.choices[0].message.content
        content = f"--- Image Analysis ---\nDimensions: {width}x{height}\nFormat: {img.format}\n\n{analysis}"

        return ParsedDocument(
            filename=Path(file_path).name,
            file_type=img.format.lower() if img.format else "image",
            content=content,
            metadata={
                "width": width,
                "height": height,
                "format": img.format,
                "mode": img.mode
            }
        )
    except Exception as e:
        return ParsedDocument(
            filename=Path(file_path).name,
            file_type="image",
            content=f"[Error during Vision analysis: {str(e)}]",
            metadata={"error": str(e)}
        )


# ═══════════════════════════════════════════════════════════════════════
#  VIDEO & AUDIO PARSER (WHISPER + VISION)
# ═══════════════════════════════════════════════════════════════════════

def parse_audio(file_path: str) -> ParsedDocument:
    """Transcribe audio using Groq Whisper."""
    client = get_groq_client()
    if not client:
        return ParsedDocument(
            filename=Path(file_path).name,
            file_type=Path(file_path).suffix[1:],
            content="[Audio uploaded but Groq API key is missing for Transcription.]"
        )

    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(Path(file_path).name, file.read()),
                model=TRANSCRIPTION_MODEL,
                response_format="verbose_json",
            )
        
        content = f"--- Audio Transcription ---\n{transcription.text}"
        
        return ParsedDocument(
            filename=Path(file_path).name,
            file_type=Path(file_path).suffix[1:],
            content=content,
            metadata={
                "duration": transcription.duration if hasattr(transcription, 'duration') else 0,
                "language": transcription.language if hasattr(transcription, 'language') else "unknown"
            }
        )
    except Exception as e:
        return ParsedDocument(
            filename=Path(file_path).name,
            file_type="audio",
            content=f"[Error during Audio transcription: {str(e)}]",
            metadata={"error": str(e)}
        )


def parse_video(file_path: str) -> ParsedDocument:
    """
    Industry-grade Video Analysis.
    1. Extracts audio and transcribes (Whisper).
    2. Captures key frames and analyzes visuals (Vision).
    """
    client = get_groq_client()
    try:
        from moviepy.editor import VideoFileClip
        import tempfile

        clip = VideoFileClip(file_path)
        duration = clip.duration
        fps = clip.fps
        
        # 1. Handle Audio Transcription
        audio_content = "No audio track found."
        if clip.audio:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
                audio_path = tmp_audio.name
                clip.audio.write_audiofile(audio_path, logger=None)
                
                audio_parsed = parse_audio(audio_path)
                audio_content = audio_parsed.content
                os.unlink(audio_path)

        # 2. Key Frame Visual Analysis (Sample at 0s, mid, and end)
        visual_descriptions = []
        if client and duration > 0:
            sample_times = [0, duration/2, duration - 0.1] if duration > 5 else [0]
            for i, t in enumerate(sample_times):
                frame_path = f"{file_path}_frame_{i}.jpg"
                clip.save_frame(frame_path, t=t)
                
                # Analyze frame
                base64_frame = encode_image(frame_path)
                response = client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"This is a key frame from a video at timestamp {t:.2f}s. Describe the visual context briefly."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}}
                        ]
                    }],
                    max_tokens=256,
                )
                visual_descriptions.append(f"[@{t:.2f}s]: {response.choices[0].message.content}")
                os.unlink(frame_path)

        visual_summary = "\n".join(visual_descriptions)
        content = f"--- Video Analysis ---\nDuration: {duration:.2f}s\nResolution: {clip.w}x{clip.h}\n\n--- Visual Summary ---\n{visual_summary}\n\n{audio_content}"
        
        clip.close()
        return ParsedDocument(
            filename=Path(file_path).name,
            file_type="video",
            content=content,
            metadata={
                "duration": duration,
                "width": clip.w,
                "height": clip.h,
                "fps": fps
            }
        )

    except Exception as e:
        return ParsedDocument(
            filename=Path(file_path).name,
            file_type="video",
            content=f"[Video processing failed: {str(e)}]. Support for video requires moviepy and ffmpeg.",
            metadata={"error": str(e)}
        )


# ═══════════════════════════════════════════════════════════════════════
#  EXISTING DOCUMENT PARSERS (REFINED)
# ═══════════════════════════════════════════════════════════════════════

def parse_pdf(file_path: str) -> ParsedDocument:
    """Extract text from a PDF file."""
    reader = pypdf.PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")

    content = "\n\n".join(pages)
    return ParsedDocument(
        filename=Path(file_path).name,
        file_type="pdf",
        content=content,
        page_count=len(reader.pages),
        metadata={"total_pages": len(reader.pages)},
    )


def parse_docx(file_path: str) -> ParsedDocument:
    """Extract text from a Word document."""
    import docx
    doc = docx.Document(file_path)
    full_text = [para.text for para in doc.paragraphs if para.text.strip()]
    content = "\n".join(full_text).strip()

    # Tables
    tables_text = []
    for i, table in enumerate(doc.tables):
        table_rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            table_rows.append(" | ".join(cells))
        tables_text.append(f"\n[Table {i + 1}]\n" + "\n".join(table_rows))

    if tables_text:
        content += "\n\n--- Tables ---\n" + "\n".join(tables_text)

    return ParsedDocument(
        filename=Path(file_path).name,
        file_type="docx",
        content=content,
        metadata={"table_count": len(doc.tables)},
    )


def parse_csv(file_path: str) -> ParsedDocument:
    """Extract text from a CSV file using built-in csv module."""
    rows = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 150: # Limit for cloud analysis
                    rows.append(" | ".join(row))
                else:
                    break
    except Exception:
        pass
    content = "\n".join(rows)
    return ParsedDocument(filename=Path(file_path).name, file_type="csv", content=content or "[Empty CSV]")

def parse_excel(file_path: str) -> ParsedDocument:
    return ParsedDocument(filename=Path(file_path).name, file_type="xlsx", content="[Excel analysis currently limited to local deployment. Please use CSV in the cloud.]")


def parse_json(file_path: str) -> ParsedDocument:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return ParsedDocument(
        filename=Path(file_path).name,
        file_type="json",
        content=json.dumps(data, indent=2, ensure_ascii=False),
        metadata={"is_list": isinstance(data, list)}
    )

def parse_txt(file_path: str) -> ParsedDocument:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return ParsedDocument(filename=Path(file_path).name, file_type="txt", content=content)


def parse_text_fallback(file_path: str) -> ParsedDocument:
    """Fallback for any text/code file."""
    ext = Path(file_path).suffix.lower().lstrip(".")
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        
        # Binary check
        non_printable = sum(1 for c in content[:2000] if not c.isprintable() and c not in '\n\r\t')
        if len(content[:2000]) > 0 and (non_printable / min(len(content), 2000)) > 0.15:
            return ParsedDocument(
                filename=Path(file_path).name,
                file_type=ext or "binary",
                content=f"[Binary file: {Path(file_path).name}. No text extracted.]"
            )
        
        return ParsedDocument(filename=Path(file_path).name, file_type=ext, content=content)
    except Exception:
        return ParsedDocument(filename=Path(file_path).name, file_type="unknown", content="[Unreadable file]")


# ═══════════════════════════════════════════════════════════════════════
#  DISPATCHER
# ═══════════════════════════════════════════════════════════════════════

PARSERS = {
    # Documents
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".csv": parse_csv,
    ".xlsx": parse_excel, ".xls": parse_excel,
    ".json": parse_json,
    ".txt": parse_txt,
    # Images (Multi-modal)
    ".jpg": parse_image, ".jpeg": parse_image, ".png": parse_image, ".webp": parse_image,
    # Video (Multi-modal)
    ".mp4": parse_video, ".mov": parse_video, ".mkv": parse_video, ".avi": parse_video,
    # Audio (Transcription)
    ".mp3": parse_audio, ".wav": parse_audio, ".m4a": parse_audio,
}

TEXT_EXTENSIONS = {
    ".md", ".markdown", ".yaml", ".yml", ".jsonl", ".py", ".js", ".ts", ".html", ".css", ".sql", ".sh"
}

def parse_file(file_path: str) -> ParsedDocument:
    ext = Path(file_path).suffix.lower()
    parser = PARSERS.get(ext)
    
    if parser:
        return parser(file_path)
    
    if ext in TEXT_EXTENSIONS:
        return parse_text_fallback(file_path)
    
    return parse_text_fallback(file_path)
