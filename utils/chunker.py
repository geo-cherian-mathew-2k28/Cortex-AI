"""
Lexi-Sense Text Chunker
Splits documents into overlapping chunks while preserving semantic boundaries.
Each chunk carries metadata for source tracking.
"""
from typing import List
from dataclasses import dataclass, field
import re
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class DocumentChunk:
    """A chunk of document text with source metadata."""
    content: str
    chunk_index: int
    filename: str
    file_type: str
    page_info: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return len(self.content.split())


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving structure."""
    # Split on sentence boundaries but keep some structure
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def _detect_page(text: str) -> str:
    """Try to detect page info from text markers like [Page N]."""
    match = re.search(r'\[Page (\d+)\]', text)
    return f"Page {match.group(1)}" if match else ""


def chunk_text(
    text: str,
    filename: str,
    file_type: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """
    Split text into overlapping chunks using a recursive approach.
    Tries to split on paragraph boundaries first, then sentences, then words.
    """
    if not text.strip():
        return []

    chunks: List[DocumentChunk] = []

    # First split by paragraphs (double newline)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    current_chunk_parts = []
    current_word_count = 0

    for para in paragraphs:
        para_word_count = len(para.split())

        # If a single paragraph exceeds chunk_size, split it further
        if para_word_count > chunk_size:
            # Flush current buffer first
            if current_chunk_parts:
                chunk_text_content = "\n\n".join(current_chunk_parts)
                chunks.append(DocumentChunk(
                    content=chunk_text_content,
                    chunk_index=len(chunks),
                    filename=filename,
                    file_type=file_type,
                    page_info=_detect_page(chunk_text_content),
                ))
                # Keep overlap
                overlap_parts = _get_overlap_parts(current_chunk_parts, chunk_overlap)
                current_chunk_parts = overlap_parts
                current_word_count = sum(len(p.split()) for p in current_chunk_parts)

            # Split long paragraph by sentences
            sentences = _split_into_sentences(para)
            for sentence in sentences:
                s_words = len(sentence.split())
                if current_word_count + s_words > chunk_size and current_chunk_parts:
                    chunk_text_content = "\n\n".join(current_chunk_parts)
                    chunks.append(DocumentChunk(
                        content=chunk_text_content,
                        chunk_index=len(chunks),
                        filename=filename,
                        file_type=file_type,
                        page_info=_detect_page(chunk_text_content),
                    ))
                    overlap_parts = _get_overlap_parts(current_chunk_parts, chunk_overlap)
                    current_chunk_parts = overlap_parts
                    current_word_count = sum(len(p.split()) for p in current_chunk_parts)

                current_chunk_parts.append(sentence)
                current_word_count += s_words
        else:
            if current_word_count + para_word_count > chunk_size and current_chunk_parts:
                chunk_text_content = "\n\n".join(current_chunk_parts)
                chunks.append(DocumentChunk(
                    content=chunk_text_content,
                    chunk_index=len(chunks),
                    filename=filename,
                    file_type=file_type,
                    page_info=_detect_page(chunk_text_content),
                ))
                overlap_parts = _get_overlap_parts(current_chunk_parts, chunk_overlap)
                current_chunk_parts = overlap_parts
                current_word_count = sum(len(p.split()) for p in current_chunk_parts)

            current_chunk_parts.append(para)
            current_word_count += para_word_count

    # Flush remaining
    if current_chunk_parts:
        chunk_text_content = "\n\n".join(current_chunk_parts)
        chunks.append(DocumentChunk(
            content=chunk_text_content,
            chunk_index=len(chunks),
            filename=filename,
            file_type=file_type,
            page_info=_detect_page(chunk_text_content),
        ))

    return chunks


def _get_overlap_parts(parts: List[str], overlap_words: int) -> List[str]:
    """Get the tail parts that contain roughly overlap_words words."""
    result = []
    word_count = 0
    for part in reversed(parts):
        pw = len(part.split())
        if word_count + pw > overlap_words and result:
            break
        result.insert(0, part)
        word_count += pw
    return result
