"""
Cortex AI Embedding Engine (Vercel-Optimized)
Lightweight hybrid search using API-based embeddings and Numpy.
Removes dependency on Torch/Sentence-Transformers to fit Lambda limits.
Created by Geo Cherian Mathew.
"""
import numpy as np
import os
import requests
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import math
import re

from backend.config import EMBEDDING_MODEL, TOP_K_RESULTS, SEMANTIC_WEIGHT, KEYWORD_WEIGHT, GROQ_API_KEY
from utils.chunker import DocumentChunk

# Constants
# Use the full model path for HuggingFace
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL_ID}"

class BM25:
    """Simple BM25 implementation for keyword search."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_dl = 0
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lens: List[int] = []
        self.tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        # Handle empty/none
        if not text: return []
        return re.findall(r'\w+', text.lower())

    def fit(self, corpus: List[str]):
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avg_dl = sum(self.doc_lens) / max(self.corpus_size, 1)
        self.doc_freqs = {}
        for doc in self.tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def score(self, query: str) -> List[float]:
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        if self.corpus_size == 0 or not query_tokens: return scores
        for token in query_tokens:
            if token not in self.doc_freqs: continue
            df = self.doc_freqs[token]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
            for i, doc in enumerate(self.tokenized_corpus):
                tf = doc.count(token)
                dl = self.doc_lens[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[i] += idf * (numerator / denominator)
        return scores

class VectorStore:
    """Session-based vector storage using Numpy for similarity."""
    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[DocumentChunk] = []
        self.bm25 = BM25()
        self._initialized = False
        # Get token from environment
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN", "")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings via HuggingFace Inference API or fallback."""
        # Standard model dimension for all-MiniLM-L6-v2 is 384
        dim = 384
        
        if not self.hf_token or not texts:
            return np.zeros((len(texts), dim), dtype="float32")

        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        for _ in range(3):
            try:
                response = requests.post(
                    HF_API_URL, 
                    headers=headers, 
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=8.0  # Keep it under Vercel's limit
                )
                if response.status_code == 200:
                    data = response.json()
                    # HF API returns a list of floats (for 1 item) or list of lists (for multiple)
                    # We need to ensure we return a 2D array
                    res = np.array(data, dtype="float32")
                    if res.ndim == 1:
                        res = res.reshape(1, -1)
                    return res
                elif response.status_code == 503: # Model loading
                    time.sleep(1.5)
                    continue
                else:
                    break
            except Exception:
                break
                
        return np.zeros((len(texts), dim), dtype="float32")

    def add_chunks(self, chunks: List[DocumentChunk]):
        if not chunks: return
        texts = [c.content for c in chunks]
        new_embeddings = self.embed(texts)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.chunks.extend(chunks)
        all_texts = [c.content for c in self.chunks]
        self.bm25.fit(all_texts)
        self._initialized = True

    def hybrid_search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Tuple[DocumentChunk, float]]:
        if not self._initialized or not self.chunks: return []
        n = len(self.chunks)
        
        # ── Semantic Scores (Cosine Similarity via Numpy) ────────────────
        query_emb = self.embed([query])[0]
        # Dot product of normalized vectors
        norm_query = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        norm_embeddings = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
        semantic_scores = np.dot(norm_embeddings, norm_query)
        
        s_max = semantic_scores.max()
        if s_max > 0: semantic_scores /= s_max

        # ── Keyword Scores ─────────────────────────────────────────────
        keyword_scores = np.array(self.bm25.score(query))
        k_max = keyword_scores.max()
        if k_max > 0: keyword_scores /= k_max

        # ── Combined ───────────────────────────────────────────────────
        combined = SEMANTIC_WEIGHT * semantic_scores + KEYWORD_WEIGHT * keyword_scores
        top_indices = np.argsort(combined)[::-1][:top_k]
        return [(self.chunks[i], float(combined[i])) for i in top_indices if combined[i] > 0]

    def clear(self):
        self.embeddings = None
        self.chunks = []
        self.bm25 = BM25()
        self._initialized = False

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
