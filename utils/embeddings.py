"""
Lexi-Sense Embedding Engine
Manages embedding generation and FAISS vector store with BM25 hybrid search.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import math
import re
from collections import Counter

from backend.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, TOP_K_RESULTS, SEMANTIC_WEIGHT, KEYWORD_WEIGHT
from utils.chunker import DocumentChunk


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
        return re.findall(r'\w+', text.lower())

    def fit(self, corpus: List[str]):
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avg_dl = sum(self.doc_lens) / max(self.corpus_size, 1)

        # Calculate document frequencies
        self.doc_freqs = {}
        for doc in self.tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def score(self, query: str) -> List[float]:
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
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
    """Manages FAISS index and BM25 for hybrid search over document chunks."""

    def __init__(self):
        self._model = None
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.bm25 = BM25()
        self._initialized = False

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def _create_index(self, dim: int):
        import faiss as faiss_lib
        return faiss_lib.IndexFlatIP(dim)



    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings, dtype="float32")

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to the vector store."""
        if not chunks:
            return

        texts = [c.content for c in chunks]
        embeddings = self.embed(texts)

        if self.index is None:
            self.index = self._create_index(embeddings.shape[1])

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        # Rebuild BM25 index with all chunks
        all_texts = [c.content for c in self.chunks]
        self.bm25.fit(all_texts)
        self._initialized = True

    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        semantic_weight: float = SEMANTIC_WEIGHT,
        keyword_weight: float = KEYWORD_WEIGHT,
        file_filter: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform hybrid search combining FAISS semantic search and BM25 keyword search.
        Returns list of (chunk, score) tuples sorted by combined score.
        """
        if not self._initialized or not self.chunks:
            return []

        n = len(self.chunks)

        # ── Semantic Scores ────────────────────────────────────────────
        query_embedding = self.embed([query])
        k = min(n, top_k * 3)  # retrieve more for re-ranking
        distances, indices = self.index.search(query_embedding, k)

        semantic_scores = np.zeros(n)
        for dist, idx in zip(distances[0], indices[0]):
            if idx < n:
                semantic_scores[idx] = max(0, dist)  # cosine similarity

        # Normalize semantic scores
        s_max = semantic_scores.max()
        if s_max > 0:
            semantic_scores /= s_max

        # ── Keyword Scores ─────────────────────────────────────────────
        keyword_scores = np.array(self.bm25.score(query))
        k_max = keyword_scores.max()
        if k_max > 0:
            keyword_scores /= k_max

        # ── Combined ───────────────────────────────────────────────────
        combined = semantic_weight * semantic_scores + keyword_weight * keyword_scores

        # Apply file filter
        if file_filter:
            for i, chunk in enumerate(self.chunks):
                if chunk.filename != file_filter:
                    combined[i] = 0.0

        top_indices = np.argsort(combined)[::-1][:top_k]
        results = [(self.chunks[i], float(combined[i])) for i in top_indices if combined[i] > 0]

        return results

    def get_files(self) -> List[str]:
        """Get list of unique filenames in the store."""
        return list(set(c.filename for c in self.chunks))

    def get_file_chunks(self, filename: str) -> List[DocumentChunk]:
        """Get all chunks for a specific file."""
        return [c for c in self.chunks if c.filename == filename]

    def clear(self):
        """Clear all data from the store."""
        self.index = None
        self.chunks = []
        self.bm25 = BM25()
        self._initialized = False

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
