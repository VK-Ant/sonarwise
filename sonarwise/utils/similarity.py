"""Similarity computation utilities."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def cosine_similarity_batch(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and a batch of embeddings.

    Args:
        query: 1D vector (embedding_dim,)
        embeddings: 2D array (n_embeddings, embedding_dim)

    Returns:
        1D array of similarity scores (n_embeddings,)
    """
    query = np.asarray(query, dtype=np.float32).flatten()
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(embeddings), dtype=np.float32)

    query_normalized = query / query_norm
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norms = np.where(emb_norms == 0, 1.0, emb_norms)
    embeddings_normalized = embeddings / emb_norms

    similarities = embeddings_normalized @ query_normalized
    return similarities.astype(np.float32)
