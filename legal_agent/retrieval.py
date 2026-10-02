from dataclasses import dataclass

import numpy as np

from legal_agent.documents import DocumentBlock, SourceRef, TextChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: TextChunk
    score: float


@dataclass(frozen=True)
class ContractIndex:
    chunks: tuple[TextChunk, ...]
    embeddings: np.ndarray


def chunk_blocks(
    blocks: list[DocumentBlock], max_words: int = 180, overlap_words: int = 30
) -> list[TextChunk]:
    if max_words < 1 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("Chunk size must be positive and overlap smaller than chunk size.")

    chunks: list[TextChunk] = []
    for block in blocks:
        words = block.text.split()
        if not words:
            continue
        step = max_words - overlap_words
        for start in range(0, len(words), step):
            part = words[start : start + max_words]
            if part:
                chunks.append(TextChunk(" ".join(part), (block.source,)))
            if start + max_words >= len(words):
                break
    return chunks


def normalize_embeddings(embeddings: list[list[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty two-dimensional matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("An embedding was empty and cannot be used for retrieval.")
    return matrix / norms


def rank_chunks(
    query_embedding: list[float] | np.ndarray,
    chunks: tuple[TextChunk, ...] | list[TextChunk],
    embeddings: np.ndarray,
    top_k: int = 4,
    min_score: float = 0.28,
) -> list[RetrievedChunk]:
    if not chunks or top_k < 1:
        return []
    normalized = normalize_embeddings(embeddings)
    if normalized.shape[0] != len(chunks):
        raise ValueError("The number of embeddings must match the number of chunks.")
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    query = normalize_embeddings(query)[0]
    if normalized.shape[1] != query.shape[0]:
        raise ValueError("Query and document embedding dimensions do not match.")

    scores = normalized @ query
    order = np.argsort(scores)[::-1][:top_k]
    return [
        RetrievedChunk(chunks[int(index)], float(scores[index]))
        for index in order
        if float(scores[index]) >= min_score
    ]
