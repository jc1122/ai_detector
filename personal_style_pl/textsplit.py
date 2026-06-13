"""Sentence-aware chunking with document IDs (avoids evaluation leakage)."""

from __future__ import annotations

from dataclasses import dataclass

from heuristic_detector import _extract_words, _split_sentences

from .config import DEFAULT_CHUNK_SENTENCES, DEFAULT_MIN_CHUNK_TOKENS


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    token_count: int
    sentence_count: int
    under_min_tokens: bool


def chunk_document(
    text: str,
    *,
    doc_id: str,
    chunk_sentences: int = DEFAULT_CHUNK_SENTENCES,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
) -> list[Chunk]:
    """Group sentences into chunks of `chunk_sentences`, merging trailing
    short groups so each chunk reaches `min_chunk_tokens` when possible."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    groups: list[list[str]] = [
        sentences[i : i + chunk_sentences]
        for i in range(0, len(sentences), chunk_sentences)
    ]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    for group in groups:
        buffer.extend(group)
        token_count = len(_extract_words(" ".join(buffer)))
        if token_count >= min_chunk_tokens:
            chunks.append(_make_chunk(doc_id, len(chunks), buffer))
            buffer = []
    if buffer:
        chunks.append(_make_chunk(doc_id, len(chunks), buffer))

    # If the only chunk is under min, keep it but flag it (caller warns).
    for chunk in chunks:
        chunk.under_min_tokens = chunk.token_count < min_chunk_tokens
    return chunks


def _make_chunk(doc_id: str, chunk_id: int, sentences: list[str]) -> Chunk:
    text = " ".join(sentences)
    token_count = len(_extract_words(text))
    return Chunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        text=text,
        token_count=token_count,
        sentence_count=len(sentences),
        under_min_tokens=False,
    )
