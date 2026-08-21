"""
retrieval.py

Loads chunked passages from chunks.json, embeds them with a multilingual
sentence-transformer model, builds a FAISS index for fast similarity search,
and exposes a search() function to retrieve the most relevant chunks for a
given query. Supports optional filtering by language.
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "chunks.json"
INDEX_PATH = "chunks.index"
MODEL_NAME = "intfloat/multilingual-e5-small"

_model = None
_chunks = None
_index = None


def _load_chunks(path=CHUNKS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks:
        raise ValueError(f"No chunks found in {path}")
    return chunks


def _get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_index(chunks_path=CHUNKS_PATH, index_path=INDEX_PATH):
    chunks = _load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]

    model = _get_model()
    prefixed_texts = [f"passage: {text}" for text in texts]
    print(f"Embedding {len(prefixed_texts)} chunks...")
    embeddings = model.encode(
        prefixed_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    print(f"Saved FAISS index with {index.ntotal} vectors to {index_path}")

    return index, chunks


def _load_index(chunks_path=CHUNKS_PATH, index_path=INDEX_PATH):
    global _index, _chunks

    if _chunks is None:
        _chunks = _load_chunks(chunks_path)

    if _index is None:
        if os.path.exists(index_path):
            _index = faiss.read_index(index_path)
        else:
            print(f"{index_path} not found, building it now...")
            _index, _chunks = build_index(chunks_path, index_path)

    return _index, _chunks


def search(query_text, top_k=5, language=None):
    """
    Embeds query_text and returns the top_k most similar chunks.

    language: optional string like "hindi", "marathi", "english" — if given,
    restricts results to chunks whose "language" field matches. Over-fetches
    a larger candidate pool from FAISS first (filtering happens after the
    search, on the Python side), then falls back to unfiltered top_k if
    fewer than top_k same-language matches turn up in that pool.

    Returns a list of dicts with the original chunk fields plus "score".
    """
    index, chunks = _load_index()
    model = _get_model()

    import time as _time  # temporary, remove after diagnosing latency

    t_encode_start = _time.perf_counter()
    query_embedding = model.encode(
        [f"query: {query_text}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    t_encode_end = _time.perf_counter()

    fetch_k = top_k * 10 if language else top_k
    fetch_k = min(fetch_k, index.ntotal)

    t_faiss_start = _time.perf_counter()
    scores, indices = index.search(query_embedding, fetch_k)
    t_faiss_end = _time.perf_counter()

    print(f"[TIMING] embed: {(t_encode_end - t_encode_start)*1000:.1f}ms | "
          f"FAISS search: {(t_faiss_end - t_faiss_start)*1000:.1f}ms")

    all_results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        all_results.append(chunk)

    if language:
        filtered = [c for c in all_results if c.get("language") == language]
        if filtered:
            return filtered[:top_k]
        # No same-language matches in the fetched pool — fall back rather
        # than returning nothing.
        return all_results[:top_k]

    return all_results[:top_k]


if __name__ == "__main__":
    build_index()

    test_query = "अंतरिक्ष अनुसंधान क्यों महत्वपूर्ण है?"
    print(f"\nTest query: {test_query}\n")
    for r in search(test_query, top_k=3, language="hindi"):
        print(f"[{r['score']:.4f}] ({r['method']}, {r['source_id']}) {r['text'][:100]}...")