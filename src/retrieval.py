"""
retrieval.py

Loads chunked passages from chunks.json, embeds them with a multilingual
sentence-transformer model, builds a FAISS index for fast similarity search,
and exposes a search() function to retrieve the most relevant chunks for a
given query.
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "chunks.json"
INDEX_PATH = "chunks.index"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Cached globals so the model/index/chunks are only loaded once per process
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
    """
    Embeds every chunk's "text" field, builds a FAISS IndexFlatIP index
    over the (normalized) embeddings, and saves it to disk.
    """
    chunks = _load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]

    model = _get_model()
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    print(f"Saved FAISS index with {index.ntotal} vectors to {index_path}")

    return index, chunks


def _load_index(chunks_path=CHUNKS_PATH, index_path=INDEX_PATH):
    """
    Loads the FAISS index and chunk metadata from disk, building the index
    first if it doesn't exist yet.
    """
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


def search(query_text, top_k=5):
    """
    Embeds query_text and returns the top_k most similar chunks.

    Returns a list of dicts, each containing the original chunk fields
    ("text", "method", "source_id") plus a "score" (cosine similarity,
    higher = more similar).
    """
    index, chunks = _load_index()
    model = _get_model()

    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)

    return results


if __name__ == "__main__":
    build_index()

    test_query = "अंतरिक्ष अनुसंधान क्यों महत्वपूर्ण है?"
    print(f"\nTest query: {test_query}\n")
    for r in search(test_query, top_k=3):
        print(f"[{r['score']:.4f}] ({r['method']}, {r['source_id']}) {r['text'][:100]}...")