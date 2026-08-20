"""
chunking.py
Three chunking strategies for HHGOA-SonicQuery:
1. fixed_size_chunking      - word-window chunks with overlap
2. semantic_chunking        - groups sentences using sentence-transformer embeddings
3. metadata_aware_chunking  - treats each original passage as its own chunk
"""

import re
import json
import numpy as np
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import Dataset
from sentence_transformers import SentenceTransformer

# ---------- 1. LOAD DATASET (same safe method from before) ----------

def load_data(rows_needed=5000, filename="train/hintrain.parquet"):
    file_path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        filename=filename,
        repo_type="dataset"
    )
    parquet_file = pq.ParquetFile(file_path)
    collected_batches = []
    rows_so_far = 0
    for batch in parquet_file.iter_batches(batch_size=500):
        collected_batches.append(batch)
        rows_so_far += batch.num_rows
        if rows_so_far >= rows_needed:
            break
    table = pa.Table.from_batches(collected_batches).slice(0, rows_needed)
    return Dataset(table)


# ---------- 2. STRATEGY 1: FIXED-SIZE CHUNKING ----------

def fixed_size_chunking(text, chunk_size=200, overlap=50):
    """Splits text into overlapping chunks of `chunk_size` words."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)  # prevents infinite loop if overlap >= chunk_size
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


# ---------- 3. STRATEGY 2: SEMANTIC CHUNKING ----------

_semantic_model = None  # loaded once, reused — loading per-call would be very slow

def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _semantic_model


def _split_into_sentences(text):
    """Hindi sentences typically end in '।' (danda); English in . ! ?"""
    sentences = re.split(r'(?<=[।.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunking(text, similarity_threshold=0.5):
    """
    Splits into sentences, embeds each, then merges consecutive sentences
    into the same chunk while they stay semantically similar. Starts a new
    chunk when similarity drops below `similarity_threshold`.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]]

    model = _get_semantic_model()
    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

    chunks = []
    current_sentences = [sentences[0]]
    current_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        similarity = float(np.dot(current_embedding, embeddings[i]))  # cosine sim (normalized)
        if similarity >= similarity_threshold:
            current_sentences.append(sentences[i])
            current_embedding = np.mean([current_embedding, embeddings[i]], axis=0)
        else:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
            current_embedding = embeddings[i]

    chunks.append(" ".join(current_sentences))
    return chunks


# ---------- 4. STRATEGY 3: METADATA-AWARE CHUNKING ----------

def metadata_aware_chunking(dataset_row):
    """Treats each original passage as its own chunk — no splitting."""
    passages = dataset_row["passages"]["Translated_passages"]
    query_id = dataset_row["query_id"]
    return [(text, f"{query_id}_p{idx}") for idx, text in enumerate(passages)]


# ---------- 5. RUN ALL THREE STRATEGIES ACROSS THE DATASET ----------

def build_all_chunks(dataset):
    all_chunks = []

    for row_num, row in enumerate(dataset):
        query_id = row["query_id"]
        passages = row["passages"]["Translated_passages"]

        for idx, passage_text in enumerate(passages):
            source_id = f"{query_id}_p{idx}"

            for chunk_text in fixed_size_chunking(passage_text):
                all_chunks.append({"text": chunk_text, "method": "fixed_size", "source_id": source_id})

            for chunk_text in semantic_chunking(passage_text):
                all_chunks.append({"text": chunk_text, "method": "semantic", "source_id": source_id})

        for passage_text, source_id in metadata_aware_chunking(row):
            all_chunks.append({"text": passage_text, "method": "metadata_aware", "source_id": source_id})

        if (row_num + 1) % 100 == 0:
            print(f"Processed {row_num + 1} rows...")

    return all_chunks


# ---------- 6. MAIN ----------

if __name__ == "__main__":
    from datasets import concatenate_datasets

    # NOTE: start with a small number while testing (see note below the code)
    ROWS_TO_PROCESS = 1000

    print("Loading Hindi dataset...")
    hindi_dataset = load_data(rows_needed=ROWS_TO_PROCESS, filename="train/hintrain.parquet")
    print(f"Loaded {len(hindi_dataset)} Hindi rows.")

    print("Loading Marathi dataset...")
    marathi_dataset = load_data(rows_needed=ROWS_TO_PROCESS, filename="train/martrain.parquet")
    print(f"Loaded {len(marathi_dataset)} Marathi rows.")

    dataset = concatenate_datasets([hindi_dataset, marathi_dataset])
    print(f"Total combined rows: {len(dataset)}")

    print("Building chunks (semantic chunking is the slow part)...")
    chunks = build_all_chunks(dataset)
    print(f"Total chunks created: {len(chunks)}")

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Saved to chunks.json")