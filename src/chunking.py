"""
chunking.py
Three chunking strategies for HHGOA-SonicQuery:
1. fixed_size_chunking      - word-window chunks with overlap
2. semantic_chunking        - groups sentences using sentence-transformer embeddings
3. metadata_aware_chunking  - treats each original passage as its own chunk

Every chunk produced is tagged with a "language" field — this is required
for retrieval.py's language filtering to work at all.
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

def metadata_aware_chunking(dataset_row, passage_field="Translated_passages"):
    """Treats each original passage as its own chunk — no splitting."""
    passages = dataset_row["passages"][passage_field]
    query_id = dataset_row["query_id"]
    return [(text, f"{query_id}_p{idx}") for idx, text in enumerate(passages)]


# ---------- 5. RUN ALL THREE STRATEGIES ACROSS THE DATASET ----------

def _chunk_passages(passages, query_id, language):
    """Runs fixed_size + semantic chunking over one row's passage list, tagged with language."""
    out = []
    for idx, passage_text in enumerate(passages):
        source_id = f"{query_id}_p{idx}"
        for chunk_text in fixed_size_chunking(passage_text):
            out.append({"text": chunk_text, "method": "fixed_size", "source_id": source_id, "language": language})
        for chunk_text in semantic_chunking(passage_text):
            out.append({"text": chunk_text, "method": "semantic", "source_id": source_id, "language": language})
    return out


def build_all_chunks(dataset, language, include_english=False):
    """
    language: tag applied to every chunk built from this dataset's
    Translated_passages (e.g. "hindi", "marathi").

    include_english: if True, ALSO chunks passages["English_passages"] from
    the same rows, tagged "language": "english". Only set this True for ONE
    dataset load — every language config shares the same underlying English
    source text, so extracting it twice just duplicates chunks and wastes
    embedding time.
    """
    all_chunks = []

    for row_num, row in enumerate(dataset):
        query_id = row["query_id"]

        # Translated passages (Hindi, Marathi, etc.), tagged with `language`
        translated_passages = row["passages"]["Translated_passages"]
        all_chunks.extend(_chunk_passages(translated_passages, query_id, language))

        for passage_text, source_id in metadata_aware_chunking(row, "Translated_passages"):
            all_chunks.append({"text": passage_text, "method": "metadata_aware", "source_id": source_id, "language": language})

        # English passages, tagged "english" — only when this row's load requests it
        if include_english:
            english_passages = row["passages"]["English_passages"]
            all_chunks.extend(_chunk_passages(english_passages, query_id, "english"))

            for passage_text, source_id in metadata_aware_chunking(row, "English_passages"):
                all_chunks.append({"text": passage_text, "method": "metadata_aware", "source_id": source_id, "language": "english"})

        if (row_num + 1) % 100 == 0:
            print(f"Processed {row_num + 1} rows ({language})...")

    return all_chunks


# ---------- 6. MAIN ----------

if __name__ == "__main__":
    # NOTE: start with a small number while testing (see note below the code)
    ROWS_TO_PROCESS = 2500

    print("Loading Hindi dataset...")
    hindi_dataset = load_data(rows_needed=ROWS_TO_PROCESS, filename="train/hintrain.parquet")
    print(f"Loaded {len(hindi_dataset)} Hindi rows.")

    print("Loading Marathi dataset...")
    marathi_dataset = load_data(rows_needed=ROWS_TO_PROCESS, filename="train/martrain.parquet")
    print(f"Loaded {len(marathi_dataset)} Marathi rows.")

    print("Building Hindi + English chunks (English piggybacks on the Hindi rows, no extra download)...")
    hindi_chunks = build_all_chunks(hindi_dataset, language="hindi", include_english=True)
    print(f"Hindi+English chunks created: {len(hindi_chunks)}")

    print("Building Marathi chunks...")
    marathi_chunks = build_all_chunks(marathi_dataset, language="marathi", include_english=False)
    print(f"Marathi chunks created: {len(marathi_chunks)}")

    chunks = hindi_chunks + marathi_chunks
    print(f"Total chunks created: {len(chunks)}")
    print(f"  hindi: {sum(1 for c in chunks if c['language'] == 'hindi')}")
    print(f"  marathi: {sum(1 for c in chunks if c['language'] == 'marathi')}")
    print(f"  english: {sum(1 for c in chunks if c['language'] == 'english')}")

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    import os
    size_mb = os.path.getsize("chunks.json") / (1024 * 1024)
    print(f"Saved to chunks.json ({size_mb:.1f} MB)")
    if size_mb > 80:
        print("WARNING: getting close to GitHub's 100MB limit — lower ROWS_TO_PROCESS")