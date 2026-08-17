from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from datasets import Dataset

# Downloads the file once (cached after this), doesn't load it into memory yet
file_path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="train/hintrain.parquet",
    repo_type="dataset"
)

# Read the file in small batches, stopping once we have 5000 rows
parquet_file = pq.ParquetFile(file_path)
rows_needed = 5000
collected_batches = []
rows_so_far = 0

for batch in parquet_file.iter_batches(batch_size=500):
    collected_batches.append(batch)
    rows_so_far += batch.num_rows
    if rows_so_far >= rows_needed:
        break

# Combine batches and trim to exactly 5000 rows
import pyarrow as pa
table = pa.Table.from_batches(collected_batches).slice(0, rows_needed)
dataset = Dataset(table)

print(f"Number of rows loaded: {len(dataset)}")
print(f"Column names: {dataset.column_names}")
print("\n--- First 3 rows ---\n")
for i in range(3):
    print(f"Row {i}:")
    print(dataset[i])
    print("-" * 60)