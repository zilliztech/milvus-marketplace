# BGE Series (BAAI)

Open-source embedding model series from Beijing Academy of Artificial Intelligence (BAAI), the strongest for Chinese. Free for commercial use.

## Model Versions

| Model | Dimensions | Context | Size | Features |
|-------|-----------|---------|------|----------|
| **BGE-M3** | 1024 | 8K | 2.2GB | Multi-lingual/multi-functional/multi-granularity, recommended |
| BGE-large-zh-v1.5 | 1024 | 512 | 1.3GB | Chinese only, lightweight |
| BGE-base-zh-v1.5 | 768 | 512 | 400MB | Medium scale |
| BGE-small-zh-v1.5 | 512 | 512 | 95MB | Ultra-lightweight |
| BGE-large-en-v1.5 | 1024 | 512 | 1.3GB | English only |

## Installation

```bash
pip install sentence-transformers
# Or
pip install FlagEmbedding  # Official library with more features
```

## Code Examples

### Basic Usage (sentence-transformers)

```python
from sentence_transformers import SentenceTransformer

# Load model (auto-downloads on first use)
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# Single encoding
text = "This is a Chinese text"
embedding = model.encode(text)
print(f"Dimensions: {len(embedding)}")  # 1024

# Batch encoding (recommended)
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,  # Normalize for cosine similarity
    show_progress_bar=True
)
```

### BGE-M3 (Recommended)

BGE-M3 supports three retrieval methods: dense, sparse, and multi-vector

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Encoding
texts = ["This is a text", "This is another text"]
embeddings = model.encode(
    texts,
    batch_size=12,
    max_length=8192  # Supports long text
)

# Returns dense vectors
dense_vectors = embeddings['dense_vecs']  # shape: (2, 1024)

# Returns sparse vectors (optional)
# sparse_vectors = embeddings['lexical_weights']
```

### GPU Acceleration

```python
from sentence_transformers import SentenceTransformer

# Specify GPU
model = SentenceTransformer('BAAI/bge-large-zh-v1.5', device='cuda')

# Multi-GPU
model = SentenceTransformer('BAAI/bge-large-zh-v1.5', device='cuda:0')

# Batch processing
embeddings = model.encode(
    texts,
    batch_size=64,  # Larger batch for GPU
    normalize_embeddings=True
)
```

### CPU Only (No GPU)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-zh-v1.5', device='cpu')

# Use smaller model for speed
model_small = SentenceTransformer('BAAI/bge-small-zh-v1.5', device='cpu')
```

### Query Prefix (Improves Retrieval)

BGE models recommend adding a prefix for queries:

```python
# Index documents: encode directly
doc_embedding = model.encode("Milvus is a vector database")

# Query: add prefix
query_embedding = model.encode("Represent this sentence for searching relevant passages: What is Milvus?")

# Simplified version
query_prefix = "Represent this sentence for searching relevant passages: "
query_embedding = model.encode(query_prefix + "What is Milvus?")
```

### Batch Processing Wrapper

```python
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class BGEEmbedding:
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", device: str = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = "Represent this sentence for searching relevant passages: "

    def encode_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode documents (no prefix)"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def encode_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode queries (with prefix)"""
        queries_with_prefix = [self.query_prefix + q for q in queries]
        return self.model.encode(
            queries_with_prefix,
            batch_size=batch_size,
            normalize_embeddings=True
        )

# Usage
embedder = BGEEmbedding(device='cuda')

# Indexing
doc_embeddings = embedder.encode_documents(documents)

# Searching
query_embeddings = embedder.encode_queries(["What is a vector database?"])
```

## Milvus Integration

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# Initialize
client = MilvusClient(uri="./milvus.db")
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# Create Collection
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("bge_embeddings", schema=schema, index_params=index_params)

# Insert data
texts = ["Milvus is a vector database", "BGE is the best Chinese embedding model"]
embeddings = model.encode(texts, normalize_embeddings=True).tolist()

data = [{"text": t, "embedding": e} for t, e in zip(texts, embeddings)]
client.insert("bge_embeddings", data)

# Search
query = "Represent this sentence for searching relevant passages: What is a vector database?"
query_embedding = model.encode(query, normalize_embeddings=True).tolist()

results = client.search(
    collection_name="bge_embeddings",
    data=[query_embedding],
    limit=10,
    output_fields=["text"]
)
```

## BGE-M3 Hybrid Search

BGE-M3 supports dense + sparse hybrid search:

```python
from FlagEmbedding import BGEM3FlagModel
from pymilvus import MilvusClient, DataType

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Encode (get both dense and sparse vectors)
texts = ["Text 1", "Text 2"]
embeddings = model.encode(texts, return_dense=True, return_sparse=True)

dense_vecs = embeddings['dense_vecs']
sparse_vecs = embeddings['lexical_weights']

# Milvus 2.4+ supports sparse vectors
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

# Hybrid search
results = client.hybrid_search(
    collection_name="hybrid_collection",
    reqs=[
        {"data": [dense_query], "anns_field": "dense_vector", "limit": 10},
        {"data": [sparse_query], "anns_field": "sparse_vector", "limit": 10}
    ],
    rerank={"strategy": "rrf", "params": {"k": 60}}  # RRF fusion
)
```

## Model Download (Offline Use)

```bash
# Method 1: huggingface-cli
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir ./bge-large-zh

# Method 2: git lfs
git lfs install
git clone https://huggingface.co/BAAI/bge-large-zh-v1.5

# Use local model
model = SentenceTransformer('./bge-large-zh')
```

## China Mirror

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
```

## Performance Comparison

Performance on C-MTEB (Chinese benchmark):

| Model | Average | Retrieval | Semantic Similarity | Classification |
|-------|---------|-----------|--------------------|----|
| BGE-M3 | 63.0 | 65.2 | 62.1 | 61.8 |
| BGE-large-zh | 64.2 | 63.5 | 65.8 | 63.2 |
| text2vec-large | 57.8 | 55.2 | 59.8 | 58.5 |

## Selection Recommendations

| Scenario | Recommended Model |
|----------|------------------|
| Chinese-focused, ample resources | BGE-M3 |
| Pure Chinese, fast | BGE-large-zh-v1.5 |
| Resource-constrained | BGE-small-zh-v1.5 |
| Multilingual mix | BGE-M3 |
| Need sparse retrieval | BGE-M3 |
