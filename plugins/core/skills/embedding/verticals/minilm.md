# all-MiniLM (Lightweight Embedding)

Lightweight models from sentence-transformers, runs quickly on CPU.

## Model Versions

| Model | Dimensions | Size | Features |
|-------|-----------|------|----------|
| **all-MiniLM-L6-v2** | 384 | 80MB | Lightest, fastest |
| all-MiniLM-L12-v2 | 384 | 120MB | Slightly larger, better performance |
| paraphrase-MiniLM-L6-v2 | 384 | 80MB | Semantic similarity optimized |
| multi-qa-MiniLM-L6-cos-v1 | 384 | 80MB | QA scenario optimized |
| all-mpnet-base-v2 | 768 | 420MB | Best performance (same series) |

## Use Cases

MiniLM series is suitable for:
- Resource-constrained environments (no GPU, limited memory)
- Quick validation and prototyping
- Latency-sensitive real-time scenarios
- Simple semantic search needs

## Installation

```bash
pip install sentence-transformers
```

## Code Examples

### Basic Usage

```python
from sentence_transformers import SentenceTransformer

# Load model (auto-downloads on first use, only 80MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Single encoding
embedding = model.encode("Hello, world!")
print(f"Dimensions: {len(embedding)}")  # 384

# Batch encoding
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = model.encode(texts, normalize_embeddings=True)
```

### Efficient CPU Inference

```python
from sentence_transformers import SentenceTransformer

# Explicitly specify CPU
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Batch processing (smaller batch for CPU)
embeddings = model.encode(
    texts,
    batch_size=32,  # Smaller batch for CPU
    normalize_embeddings=True,
    show_progress_bar=True
)
```

### Higher Accuracy Models

```python
# L12 version (slightly slower but better performance)
model = SentenceTransformer('all-MiniLM-L12-v2')

# MPNet version (best performance, 768 dimensions)
model = SentenceTransformer('all-mpnet-base-v2')
```

### Scenario-Specific Optimization

```python
# QA scenario (question answering)
model_qa = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Semantic similarity
model_paraphrase = SentenceTransformer('paraphrase-MiniLM-L6-v2')
```

### Batch Processing Wrapper

```python
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class MiniLMEmbedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device='cpu')

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

# Usage
embedder = MiniLMEmbedding()
embeddings = embedder.encode(["Text 1", "Text 2"])
print(f"Dimensions: {embedder.dimension}")  # 384
```

### Multi-threading Speedup (CPU)

```python
from sentence_transformers import SentenceTransformer
import torch

# Set CPU thread count
torch.set_num_threads(4)  # Adjust based on CPU cores

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Encode
embeddings = model.encode(texts, batch_size=32)
```

## Milvus Integration

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# Initialize
client = MilvusClient(uri="./milvus.db")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create Collection (384 dimensions)
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=384)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("minilm_embeddings", schema=schema, index_params=index_params)

# Insert data
texts = ["MiniLM is a lightweight model", "Great for CPU inference"]
embeddings = model.encode(texts, normalize_embeddings=True).tolist()

data = [{"text": t, "embedding": e} for t, e in zip(texts, embeddings)]
client.insert("minilm_embeddings", data)

# Search
query = "lightweight embedding model"
query_embedding = model.encode(query, normalize_embeddings=True).tolist()

results = client.search(
    collection_name="minilm_embeddings",
    data=[query_embedding],
    limit=10,
    output_fields=["text"]
)
```

## Combined with Milvus Lite (Pure Local Solution)

MiniLM + Milvus Lite is a perfect pure local lightweight solution:

```python
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# No external services needed, pure local execution
client = MilvusClient(uri="./local.db")  # File database
model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB model

# Entire system has minimal resource usage
# - Model: ~80MB memory
# - Database: Tens of MB (depends on data volume)
```

## Performance Benchmarks

On regular laptop CPU (Intel i5):

| Model | 1000 texts encoding time | Memory usage |
|-------|--------------------------|--------------|
| all-MiniLM-L6-v2 | ~5 seconds | ~100MB |
| all-MiniLM-L12-v2 | ~8 seconds | ~150MB |
| all-mpnet-base-v2 | ~15 seconds | ~500MB |

## Model Download (Offline Use)

```bash
# Download model
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./minilm

# Use local model
model = SentenceTransformer('./minilm')
```

## China Mirror

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

## Limits and Notes

| Limit | Value |
|-------|-------|
| Max sequence length | 256 tokens |
| Dimensions | 384 (L6/L12) / 768 (mpnet) |
| Language | Primarily English, average Chinese performance |

**Notes**:
- Chinese performance not as good as BGE, Qwen3, etc.
- 256 token limit, long text needs chunking
- Best for English scenarios or quick validation

## Selection Recommendations

| Scenario | Recommended Model |
|----------|------------------|
| Ultra-lightweight | all-MiniLM-L6-v2 |
| Slightly higher accuracy | all-MiniLM-L12-v2 |
| Best performance | all-mpnet-base-v2 |
| QA question answering | multi-qa-MiniLM-L6-cos-v1 |
| Semantic similarity | paraphrase-MiniLM-L6-v2 |

## Why Choose MiniLM?

1. **Ultra-lightweight**: 80MB model, downloads in seconds
2. **CPU-friendly**: No GPU needed, runs on laptops
3. **Low latency**: Extremely fast encoding
4. **Ready to use**: Just pip install
5. **Good for prototyping**: Quickly validate ideas

## When NOT to Choose MiniLM?

- Need high-precision semantic understanding
- Chinese-focused scenarios
- Long text processing (exceeds 256 tokens)
- Extremely high retrieval precision requirements

For these scenarios, consider BGE-M3, Qwen3-Embedding, or OpenAI API.
