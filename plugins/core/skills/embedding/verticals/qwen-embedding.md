# Qwen3-Embedding (Alibaba Qwen Series)

The latest open-source embedding model from Alibaba's Qwen team, strongest for Chinese-English bilingual.

## Model Versions

| Model | Dimensions | Context | Size | Features |
|-------|-----------|---------|------|----------|
| **Qwen3-Embedding-8B** | 4096 | 32K | 16GB | Strongest, MTEB top-tier |
| Qwen3-Embedding-4B | 2560 | 32K | 8GB | Balanced |
| Qwen3-Embedding-0.6B | 1024 | 32K | 1.2GB | Lightweight, runs on CPU |

## Installation

```bash
pip install sentence-transformers
# Or
pip install transformers torch
```

## Code Examples

### Basic Usage (sentence-transformers)

```python
from sentence_transformers import SentenceTransformer

# Load model (auto-downloads on first use)
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Single encoding
embedding = model.encode("This is some Chinese text")
print(f"Dimensions: {len(embedding)}")  # 1024

# Batch encoding
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = model.encode(texts, normalize_embeddings=True)
```

### Using Larger Models

```python
# 8B model (requires GPU)
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")

# 4B model
model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", device="cuda")
```

### Using transformers (More Flexible)

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "Qwen/Qwen3-Embedding-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token or mean pooling
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.numpy()

# Usage
embeddings = get_embedding(["Text 1", "Text 2"])
```

### GPU Acceleration

```python
from sentence_transformers import SentenceTransformer

# Specify GPU
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")

# Batch processing
embeddings = model.encode(
    texts,
    batch_size=64,  # Larger batch for GPU
    normalize_embeddings=True,
    show_progress_bar=True
)
```

### Half-Precision Inference (Save VRAM)

```python
from sentence_transformers import SentenceTransformer
import torch

# Use FP16
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda"
)

# Or use BF16 (more stable)
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda"
)
```

### Batch Processing Wrapper

```python
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class QwenEmbedding:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = None):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def encode_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode queries (Qwen3 doesn't need special prefix)"""
        return self.encode(queries, batch_size)

    def encode_documents(self, docs: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode documents"""
        return self.encode(docs, batch_size)

# Usage
embedder = QwenEmbedding(device="cuda")
doc_embeddings = embedder.encode_documents(documents)
query_embedding = embedder.encode_queries(["Search content"])
```

## Milvus Integration

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# Initialize
client = MilvusClient(uri="./milvus.db")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Create Collection (0.6B model has 1024 dimensions)
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("qwen_embeddings", schema=schema, index_params=index_params)

# Insert data
texts = ["Qwen3 is Alibaba's latest embedding model", "Milvus is an open-source vector database"]
embeddings = model.encode(texts, normalize_embeddings=True).tolist()

data = [{"text": t, "embedding": e} for t, e in zip(texts, embeddings)]
client.insert("qwen_embeddings", data)

# Search
query = "What is Qwen3?"
query_embedding = model.encode(query, normalize_embeddings=True).tolist()

results = client.search(
    collection_name="qwen_embeddings",
    data=[query_embedding],
    limit=10,
    output_fields=["text"]
)
```

## Long Text Processing

Qwen3-Embedding supports 32K context, can handle long documents:

```python
# Encode long text directly
long_text = "This is a very long article..." * 1000  # Assume very long

# Model will auto-truncate to 32K tokens
embedding = model.encode(long_text, normalize_embeddings=True)

# If complete information is needed, recommend chunking
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100):
    """Simple chunking"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = chunk_text(long_text)
chunk_embeddings = model.encode(chunks, normalize_embeddings=True)
```

## Model Download (Offline Use)

```bash
# Method 1: huggingface-cli
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./qwen3-0.6b

# Use local model
model = SentenceTransformer('./qwen3-0.6b')
```

## China Mirror

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
```

## Performance Comparison

Performance on MTEB Chinese-English benchmarks:

| Model | MTEB English | C-MTEB Chinese | Parameters |
|-------|-------------|----------------|------------|
| Qwen3-Embedding-8B | 70.5 | 72.3 | 8B |
| Qwen3-Embedding-0.6B | 65.2 | 67.8 | 0.6B |
| BGE-M3 | 66.8 | 63.0 | 0.5B |
| text-embedding-3-large | 64.6 | 60.2 | - |

## Selection Recommendations

| Scenario | Recommended Model | VRAM Requirement |
|----------|------------------|------------------|
| Highest performance | Qwen3-Embedding-8B | 16GB+ |
| Balance performance & resources | Qwen3-Embedding-4B | 8GB |
| Resource-constrained / CPU | Qwen3-Embedding-0.6B | 2GB / CPU capable |
| Production stability | Qwen3-Embedding-0.6B | Recommended |

## Why Choose Qwen3-Embedding?

1. **Strongest Chinese-English bilingual**: Excellent performance on both MTEB and C-MTEB
2. **Long context**: Supports 32K tokens, no chunking needed
3. **Open-source free**: Apache 2.0 license, commercial use allowed
4. **Multiple sizes**: From 0.6B to 8B, fits different resources
5. **Latest architecture**: Fine-tuned on Qwen3 LLM, better performance
