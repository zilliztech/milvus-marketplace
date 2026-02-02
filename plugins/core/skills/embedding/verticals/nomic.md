# Nomic Embed

Nomic AI's open-source embedding model, fully open-source with long context support.

## Model Versions

| Model | Dimensions | Context | Size | Features |
|-------|-----------|---------|------|----------|
| **nomic-embed-text-v1.5** | 768 | 8K | 548MB | Latest, best performance |
| nomic-embed-text-v1 | 768 | 8K | 548MB | Stable version |
| **nomic-embed-vision** | 768 | - | 640MB | Image embedding |

## Features

- **Fully open-source**: Code and data are both open-source, Apache 2.0 license
- **Long context**: Native support for 8192 tokens
- **Reproducible**: Training data is public, results can be reproduced
- **Variable dimensions**: Supports Matryoshka dimensions (64-768)

## Installation

```bash
pip install sentence-transformers
```

## Code Examples

### Basic Usage

```python
from sentence_transformers import SentenceTransformer

# Load model (auto-downloads on first use)
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Encode text (requires adding prefix)
texts = ["search_document: This is a document", "search_document: Another document"]
embeddings = model.encode(texts)
print(f"Dimensions: {len(embeddings[0])}")  # 768
```

### Task Prefixes

Nomic models recommend using task prefixes:

```python
# Document indexing
doc_texts = ["search_document: " + doc for doc in documents]
doc_embeddings = model.encode(doc_texts)

# Search queries
query_texts = ["search_query: " + q for q in queries]
query_embeddings = model.encode(query_texts)

# Clustering task
cluster_texts = ["clustering: " + t for t in texts]
cluster_embeddings = model.encode(cluster_texts)

# Classification task
class_texts = ["classification: " + t for t in texts]
class_embeddings = model.encode(class_texts)
```

### Variable Dimensions (Matryoshka)

```python
# Use lower dimensions (save storage)
embeddings_256 = model.encode(
    texts,
    normalize_embeddings=True
)[:, :256]  # Truncate to first 256 dimensions

# Available dimensions: 64, 128, 256, 512, 768
embeddings_128 = model.encode(texts)[:, :128]
```

### GPU Acceleration

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    device="cuda"
)

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True
)
```

### Batch Processing Wrapper

```python
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class NomicEmbedding:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", device: str = None):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    def encode_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode documents"""
        prefixed = ["search_document: " + t for t in texts]
        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries"""
        prefixed = ["search_query: " + q for q in queries]
        return self.model.encode(
            prefixed,
            normalize_embeddings=True
        )

# Usage
embedder = NomicEmbedding(device="cuda")
doc_embeddings = embedder.encode_documents(documents)
query_embedding = embedder.encode_queries(["Search content"])
```

## Milvus Integration

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# Initialize
client = MilvusClient(uri="./milvus.db")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Create Collection
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=768)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("nomic_embeddings", schema=schema, index_params=index_params)

# Insert data
texts = ["Nomic AI is an open-source AI company", "Focuses on embedding and data visualization"]
prefixed = ["search_document: " + t for t in texts]
embeddings = model.encode(prefixed, normalize_embeddings=True).tolist()

data = [{"text": t, "embedding": e} for t, e in zip(texts, embeddings)]
client.insert("nomic_embeddings", data)

# Search
query = "search_query: What is Nomic?"
query_embedding = model.encode(query, normalize_embeddings=True).tolist()

results = client.search(
    collection_name="nomic_embeddings",
    data=[query_embedding],
    limit=10,
    output_fields=["text"]
)
```

## Nomic Embed Vision (Images)

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

# Load vision model
model = SentenceTransformer("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

# Encode images
image = Image.open("image.jpg")
image_embedding = model.encode(image)

# Batch images
images = [Image.open(f"img_{i}.jpg") for i in range(10)]
image_embeddings = model.encode(images)

# Note: Image and text embeddings are in the same space, enabling cross-modal retrieval
```

## Model Download (Offline Use)

```bash
# Download model
huggingface-cli download nomic-ai/nomic-embed-text-v1.5 --local-dir ./nomic-embed

# Use local model
model = SentenceTransformer('./nomic-embed', trust_remote_code=True)
```

## China Mirror

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
```

## Limits and Notes

| Limit | Value |
|-------|-------|
| Max context | 8192 tokens |
| Default dimensions | 768 |
| Variable dimensions | 64-768 |

**Notes**:
- Requires `trust_remote_code=True`
- Recommend using task prefixes for better results
- Chinese performance not as good as BGE/Qwen, better suited for English

## Selection Recommendations

| Scenario | Recommendation |
|----------|---------------|
| Open-source priority | nomic-embed-text-v1.5 |
| Long text | nomic-embed-text-v1.5 |
| Image embedding | nomic-embed-vision |
| Chinese scenarios | BGE-M3 / Qwen3 are better |

## Why Choose Nomic?

1. **Fully open-source**: Code, models, and data all open-source
2. **Long context**: 8K tokens native support
3. **Variable dimensions**: Matryoshka support for flexible choices
4. **Unified image-text**: Text and images in same vector space
5. **Reproducible**: Training process fully public
