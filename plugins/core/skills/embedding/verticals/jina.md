# Jina Embeddings

Embedding models developed by Jina AI, supporting long text and multilingual.

## Model Versions

| Model | Dimensions | Context | Features |
|-------|-----------|---------|----------|
| **jina-embeddings-v3** | 1024 | 8K | Latest, multilingual, Late Chunking |
| jina-embeddings-v2-base-en | 768 | 8K | English, 8K context |
| jina-embeddings-v2-base-zh | 768 | 8K | Chinese optimized |
| jina-embeddings-v2-small-en | 512 | 8K | Lightweight English |
| **jina-colbert-v2** | 128 | 8K | ColBERT multi-vector |

## API vs Local

Jina provides two usage methods:
- **API**: Register and use, no GPU required
- **Local deployment**: Open-source models, free for commercial use

## Option 1: Jina API

### Register and Get API Key

1. Visit https://jina.ai/embeddings/
2. Click "Start for Free"
3. Register account (Google login supported)
4. Get API Key

**Free quota**: 1 million tokens per month

### API Call

```python
import requests

API_KEY = "jina_xxx..."  # Your API Key

def get_embeddings(texts: list, model: str = "jina-embeddings-v3"):
    """Call Jina API"""
    url = "https://api.jina.ai/v1/embeddings"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    data = {
        "model": model,
        "input": texts,
        "encoding_type": "float"  # Or "binary", "ubinary"
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    return [item["embedding"] for item in result["data"]]

# Usage
embeddings = get_embeddings(["Hello world", "Hello world"])
```

### OpenAI-Compatible Interface

```python
from openai import OpenAI

client = OpenAI(
    api_key="jina_xxx...",
    base_url="https://api.jina.ai/v1/"
)

response = client.embeddings.create(
    model="jina-embeddings-v3",
    input=["Hello world", "Hello world"]
)

embeddings = [item.embedding for item in response.data]
```

## Option 2: Local Deployment

### Installation

```bash
pip install sentence-transformers
```

### Basic Usage

```python
from sentence_transformers import SentenceTransformer

# Load model (auto-downloads on first use)
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

# Single encoding
embedding = model.encode("This is some text")
print(f"Dimensions: {len(embedding)}")  # 1024

# Batch encoding
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = model.encode(texts, normalize_embeddings=True)
```

### Using v2 Models (Lighter)

```python
# English model
model_en = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)

# Chinese model
model_zh = SentenceTransformer("jinaai/jina-embeddings-v2-base-zh", trust_remote_code=True)

# Small model
model_small = SentenceTransformer("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)
```

### GPU Acceleration

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "jinaai/jina-embeddings-v3",
    trust_remote_code=True,
    device="cuda"
)

embeddings = model.encode(
    texts,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)
```

### Task Types (v3 Feature)

Jina v3 supports specifying task types for optimization:

```python
# Retrieval task
embeddings = model.encode(
    texts,
    task="retrieval.passage"  # Documents
)

query_embedding = model.encode(
    ["Search query"],
    task="retrieval.query"  # Queries
)

# Other task types
# - separation: Clustering/classification
# - classification: Classification
# - text-matching: Semantic similarity
```

### Late Chunking (v3 Feature)

Preserve context when processing long text:

```python
# Long text encoding with context-aware chunking
long_text = "This is a very long article..." * 500

# v3 internally handles long text context relationships
embedding = model.encode(long_text)
```

### Batch Processing Wrapper

```python
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class JinaEmbedding:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3", device: str = None):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    def encode_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode documents"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            task="retrieval.passage",  # v3
            show_progress_bar=True
        )

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries"""
        return self.model.encode(
            queries,
            normalize_embeddings=True,
            task="retrieval.query"  # v3
        )

# Usage
embedder = JinaEmbedding(device="cuda")
doc_embeddings = embedder.encode_documents(documents)
query_embedding = embedder.encode_queries(["Search content"])
```

## Milvus Integration

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# Initialize
client = MilvusClient(uri="./milvus.db")
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

# Create Collection
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("jina_embeddings", schema=schema, index_params=index_params)

# Insert data
texts = ["Jina AI focuses on search and vector technology", "Supports multilingual and long text"]
embeddings = model.encode(texts, normalize_embeddings=True, task="retrieval.passage").tolist()

data = [{"text": t, "embedding": e} for t, e in zip(texts, embeddings)]
client.insert("jina_embeddings", data)

# Search
query = "What is Jina?"
query_embedding = model.encode(query, normalize_embeddings=True, task="retrieval.query").tolist()

results = client.search(
    collection_name="jina_embeddings",
    data=[query_embedding],
    limit=10,
    output_fields=["text"]
)
```

## Jina ColBERT (Multi-Vector Retrieval)

ColBERT uses multi-vector representation for higher precision:

```python
from sentence_transformers import SentenceTransformer

# Load ColBERT model
model = SentenceTransformer("jinaai/jina-colbert-v2", trust_remote_code=True)

# Encode (returns multiple vectors)
embeddings = model.encode("This is some text", output_value="token_embeddings")
print(f"Token count: {len(embeddings)}")  # One vector per token
print(f"Vector dimension: {len(embeddings[0])}")  # 128

# ColBERT requires special retrieval method (MaxSim)
# Milvus 2.5+ supports ColBERT retrieval
```

## Model Download (Offline Use)

```bash
# Download model
huggingface-cli download jinaai/jina-embeddings-v3 --local-dir ./jina-v3

# Use local model
model = SentenceTransformer('./jina-v3', trust_remote_code=True)
```

## China Mirror

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)
```

## Limits and Notes

| Limit | Value |
|-------|-------|
| v3 max context | 8192 tokens |
| v2 max context | 8192 tokens |
| API free quota | 1M tokens/month |

**Notes**:
- Jina models require `trust_remote_code=True`
- v3 task type parameter improves performance
- ColBERT requires special retrieval logic

## Selection Recommendations

| Scenario | Recommended Model |
|----------|------------------|
| General multilingual | jina-embeddings-v3 |
| Chinese-focused | jina-embeddings-v2-base-zh |
| Resource-constrained | jina-embeddings-v2-small-en |
| High-precision retrieval | jina-colbert-v2 |
| Quick validation | Jina API |

## Why Choose Jina?

1. **Long context**: Native 8K tokens support
2. **Multilingual**: v3 supports 30+ languages
3. **Late Chunking**: Smarter long text processing
4. **Free API**: 1M tokens/month
5. **Open source commercial**: Apache 2.0 license
