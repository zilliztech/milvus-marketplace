# Voyage AI

Domain-specific embedding by Stanford AI team, strongest in legal, finance, and code domains.

## Model Versions

| Model | Dimensions | Context | Price ($/1M tokens) | Features |
|-------|-----------|---------|---------------------|----------|
| **voyage-3-large** | 1024 | 32K | $0.06 | Strongest general, MTEB #1 |
| voyage-3 | 1024 | 32K | $0.06 | General |
| voyage-3-lite | 512 | 32K | $0.02 | Low cost, low latency |
| **voyage-code-3** | 1024 | 32K | $0.06 | Code-specific |
| **voyage-finance-2** | 1024 | 32K | $0.12 | Finance-specific |
| **voyage-law-2** | 1024 | 16K | $0.12 | Legal-specific |
| voyage-multilingual-2 | 1024 | 32K | $0.12 | Multilingual |

## Registration and API Key

1. Visit https://dash.voyageai.com/
2. Register account (Google login supported)
3. Go to API Keys page
4. Create new API Key

**Free quota**: $3 free credit for new users.

## Installation

```bash
pip install voyageai
```

## Environment Configuration

```bash
export VOYAGE_API_KEY="xxx..."
```

## Code Examples

### Basic Usage

```python
import voyageai

vo = voyageai.Client()  # Auto-reads VOYAGE_API_KEY

# Encode documents
doc_embeddings = vo.embed(
    ["This is the first text", "This is the second text"],
    model="voyage-3",
    input_type="document"
)
embeddings = doc_embeddings.embeddings

# Encode queries
query_embeddings = vo.embed(
    ["Search query"],
    model="voyage-3",
    input_type="query"
)
query_embedding = query_embeddings.embeddings[0]
```

### input_type Explained

| input_type | Use Case |
|-----------|----------|
| document | Index documents |
| query | Search queries |
| None | General (clustering, etc.) |

### Dimension Truncation

```python
# Truncate to 512 dimensions (save storage)
embeddings = vo.embed(
    ["Text"],
    model="voyage-3-large",
    input_type="document",
    output_dimension=512  # Options: 256, 512, 1024
)
```

### Code-Specific Model

```python
# Code search
code_snippets = [
    "def quicksort(arr): ...",
    "function mergeSort(arr) { ... }",
    "SELECT * FROM users WHERE id = 1"
]

code_embeddings = vo.embed(
    code_snippets,
    model="voyage-code-3",
    input_type="document"
)

# Search code with natural language
query_embedding = vo.embed(
    ["quicksort algorithm implementation"],
    model="voyage-code-3",
    input_type="query"
)
```

### Legal-Specific Model

```python
# Legal documents
legal_docs = [
    "Article 1076 of Civil Code: When both parties voluntarily agree to divorce...",
    "Party A agrees to pay Party B a penalty of 10,000 CNY..."
]

legal_embeddings = vo.embed(
    legal_docs,
    model="voyage-law-2",
    input_type="document"
)

# Legal query
query = "What are the conditions for divorce?"
query_embedding = vo.embed([query], model="voyage-law-2", input_type="query")
```

### Finance-Specific Model

```python
# Financial documents
finance_docs = [
    "Q3 revenue up 23% YoY, gross margin maintained at 35%...",
    "Fed announces 25 basis points rate hike, market expects dovish signals..."
]

finance_embeddings = vo.embed(
    finance_docs,
    model="voyage-finance-2",
    input_type="document"
)
```

### Batch Processing Wrapper

```python
import voyageai
from typing import List

class VoyageEmbedding:
    def __init__(self, model: str = "voyage-3"):
        self.vo = voyageai.Client()
        self.model = model

    def encode_documents(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """Encode documents"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = self.vo.embed(batch, model=self.model, input_type="document")
            all_embeddings.extend(result.embeddings)

        return all_embeddings

    def encode_queries(self, queries: List[str]) -> List[List[float]]:
        """Encode queries"""
        result = self.vo.embed(queries, model=self.model, input_type="query")
        return result.embeddings

# Usage
embedder = VoyageEmbedding(model="voyage-3-large")
doc_embeddings = embedder.encode_documents(documents)
query_embeddings = embedder.encode_queries(["Search content"])
```

### Async Calls

```python
import voyageai
import asyncio

async def embed_async(texts: list, model: str = "voyage-3"):
    vo = voyageai.AsyncClient()
    result = await vo.embed(texts, model=model, input_type="document")
    return result.embeddings

# Usage
embeddings = asyncio.run(embed_async(["Text 1", "Text 2"]))
```

## Milvus Integration

```python
from pymilvus import MilvusClient, DataType
import voyageai

# Initialize
client = MilvusClient(uri="./milvus.db")
vo = voyageai.Client()

# Create Collection
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("voyage_embeddings", schema=schema, index_params=index_params)

# Insert data
texts = ["Voyage AI is built by Stanford team", "Focuses on domain-specific embedding"]
result = vo.embed(texts, model="voyage-3", input_type="document")
embeddings = result.embeddings

data = [{"text": t, "embedding": e} for t, e in zip(texts, embeddings)]
client.insert("voyage_embeddings", data)

# Search
query = "Who built Voyage AI?"
query_result = vo.embed([query], model="voyage-3", input_type="query")
query_embedding = query_result.embeddings[0]

results = client.search(
    collection_name="voyage_embeddings",
    data=[query_embedding],
    limit=10,
    output_fields=["text"]
)
```

## Limits and Notes

| Limit | Value |
|-------|-------|
| Max texts per call | 128 |
| Max tokens per text | 32K (voyage-3) / 16K (law) |
| Rate limit | 300 RPM / 1M TPM |

## Selection Recommendations

| Scenario | Recommended Model |
|----------|------------------|
| General search, RAG | voyage-3-large |
| Code search | voyage-code-3 |
| Legal documents | voyage-law-2 |
| Financial research | voyage-finance-2 |
| Low cost, low latency | voyage-3-lite |
| Multilingual (Chinese) | voyage-multilingual-2 |

## Why Choose Voyage?

1. **Strongest for domains**: Significantly outperforms OpenAI in legal, finance, code
2. **Long context**: Supports 32K tokens, no chunking needed
3. **Cost-effective**: 50%+ cheaper than OpenAI large
4. **Low latency**: Lite model suitable for real-time scenarios
5. **Flexible dimensions**: Control storage cost with truncation
