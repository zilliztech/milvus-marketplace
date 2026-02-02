# Cohere Embed v4

Cohere's multilingual Embedding API, featuring compressible dimensions and 100+ language support.

## Model Versions

| Model | Dimensions | Context | Price ($/1M tokens) | Features |
|-------|-----------|---------|---------------------|----------|
| embed-v4.0 | 1024 | 512 | $0.10 | Latest, strongest multilingual |
| embed-english-v3.0 | 1024 | 512 | $0.10 | English only |
| embed-multilingual-v3.0 | 1024 | 512 | $0.10 | Multilingual |
| embed-english-light-v3.0 | 384 | 512 | $0.10 | Lightweight |

## Registration and API Key

1. Visit https://dashboard.cohere.com/welcome/register
2. Register account (Google login supported)
3. Go to https://dashboard.cohere.com/api-keys
4. Copy Trial API Key (free quota) or create Production Key

**Free quota**: Trial Key has 1000 calls per month, suitable for testing.

## Installation

```bash
pip install cohere
```

## Environment Configuration

```bash
export COHERE_API_KEY="xxx..."
```

## Code Examples

### Basic Usage

```python
import cohere

co = cohere.Client()  # Auto-reads COHERE_API_KEY

# Encode documents
response = co.embed(
    texts=["This is the first text", "This is the second text"],
    model="embed-v4.0",
    input_type="search_document",  # Use search_document for documents
    embedding_types=["float"]
)
embeddings = response.embeddings.float

# Encode queries
query_response = co.embed(
    texts=["Search query"],
    model="embed-v4.0",
    input_type="search_query",  # Use search_query for queries
    embedding_types=["float"]
)
query_embedding = query_response.embeddings.float[0]
```

### input_type Explained

Cohere distinguishes different use cases with input_type:

| input_type | Use Case |
|-----------|----------|
| search_document | Index documents |
| search_query | Search queries |
| classification | Classification tasks |
| clustering | Clustering tasks |

### Dimension Compression

```python
# Compress to 256 dimensions
response = co.embed(
    texts=["Text"],
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"],
    truncate="END"  # Truncate if too long
)

# Use binary vectors (smaller storage)
response = co.embed(
    texts=["Text"],
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["ubinary"]  # Unsigned binary
)
binary_embeddings = response.embeddings.ubinary
```

### Batch Processing Wrapper

```python
import cohere
from typing import List
import time

class CohereEmbedding:
    def __init__(self, model: str = "embed-v4.0"):
        self.co = cohere.Client()
        self.model = model

    def encode_documents(self, texts: List[str], batch_size: int = 96) -> List[List[float]]:
        """Encode documents"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self.co.embed(
                texts=batch,
                model=self.model,
                input_type="search_document",
                embedding_types=["float"]
            )
            all_embeddings.extend(response.embeddings.float)

            if i + batch_size < len(texts):
                time.sleep(0.1)  # Avoid rate limiting

        return all_embeddings

    def encode_queries(self, queries: List[str]) -> List[List[float]]:
        """Encode queries"""
        response = self.co.embed(
            texts=queries,
            model=self.model,
            input_type="search_query",
            embedding_types=["float"]
        )
        return response.embeddings.float

# Usage
embedder = CohereEmbedding()
doc_embeddings = embedder.encode_documents(documents)
query_embeddings = embedder.encode_queries(["Search content"])
```

### Async Calls

```python
import cohere
import asyncio

async def embed_async(texts: list):
    co = cohere.AsyncClient()

    response = await co.embed(
        texts=texts,
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    return response.embeddings.float

# Usage
embeddings = asyncio.run(embed_async(["Text 1", "Text 2"]))
```

## Milvus Integration

```python
from pymilvus import MilvusClient, DataType
import cohere

# Initialize
client = MilvusClient(uri="./milvus.db")
co = cohere.Client()

# Create Collection
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("cohere_embeddings", schema=schema, index_params=index_params)

# Insert data
texts = ["Cohere is an AI company", "Milvus is a vector database"]
response = co.embed(
    texts=texts,
    model="embed-v4.0",
    input_type="search_document",
    embedding_types=["float"]
)
embeddings = response.embeddings.float

data = [{"text": t, "embedding": e} for t, e in zip(texts, embeddings)]
client.insert("cohere_embeddings", data)

# Search
query = "What is a vector database?"
query_response = co.embed(
    texts=[query],
    model="embed-v4.0",
    input_type="search_query",
    embedding_types=["float"]
)
query_embedding = query_response.embeddings.float[0]

results = client.search(
    collection_name="cohere_embeddings",
    data=[query_embedding],
    limit=10,
    output_fields=["text"]
)
```

## Limits and Notes

| Limit | Value |
|-------|-------|
| Max texts per call | 96 |
| Max tokens per text | 512 |
| Trial Key limit | 1000 calls/month |

**Notes**:
- Texts exceeding 512 tokens will be truncated
- Documents and queries need different input_type
- Trial Key has call count limits

## Cohere Special Features

### Rerank (Re-ranking)

Cohere also provides Rerank API to improve retrieval precision:

```python
# First retrieve with embedding
results = client.search(...)

# Then rerank for precision
docs = [r["entity"]["text"] for r in results[0]]
rerank_response = co.rerank(
    query="Search query",
    documents=docs,
    model="rerank-v3.5",
    top_n=5
)

# Returns reranked results
for result in rerank_response.results:
    print(f"Index: {result.index}, Score: {result.relevance_score}")
```

## Cost Optimization

1. **Use light model**: If accuracy requirements are not high
2. **Batch processing**: Max 96 per batch, reduce API calls
3. **Use binary vectors**: Smaller storage, faster search
4. **Upgrade to Production Key**: No call count limits
