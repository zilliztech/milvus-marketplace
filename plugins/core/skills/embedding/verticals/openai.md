# OpenAI text-embedding-3

OpenAI's text embedding API, the most mature commercial solution.

## Model Versions

| Model | Dimensions | Price ($/1M tokens) | Features |
|-------|-----------|---------------------|----------|
| text-embedding-3-large | 3072 | $0.13 | Highest precision |
| text-embedding-3-small | 1536 | $0.02 | Best value |
| text-embedding-ada-002 | 1536 | $0.10 | Legacy, not recommended |

## Registration and API Key

1. Visit https://platform.openai.com/signup
2. Register account (phone verification required)
3. Go to https://platform.openai.com/api-keys
4. Click "Create new secret key"
5. Copy and save API Key (shown only once)

**Note**: New accounts get $5 free credit for testing.

## Installation

```bash
pip install openai
```

## Environment Configuration

```bash
# Method 1: Environment variable
export OPENAI_API_KEY="sk-xxx..."

# Method 2: .env file
echo 'OPENAI_API_KEY=sk-xxx...' >> .env
```

## Code Examples

### Basic Usage

```python
from openai import OpenAI

client = OpenAI()  # Auto-reads OPENAI_API_KEY

# Single text
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="This is text that needs to be vectorized"
)
embedding = response.data[0].embedding
print(f"Dimensions: {len(embedding)}")  # 1536

# Batch texts (recommended)
texts = ["Text 1", "Text 2", "Text 3"]
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)
embeddings = [item.embedding for item in response.data]
```

### Dimension Reduction (Save Storage)

text-embedding-3 supports reduced dimension output:

```python
# Output 512 dimensions (from original 1536)
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Text",
    dimensions=512  # Options: 256, 512, 1024, 1536
)
```

### Batch Processing Wrapper

```python
from openai import OpenAI
from typing import List
import time

class OpenAIEmbedding:
    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = None):
        self.client = OpenAI()
        self.model = model
        self.dimensions = dimensions

    def encode(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Batch encode with automatic rate limit handling"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            params = {"model": self.model, "input": batch}
            if self.dimensions:
                params["dimensions"] = self.dimensions

            try:
                response = self.client.embeddings.create(**params)
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    time.sleep(60)  # Wait for rate limit to reset
                    response = self.client.embeddings.create(**params)
                    embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(embeddings)
                else:
                    raise

        return all_embeddings

# Usage
embedder = OpenAIEmbedding(model="text-embedding-3-small", dimensions=512)
embeddings = embedder.encode(["Text 1", "Text 2", "Text 3"])
```

### Async Batch (High Performance)

```python
import asyncio
from openai import AsyncOpenAI

async def batch_embed_async(texts: list, model: str = "text-embedding-3-small"):
    client = AsyncOpenAI()

    async def embed_batch(batch):
        response = await client.embeddings.create(model=model, input=batch)
        return [item.embedding for item in response.data]

    # Concurrent batching
    batch_size = 100
    tasks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tasks.append(embed_batch(batch))

    results = await asyncio.gather(*tasks)
    return [emb for batch in results for emb in batch]

# Usage
embeddings = asyncio.run(batch_embed_async(texts))
```

## Milvus Integration

```python
from pymilvus import MilvusClient, DataType
from openai import OpenAI

# Initialize
client = MilvusClient(uri="./milvus.db")
openai_client = OpenAI()

# Create Collection
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("openai_embeddings", schema=schema, index_params=index_params)

# Insert data
texts = ["Text 1", "Text 2", "Text 3"]
response = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
embeddings = [item.embedding for item in response.data]

data = [{"text": t, "embedding": e} for t, e in zip(texts, embeddings)]
client.insert("openai_embeddings", data)

# Search
query = "Query text"
query_response = openai_client.embeddings.create(model="text-embedding-3-small", input=[query])
query_embedding = query_response.data[0].embedding

results = client.search(
    collection_name="openai_embeddings",
    data=[query_embedding],
    limit=10,
    output_fields=["text"]
)
```

## Limits and Notes

| Limit | Value |
|-------|-------|
| Max tokens per call | 8191 |
| Max texts per call | 2048 |
| Rate limit (TPM) | 1M tokens/min (Tier 1) |

**Recommendations**:
- Chunk long texts first (see `core:chunking`)
- Watch rate limits when batch processing
- Use dimensions parameter to reduce storage

## Proxy Setup (China Users)

```python
import httpx
from openai import OpenAI

# Method 1: Environment variable
# export HTTPS_PROXY=http://127.0.0.1:7890

# Method 2: Code setup
client = OpenAI(
    http_client=httpx.Client(proxy="http://127.0.0.1:7890")
)
```

## Cost Optimization

1. **Use small model**: Small is sufficient for most scenarios, 6.5x cheaper
2. **Reduce dimensions**: 512 dimensions usually enough, saves 2/3 storage
3. **Batch processing**: Reduce API call count
4. **Cache results**: Avoid re-encoding same texts
