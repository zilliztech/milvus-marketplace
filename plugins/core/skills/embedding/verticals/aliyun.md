# Aliyun DashScope Embedding

Aliyun Qwen API, China-friendly with no VPN required.

## Model Versions

| Model | Dimensions | Context | Price (CNY/1K tokens) | Features |
|-------|-----------|---------|----------------------|----------|
| **text-embedding-v3** | 1024 | 8K | 0.0007 | Latest, best performance |
| text-embedding-v2 | 1536 | 2K | 0.0007 | Previous generation |
| text-embedding-v1 | 1536 | 2K | 0.0007 | Legacy |

## Registration and API Key

1. Visit https://dashscope.console.aliyun.com/
2. Log in with Aliyun account (Alipay scan supported)
3. Enable DashScope service
4. Go to https://dashscope.console.aliyun.com/apiKey
5. Create API Key

**Free quota**: 1 million tokens free for new users.

## Installation

```bash
pip install dashscope
# Or use OpenAI-compatible interface
pip install openai
```

## Environment Configuration

```bash
export DASHSCOPE_API_KEY="sk-xxx..."
```

## Code Examples

### Basic Usage (Official SDK)

```python
import dashscope
from dashscope import TextEmbedding

# Set API Key (or via environment variable)
dashscope.api_key = "sk-xxx..."

# Single text
response = TextEmbedding.call(
    model=TextEmbedding.Models.text_embedding_v3,
    input="This is text that needs to be vectorized"
)

embedding = response.output['embeddings'][0]['embedding']
print(f"Dimensions: {len(embedding)}")  # 1024

# Batch texts
texts = ["Text 1", "Text 2", "Text 3"]
response = TextEmbedding.call(
    model=TextEmbedding.Models.text_embedding_v3,
    input=texts
)

embeddings = [item['embedding'] for item in response.output['embeddings']]
```

### OpenAI-Compatible Interface

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxx...",  # DashScope API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# Single text
response = client.embeddings.create(
    model="text-embedding-v3",
    input="This is some text"
)
embedding = response.data[0].embedding

# Batch texts
response = client.embeddings.create(
    model="text-embedding-v3",
    input=["Text 1", "Text 2", "Text 3"]
)
embeddings = [item.embedding for item in response.data]
```

### Dimension Control

```python
# Reduced dimension output (saves storage)
response = TextEmbedding.call(
    model=TextEmbedding.Models.text_embedding_v3,
    input="Text",
    dimension=512  # Options: 512, 1024
)
```

### Batch Processing Wrapper

```python
import dashscope
from dashscope import TextEmbedding
from typing import List
import time

class DashScopeEmbedding:
    def __init__(self, model: str = "text-embedding-v3", dimension: int = 1024):
        self.model = model
        self.dimension = dimension

    def encode(self, texts: List[str], batch_size: int = 25) -> List[List[float]]:
        """Batch encoding"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = TextEmbedding.call(
                model=self.model,
                input=batch,
                dimension=self.dimension
            )

            if response.status_code == 200:
                embeddings = [item['embedding'] for item in response.output['embeddings']]
                all_embeddings.extend(embeddings)
            else:
                raise Exception(f"API call failed: {response.code} - {response.message}")

            # Avoid rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.1)

        return all_embeddings

# Usage
embedder = DashScopeEmbedding()
embeddings = embedder.encode(["Text 1", "Text 2", "Text 3"])
```

### Async Calls

```python
import dashscope
from dashscope import TextEmbedding
import asyncio

async def embed_async(texts: list, model: str = "text-embedding-v3"):
    """Async call"""
    # DashScope doesn't natively support async, but we can use asyncio.to_thread
    response = await asyncio.to_thread(
        TextEmbedding.call,
        model=model,
        input=texts
    )
    return [item['embedding'] for item in response.output['embeddings']]

# Usage
embeddings = asyncio.run(embed_async(["Text 1", "Text 2"]))
```

## Milvus Integration

```python
from pymilvus import MilvusClient, DataType
import dashscope
from dashscope import TextEmbedding

# Initialize
client = MilvusClient(uri="./milvus.db")
dashscope.api_key = "sk-xxx..."

# Create Collection
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("dashscope_embeddings", schema=schema, index_params=index_params)

# Insert data
texts = ["Aliyun DashScope is the Qwen API service", "Milvus is an open-source vector database"]

response = TextEmbedding.call(
    model=TextEmbedding.Models.text_embedding_v3,
    input=texts
)
embeddings = [item['embedding'] for item in response.output['embeddings']]

data = [{"text": t, "embedding": e} for t, e in zip(texts, embeddings)]
client.insert("dashscope_embeddings", data)

# Search
query = "What is Qwen?"
query_response = TextEmbedding.call(
    model=TextEmbedding.Models.text_embedding_v3,
    input=[query]
)
query_embedding = query_response.output['embeddings'][0]['embedding']

results = client.search(
    collection_name="dashscope_embeddings",
    data=[query_embedding],
    limit=10,
    output_fields=["text"]
)
```

## Limits and Notes

| Limit | Value |
|-------|-------|
| Max texts per call | 25 |
| Max text length | 8K tokens (v3) / 2K (v1/v2) |
| Rate limit | 120 RPM |
| Concurrency limit | 10 |

**Notes**:
- Batch processing max 25 items
- China-based service with low latency
- Supports Aliyun internal network calls (faster and cheaper)

## Integration with Other Aliyun Products

### Aliyun Elasticsearch

```python
# Aliyun ES has built-in DashScope embedding
# Can be configured directly in ES without code calls
```

### Function Compute (FC)

```python
# Using in Aliyun Function Compute (internal network call)
import dashscope

# Internal network calls are faster and cheaper
dashscope.base_http_api_url = "https://dashscope.aliyuncs-inc.com/api/v1"
```

## Cost Estimation

| Scenario | Document Count | Estimated Cost |
|----------|---------------|----------------|
| Small knowledge base | 10K | ~7 CNY |
| Medium application | 100K | ~70 CNY |
| Large application | 1M | ~700 CNY |

*Assuming average 1000 tokens per document*

## Selection Recommendations

| Scenario | Recommendation |
|----------|---------------|
| China users, no VPN | DashScope |
| Already using Aliyun | DashScope |
| Highest accuracy | OpenAI / BGE-M3 |
| Long context needed | text-embedding-v3 (8K) |

## Why Choose DashScope?

1. **China-friendly**: No VPN required, low latency
2. **Low cost**: 0.0007 CNY/1K tokens
3. **Free quota**: 1M tokens for new users
4. **OpenAI-compatible**: Easy code migration
5. **Aliyun ecosystem**: Good integration with other Aliyun services

## FAQ

### Q: What's the difference from Qwen3-Embedding open-source model?

A: DashScope is an API service requiring no deployment; Qwen3-Embedding is an open-source model requiring self-deployment. Performance-wise, DashScope API and Qwen3 open-source are not identical - the API version may have additional optimizations.

### Q: Does it support image embedding?

A: Currently DashScope embedding API only supports text. For image embedding, use Qwen VL model or local CLIP.
