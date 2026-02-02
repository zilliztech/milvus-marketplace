# Embedding Models for RAG

This reference helps you choose the right embedding model based on your requirements for quality, cost, speed, and language support.

## Model Comparison Matrix

### OpenAI Models

| Model | Dimensions | Max Tokens | Cost (per 1M) | Quality | Speed |
|-------|------------|------------|---------------|---------|-------|
| text-embedding-3-small | 1536 | 8191 | $0.02 | Good | Fast |
| text-embedding-3-large | 3072 | 8191 | $0.13 | Better | Medium |
| text-embedding-ada-002 | 1536 | 8191 | $0.10 | Good | Fast |

**Recommendation**: Start with `text-embedding-3-small`. Upgrade to `large` only if retrieval quality is insufficient.

### Open Source Models (Local/Self-Hosted)

| Model | Dimensions | Languages | Quality (MTEB) | GPU Required |
|-------|------------|-----------|----------------|--------------|
| BAAI/bge-large-en-v1.5 | 1024 | English | 64.23 | Yes |
| BAAI/bge-large-zh-v1.5 | 1024 | Chinese | 63.96 | Yes |
| BAAI/bge-m3 | 1024 | 100+ | 66.48 | Yes |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | English | 56.26 | No |
| sentence-transformers/all-mpnet-base-v2 | 768 | English | 57.80 | No |
| intfloat/multilingual-e5-large | 1024 | 100+ | 61.50 | Yes |

**Recommendation**:
- English only, local: `bge-large-en-v1.5`
- Chinese: `bge-large-zh-v1.5`
- Multilingual: `bge-m3` or `multilingual-e5-large`
- CPU only: `all-MiniLM-L6-v2` (quality tradeoff)

### Commercial Alternatives

| Provider | Model | Dimensions | Cost (per 1M) | Notes |
|----------|-------|------------|---------------|-------|
| Cohere | embed-english-v3.0 | 1024 | $0.10 | Good for English |
| Cohere | embed-multilingual-v3.0 | 1024 | $0.10 | 100+ languages |
| Voyage | voyage-2 | 1024 | $0.10 | High quality |
| Google | text-embedding-004 | 768 | $0.025 | Good value |

## Decision Tree

```
                    ┌─────────────────┐
                    │ What's your     │
                    │ priority?       │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    ┌─────────┐        ┌─────────┐         ┌─────────┐
    │ Cost    │        │ Quality │         │ Privacy │
    └────┬────┘        └────┬────┘         └────┬────┘
         │                  │                   │
         ▼                  ▼                   ▼
    text-embedding-    text-embedding-      bge-m3
    3-small            3-large              (self-hosted)
```

## Language-Specific Recommendations

| Language | Recommended Model | Alternative |
|----------|-------------------|-------------|
| English | text-embedding-3-small | bge-large-en |
| Chinese | bge-large-zh-v1.5 | text-embedding-3-small |
| Japanese | multilingual-e5-large | bge-m3 |
| Korean | bge-m3 | multilingual-e5-large |
| Mixed/Unknown | bge-m3 | text-embedding-3-small |
| Code | text-embedding-3-small | CodeBERT |

## Implementation Examples

### OpenAI

```python
from openai import OpenAI

client = OpenAI()

def embed_openai(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]

# Batch processing (max 2048 texts per call)
def embed_batch(texts: list[str], batch_size: int = 2048):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(embed_openai(batch))
    return embeddings
```

### BGE (Local)

```python
from sentence_transformers import SentenceTransformer

# Load model (downloads on first use)
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

def embed_bge(texts: list[str]) -> list[list[float]]:
    # BGE models recommend adding instruction prefix for queries
    return model.encode(texts, normalize_embeddings=True).tolist()

def embed_query(query: str) -> list[float]:
    # For queries, add instruction prefix
    return model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        normalize_embeddings=True
    ).tolist()
```

### BGE-M3 (Multilingual)

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def embed_m3(texts: list[str]) -> list[list[float]]:
    # Returns dict with 'dense_vecs', 'lexical_weights', 'colbert_vecs'
    result = model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    return result['dense_vecs'].tolist()
```

## Dimension Reduction

Higher dimensions don't always mean better quality. You can reduce dimensions to save storage:

```python
from openai import OpenAI

client = OpenAI()

# OpenAI supports native dimension reduction
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=["your text"],
    dimensions=1024  # Reduce from 3072 to 1024
)
```

**Tradeoffs**:
| Original Dim | Reduced Dim | Quality Loss | Storage Savings |
|--------------|-------------|--------------|-----------------|
| 3072 | 1536 | ~2% | 50% |
| 3072 | 1024 | ~5% | 67% |
| 1536 | 768 | ~3% | 50% |

## Cost Optimization

### Caching Embeddings

```python
import hashlib
import json

class EmbeddingCache:
    def __init__(self, cache_file: str = "embedding_cache.json"):
        self.cache_file = cache_file
        try:
            with open(cache_file) as f:
                self.cache = json.load(f)
        except:
            self.cache = {}

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get_or_embed(self, texts: list[str], embed_fn) -> list[list[float]]:
        results = []
        to_embed = []
        indices = []

        for i, text in enumerate(texts):
            h = self._hash(text)
            if h in self.cache:
                results.append(self.cache[h])
            else:
                to_embed.append(text)
                indices.append(i)
                results.append(None)

        if to_embed:
            new_embeddings = embed_fn(to_embed)
            for idx, text, emb in zip(indices, to_embed, new_embeddings):
                h = self._hash(text)
                self.cache[h] = emb
                results[idx] = emb

            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)

        return results
```

### Cost Estimation

```python
def estimate_cost(num_tokens: int, model: str = "text-embedding-3-small") -> float:
    costs = {
        "text-embedding-3-small": 0.02 / 1_000_000,
        "text-embedding-3-large": 0.13 / 1_000_000,
        "text-embedding-ada-002": 0.10 / 1_000_000,
    }
    return num_tokens * costs.get(model, 0)

# Example: 1M tokens ≈ 750K words ≈ 1500 pages
# Cost with small model: $0.02
```

## Quality Benchmarks

MTEB (Massive Text Embedding Benchmark) scores for retrieval tasks:

| Model | Retrieval Score | Classification | Clustering |
|-------|-----------------|----------------|------------|
| text-embedding-3-large | 64.6 | 75.5 | 49.0 |
| text-embedding-3-small | 62.3 | 74.1 | 47.5 |
| bge-large-en-v1.5 | 63.6 | 75.2 | 46.3 |
| bge-m3 | 66.5 | 76.8 | 48.9 |
| all-MiniLM-L6-v2 | 49.5 | 63.1 | 42.4 |

**Note**: Benchmark scores don't always reflect real-world performance. Test on your actual data.

## Common Mistakes

### 1. Mixing Embedding Models
**Mistake**: Using different models for indexing vs querying
**Fix**: Always use the same model for both

### 2. Not Normalizing
**Mistake**: Using raw embeddings with cosine similarity
**Fix**: Most models return normalized vectors, but verify

### 3. Ignoring Token Limits
**Mistake**: Feeding text longer than max tokens
**Fix**: Chunk before embedding, or truncate

```python
def safe_embed(text: str, max_tokens: int = 8000):
    # Rough estimate: 1 token ≈ 4 characters
    if len(text) > max_tokens * 4:
        text = text[:max_tokens * 4]
    return embed(text)
```

### 4. Not Batching
**Mistake**: Embedding one text at a time
**Fix**: Batch for efficiency

## Quick Reference

```python
# Recommended defaults by use case

# General English RAG
ENGLISH_DEFAULT = {
    "model": "text-embedding-3-small",
    "dimensions": 1536,
}

# High-quality English RAG
ENGLISH_QUALITY = {
    "model": "text-embedding-3-large",
    "dimensions": 3072,
}

# Chinese RAG
CHINESE = {
    "model": "BAAI/bge-large-zh-v1.5",
    "dimensions": 1024,
}

# Multilingual RAG
MULTILINGUAL = {
    "model": "BAAI/bge-m3",
    "dimensions": 1024,
}

# Budget/Local
LOCAL = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": 384,
}
```
