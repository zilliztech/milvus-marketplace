# Reranker Model Comparison

This reference provides detailed benchmarks and guidance for selecting a reranker model.

## Model Overview

### Open Source Models

| Model | Parameters | Languages | MTEB Rerank | Latency (50 docs) |
|-------|------------|-----------|-------------|-------------------|
| BAAI/bge-reranker-v2-m3 | 568M | 100+ | 68.2 | ~150ms |
| BAAI/bge-reranker-large | 560M | zh/en | 67.1 | ~120ms |
| BAAI/bge-reranker-base | 278M | zh/en | 64.8 | ~60ms |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M | en | 58.4 | ~20ms |
| cross-encoder/ms-marco-TinyBERT-L-2-v2 | 4M | en | 53.1 | ~5ms |
| jinaai/jina-reranker-v2-base-multilingual | 278M | 100+ | 65.3 | ~80ms |

### Commercial APIs

| Provider | Model | Languages | Quality | Cost (1K queries) |
|----------|-------|-----------|---------|-------------------|
| Cohere | rerank-v3.0 | en | High | $1.00 |
| Cohere | rerank-multilingual-v3.0 | 100+ | High | $1.00 |
| Voyage | rerank-2 | en | High | $0.05 |
| Jina | jina-reranker-v2 | 100+ | High | $0.02 |

## Decision Tree

```
                        ┌──────────────────┐
                        │ What's your      │
                        │ constraint?      │
                        └────────┬─────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
   ┌─────────┐             ┌─────────┐              ┌─────────┐
   │ Latency │             │ Quality │              │ No GPU  │
   │ <50ms   │             │ First   │              │         │
   └────┬────┘             └────┬────┘              └────┬────┘
        │                       │                        │
        ▼                       ▼                        ▼
   ms-marco-              bge-reranker-            Cohere API
   MiniLM-L-6             v2-m3                    or Jina API
```

## Language-Specific Recommendations

| Language | Best Model | Alternative |
|----------|------------|-------------|
| English | bge-reranker-v2-m3 | ms-marco-MiniLM |
| Chinese | bge-reranker-large | bge-reranker-v2-m3 |
| Japanese | bge-reranker-v2-m3 | jina-reranker-v2 |
| Korean | bge-reranker-v2-m3 | jina-reranker-v2 |
| Mixed/Unknown | bge-reranker-v2-m3 | Cohere multilingual |

## Implementation Examples

### BAAI BGE Reranker

```python
from sentence_transformers import CrossEncoder

# Load model
reranker = CrossEncoder('BAAI/bge-reranker-large', max_length=512)

def rerank(query: str, documents: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    """Rerank documents by relevance to query"""
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)

    # Sort by score descending
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# Usage
query = "What is the capital of France?"
docs = ["Paris is the capital of France.", "London is in England.", "France is in Europe."]
results = rerank(query, docs)
# [('Paris is the capital of France.', 0.98), ('France is in Europe.', 0.42), ...]
```

### Cohere API

```python
import cohere

co = cohere.Client("your-api-key")

def rerank_cohere(query: str, documents: list[str], top_k: int = 5) -> list[dict]:
    """Rerank using Cohere API"""
    response = co.rerank(
        model="rerank-v3.0",
        query=query,
        documents=documents,
        top_n=top_k,
        return_documents=True
    )

    return [
        {"text": r.document.text, "score": r.relevance_score, "index": r.index}
        for r in response.results
    ]
```

### Jina API

```python
import requests

def rerank_jina(query: str, documents: list[str], top_k: int = 5) -> list[dict]:
    """Rerank using Jina API"""
    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": "Bearer your-api-key"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": documents,
            "top_n": top_k
        }
    )
    return response.json()["results"]
```

## Batch Processing

For high throughput, batch queries together:

```python
from sentence_transformers import CrossEncoder
import numpy as np

class BatchReranker:
    def __init__(self, model_name: str = 'BAAI/bge-reranker-large'):
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank_batch(
        self,
        queries: list[str],
        documents_per_query: list[list[str]],
        top_k: int = 5
    ) -> list[list[tuple[str, float]]]:
        """Rerank multiple queries in a single batch"""
        # Flatten all pairs
        all_pairs = []
        query_indices = []
        for i, (query, docs) in enumerate(zip(queries, documents_per_query)):
            for doc in docs:
                all_pairs.append([query, doc])
                query_indices.append(i)

        # Single batch prediction
        all_scores = self.model.predict(all_pairs, batch_size=32, show_progress_bar=False)

        # Reconstruct per-query results
        results = [[] for _ in queries]
        for idx, pair, score in zip(query_indices, all_pairs, all_scores):
            results[idx].append((pair[1], float(score)))

        # Sort each query's results
        return [
            sorted(r, key=lambda x: x[1], reverse=True)[:top_k]
            for r in results
        ]

# Usage
reranker = BatchReranker()
queries = ["What is RAG?", "How does vector search work?"]
docs_per_query = [
    ["RAG combines retrieval with generation...", "LLMs are language models..."],
    ["Vector search uses embeddings...", "SQL databases store structured data..."]
]
results = reranker.rerank_batch(queries, docs_per_query, top_k=1)
```

## Performance Tuning

### GPU Acceleration

```python
from sentence_transformers import CrossEncoder
import torch

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model to GPU
reranker = CrossEncoder('BAAI/bge-reranker-large', device=device)

# For multi-GPU
if torch.cuda.device_count() > 1:
    reranker.model = torch.nn.DataParallel(reranker.model)
```

### Max Length Tuning

```python
# Default max_length is 512 tokens
# Reduce for faster inference, increase for longer documents

# Fast mode (shorter docs)
reranker_fast = CrossEncoder('BAAI/bge-reranker-large', max_length=256)

# Long document mode
reranker_long = CrossEncoder('BAAI/bge-reranker-large', max_length=1024)
```

### FP16 for Faster Inference

```python
from sentence_transformers import CrossEncoder
import torch

reranker = CrossEncoder('BAAI/bge-reranker-large')
reranker.model.half()  # Convert to FP16

# Note: May slightly reduce precision
```

## Benchmark Results

### Latency Comparison (50 documents, GPU)

| Model | P50 Latency | P99 Latency | Throughput |
|-------|-------------|-------------|------------|
| bge-reranker-v2-m3 | 145ms | 180ms | 6.5 qps |
| bge-reranker-large | 118ms | 150ms | 8 qps |
| bge-reranker-base | 58ms | 75ms | 16 qps |
| ms-marco-MiniLM-L-6 | 18ms | 25ms | 50 qps |
| Cohere API | 85ms | 150ms | - |

### Quality Comparison (MS MARCO Dev)

| Model | MRR@10 | NDCG@10 |
|-------|--------|---------|
| bge-reranker-v2-m3 | 0.421 | 0.489 |
| bge-reranker-large | 0.408 | 0.475 |
| Cohere rerank-v3.0 | 0.415 | 0.482 |
| ms-marco-MiniLM-L-6 | 0.365 | 0.428 |

## Cost Analysis

### Self-Hosted (GPU)

| Model | GPU Memory | $/1M queries (A100) |
|-------|------------|---------------------|
| bge-reranker-v2-m3 | 2.5 GB | $0.50 |
| bge-reranker-large | 2.3 GB | $0.40 |
| bge-reranker-base | 1.2 GB | $0.20 |
| ms-marco-MiniLM | 0.2 GB | $0.05 |

### API-Based

| Provider | $/1M queries | Rate Limit |
|----------|--------------|------------|
| Cohere | $1,000 | 10K/min |
| Voyage | $50 | 100K/day |
| Jina | $20 | 500/min (free tier) |

## Common Issues

### Out of Memory

```python
# Reduce batch size
scores = reranker.predict(pairs, batch_size=8)  # Default is 32

# Or use gradient checkpointing (slower but uses less memory)
reranker.model.gradient_checkpointing_enable()
```

### Inconsistent Scores

```python
# Normalize scores to [0, 1] range
import torch.nn.functional as F

raw_scores = reranker.predict(pairs)
normalized = F.sigmoid(torch.tensor(raw_scores)).tolist()
```

### Slow First Query

```python
# Warm up the model with a dummy query
_ = reranker.predict([["warmup", "warmup"]])
```

## Quick Reference

```python
# Recommended configurations by use case

# High Quality (accuracy-first)
HIGH_QUALITY = {
    "model": "BAAI/bge-reranker-v2-m3",
    "max_length": 512,
    "retrieve_k": 100,
    "rerank_k": 5
}

# Balanced (good default)
BALANCED = {
    "model": "BAAI/bge-reranker-large",
    "max_length": 512,
    "retrieve_k": 50,
    "rerank_k": 5
}

# Fast (latency-first)
FAST = {
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "max_length": 256,
    "retrieve_k": 30,
    "rerank_k": 5
}

# API-based (no GPU)
API_BASED = {
    "provider": "cohere",
    "model": "rerank-v3.0",
    "retrieve_k": 50,
    "rerank_k": 5
}
```
