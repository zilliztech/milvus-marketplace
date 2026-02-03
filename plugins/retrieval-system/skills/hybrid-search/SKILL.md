---
name: hybrid-search
description: "Use when user needs both keyword and semantic search combined. Triggers on: hybrid search, keyword + semantic, BM25, full-text search, combined search, lexical search, exact match with meaning."
---

# Hybrid Search

Combine vector semantic search and BM25 keyword search for the best of both worlds — exact keyword matching plus semantic understanding.

## When to Activate

Activate this skill when:
- User needs **both keyword precision AND semantic understanding**
- User mentions "hybrid", "BM25", "keyword + vector", "lexical + semantic"
- User has queries with **specific terms** (product codes, names, technical terms)
- User's domain has **specialized vocabulary** that semantic models miss

**Do NOT activate** when:
- User only needs semantic similarity → use `semantic-search`
- User has no keyword matching requirements
- User's queries are purely conversational/natural language

## Interactive Flow

### Step 1: Understand the Need

"Why do you need hybrid search?"

A) **Product/SKU codes**: Users search by specific identifiers
   - Example: "iPhone 15 Pro Max 256GB"
   - Keywords matter: exact model names

B) **Technical/Legal documents**: Specific terminology
   - Example: "Section 401(k) retirement plan"
   - Terms like "401(k)" must match exactly

C) **Better recall**: Don't want to miss relevant results
   - Semantic alone misses keyword matches
   - Keyword alone misses paraphrases

Which describes your use case? (A/B/C)

### Step 2: Determine Fusion Strategy

"How should we balance keyword vs semantic?"

| Strategy | When to Use | Configuration |
|----------|-------------|---------------|
| **RRF (default)** | Balanced, no tuning needed | `RRFRanker(k=60)` |
| **Weighted** | You know which matters more | `WeightedRanker(0.7, 0.3)` |

For most cases, **RRF is recommended** — it works well without tuning.

### Step 3: Confirm Configuration

"Based on your requirements:

- **Fusion strategy**: RRF (k=60)
- **Vector model**: `BAAI/bge-large-en-v1.5`
- **BM25**: Built-in tokenizer

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Two Detectives

Think of hybrid search as **two detectives** working the same case:

- **Detective Keyword (BM25)**: "I'm looking for exact clues — fingerprints, names, specific words"
- **Detective Meaning (Vector)**: "I'm looking for patterns — motive, connections, similar behaviors"

Together, they catch what either would miss alone.

```
┌─────────────────────────────────────────────────────────────┐
│                      Hybrid Search                           │
│                                                              │
│  Query: "iPhone 15 phone"                                    │
│                                                              │
│    ┌─────────────────┐     ┌─────────────────┐              │
│    │  BM25 Search    │     │  Vector Search   │              │
│    │  (Keywords)     │     │  (Semantics)     │              │
│    │                 │     │                  │              │
│    │  Looks for:     │     │  Understands:    │              │
│    │  - "iPhone"     │     │  - smartphones   │              │
│    │  - "15"         │     │  - mobile devices│              │
│    │  - "phone"      │     │  - Apple products│              │
│    └────────┬────────┘     └────────┬─────────┘              │
│             │                       │                         │
│             └───────────┬───────────┘                         │
│                         │                                     │
│                         ▼                                     │
│               ┌─────────────────┐                            │
│               │  Score Fusion   │                            │
│               │  (RRF/Weighted) │                            │
│               └────────┬────────┘                            │
│                        │                                     │
│                        ▼                                     │
│  Results: "iPhone 15 Pro Max" (keyword + semantic match)     │
│           "Latest Apple smartphone" (semantic match)          │
│           "iPhone 15 case" (keyword match)                   │
└─────────────────────────────────────────────────────────────┘
```

### Why Hybrid Beats Single Approach

| Query | Keyword Only | Semantic Only | Hybrid |
|-------|--------------|---------------|--------|
| "iPhone 15 Pro" | ✅ Finds exact | ❌ May return Samsung | ✅ Exact + similar |
| "affordable smartphone" | ❌ No "affordable" in data | ✅ Finds budget phones | ✅ Both |
| "401(k) retirement" | ✅ Matches "401(k)" | ❌ Misses special term | ✅ Both |

### Reciprocal Rank Fusion (RRF)

RRF combines rankings from both searches:

```
RRF_score = Σ 1 / (k + rank_i)
```

Where `k` is a constant (default 60) and `rank_i` is the rank in each search.

**Why RRF works**:
- Doesn't require score normalization
- Naturally handles different score scales
- Top-ranked items from either search get boosted

## Why Hybrid Over Alternatives

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| Exact terms matter (codes, names) | ✅ Hybrid | Keywords catch exact matches |
| Purely conversational queries | ⚠️ Semantic may suffice | No specific terms to match |
| Domain with unique vocabulary | ✅ Hybrid | BM25 catches domain terms |
| Maximum recall needed | ✅ Hybrid | Covers both bases |

### Trade-offs

| Aspect | Hybrid | Semantic Only |
|--------|--------|---------------|
| Latency | Higher (2 searches) | Lower |
| Complexity | More setup | Simpler |
| Recall | Higher | May miss keywords |
| Tuning | May need weight tuning | Less tuning |

## Implementation

```python
from pymilvus import (
    MilvusClient, DataType, Function, FunctionType,
    AnnSearchRequest, RRFRanker
)
from sentence_transformers import SentenceTransformer

class HybridSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.dim = 1024
        self.collection_name = "hybrid_search"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        # Create schema with both dense and sparse vectors
        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)  # BM25 sparse vector
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)

        # BM25 function - auto-generates sparse vectors from text
        bm25_fn = Function(
            name="bm25",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25
        )
        schema.add_function(bm25_fn)

        # Create indexes
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add(self, texts: list):
        """Add documents"""
        dense_embeddings = self.model.encode(texts).tolist()
        data = [{"text": t, "dense": e} for t, e in zip(texts, dense_embeddings)]
        self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query: str, limit: int = 10):
        """Hybrid search with RRF fusion"""
        dense_embedding = self.model.encode(query).tolist()

        # Dense vector search request
        dense_req = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense",
            param={"metric_type": "COSINE"},
            limit=limit * 2  # Fetch more for fusion
        )

        # BM25 sparse search request
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=limit * 2
        )

        # RRF fusion
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=limit,
            output_fields=["text"]
        )

        return [{"text": hit["entity"]["text"], "score": hit["distance"]} for hit in results[0]]

# Usage
search = HybridSearch()
search.add([
    "iPhone 15 Pro Max 256GB Natural Titanium",
    "Latest Apple smartphone with titanium design",
    "Samsung Galaxy S24 Ultra with AI features",
])

# "iPhone phone" matches keyword "iPhone" AND semantic "phone"
results = search.search("iPhone phone")
for r in results:
    print(f"{r['score']:.3f}: {r['text']}")
```

## Fusion Strategy Guide

### RRF (Recommended Default)

```python
from pymilvus import RRFRanker

# k controls how much weight goes to lower-ranked results
# Higher k = more weight to lower ranks
ranker = RRFRanker(k=60)  # Default, works well for most cases

# Lower k = top results dominate
ranker = RRFRanker(k=20)  # When you trust top results more
```

### Weighted Fusion

```python
from pymilvus import WeightedRanker

# When keyword precision is more important
ranker = WeightedRanker(0.3, 0.7)  # 30% semantic, 70% keyword

# When semantic understanding is more important
ranker = WeightedRanker(0.7, 0.3)  # 70% semantic, 30% keyword

# Balanced
ranker = WeightedRanker(0.5, 0.5)
```

### Choosing Strategy

| Scenario | Recommended | Why |
|----------|-------------|-----|
| General purpose | RRF(k=60) | No tuning needed |
| Known keyword importance | Weighted(0.3, 0.7) | Boost keywords |
| Known semantic importance | Weighted(0.7, 0.3) | Boost semantics |
| A/B testing | Both | Compare performance |

## Common Pitfalls

### ❌ Pitfall 1: Forgetting to Enable Analyzer

**Problem**: BM25 search returns nothing

**Why**: Text field not tokenized

**Fix**: Enable analyzer in schema
```python
schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
```

### ❌ Pitfall 2: Wrong Score Expectations

**Problem**: Scores look strange (e.g., very high or very low)

**Why**: RRF scores are rank-based, not similarity-based

**Fix**: Don't compare RRF scores to cosine similarity scores
```python
# RRF scores are typically small numbers like 0.03, 0.02
# This is normal - they represent fused rank scores, not similarity
```

### ❌ Pitfall 3: Over-tuning Weights

**Problem**: Spent hours tuning weights with minimal improvement

**Why**: RRF often works as well as tuned weights

**Fix**: Start with RRF, only tune weights if clear evidence suggests improvement

### ❌ Pitfall 4: Mismatched Languages

**Problem**: Chinese queries get poor BM25 results

**Why**: Default tokenizer is for English

**Fix**: Use appropriate tokenizer for your language
```python
# For Chinese, consider jieba tokenizer or multilingual models
```

## When to Level Up

| Need | Upgrade To |
|------|------------|
| Multiple text fields (title, description) | `multi-vector-search` |
| Filter by attributes too | Add filter parameter |
| Better reranking | Add `core:rerank` step |

## References

- Fusion strategy details: `references/fusion-strategies.md`
- BM25 algorithm explanation: `references/fusion-strategies.md`
- Embedding models: `core:embedding`
