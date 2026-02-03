---
name: multi-vector-search
description: "Use when user needs to search across multiple vector fields. Triggers on: multi-vector, multiple embeddings, multi-field search, title + content, combined vectors, different aspects of same item."
---

# Multi-Vector Search

Search across multiple vector fields simultaneously — find items that match across different semantic aspects like title, description, and reviews.

## When to Activate

Activate this skill when:
- User has **multiple text fields** per item (title + description, question + answer)
- User wants to **search different aspects** with different weights
- User mentions "multi-vector", "title and content", "multiple embeddings"
- User's items have **semantically distinct parts** that should be searched together

**Do NOT activate** when:
- User has single text field → use `semantic-search`
- User needs keyword + semantic → use `hybrid-search`
- User has image + text → use `multimodal-retrieval`

## Interactive Flow

### Step 1: Identify Vector Fields

"What text fields do you have per item?"

| Common Patterns | Fields |
|-----------------|--------|
| **Products** | title, description, reviews |
| **Documents** | title, abstract, body |
| **Q&A** | question, answer |
| **Resumes** | skills, experience, education |
| **News** | headline, body |

List your fields: ___

### Step 2: Determine Field Importance

"How important is each field for search relevance?"

| Field | Importance | Suggested Weight |
|-------|------------|------------------|
| Title | High (exact matches) | 0.4 - 0.5 |
| Description | Medium (detailed info) | 0.3 - 0.4 |
| Reviews | Low (supplementary) | 0.1 - 0.2 |

**Note**: Weights should sum to 1.0

### Step 3: Confirm Configuration

"Based on your requirements:

- **Vector fields**: title_embedding, content_embedding
- **Weights**: 0.5, 0.5 (balanced)
- **Fusion**: RRF (recommended)

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Multi-Criteria Job Interview

Think of multi-vector search as **evaluating a candidate on multiple criteria**:

- Resume alone might miss communication skills
- Interview alone might miss technical depth
- Both together give complete picture

```
┌─────────────────────────────────────────────────────────┐
│                  Multi-Vector Search                     │
│                                                          │
│  Item: Product Listing                                   │
│  ┌─────────────┬─────────────┬─────────────┐           │
│  │   Title     │ Description │   Reviews   │           │
│  │  "iPhone    │ "Latest     │ "Great      │           │
│  │   15 Pro"   │  Apple..."  │  camera!"   │           │
│  └──────┬──────┴──────┬──────┴──────┬──────┘           │
│         │             │             │                   │
│         ▼             ▼             ▼                   │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│    │ Embed   │  │ Embed   │  │ Embed   │              │
│    └────┬────┘  └────┬────┘  └────┬────┘              │
│         │            │            │                    │
│         ▼            ▼            ▼                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │title_vec │ │ desc_vec │ │review_vec│              │
│  │ (1024d)  │ │ (1024d)  │ │ (1024d)  │              │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘              │
│       │            │            │                     │
│       └────────────┼────────────┘                     │
│                    │                                   │
│                    ▼                                   │
│  Query: "good camera phone"                            │
│                    │                                   │
│           ┌───────┴────────┐                          │
│           │  Search Each   │                          │
│           │  Vector Field  │                          │
│           └───────┬────────┘                          │
│                   │                                    │
│                   ▼                                    │
│           ┌───────────────┐                           │
│           │ Score Fusion  │  RRF or Weighted          │
│           └───────┬───────┘                           │
│                   │                                    │
│                   ▼                                    │
│           Combined Results                             │
│           (matches across all aspects)                 │
└─────────────────────────────────────────────────────────┘
```

### Why Multiple Vectors?

| Single Vector | Multi-Vector |
|---------------|--------------|
| Embeds entire item as one | Preserves distinct semantic aspects |
| Title dominates or gets lost | Title and description weighted separately |
| Can't tune importance | Can adjust weights per field |
| Simpler | More flexible |

### When Multi-Vector Helps

| Scenario | Problem with Single Vector | Multi-Vector Solution |
|----------|---------------------------|----------------------|
| Product search | Long description dilutes title | Title vector captures exact name |
| Academic papers | Abstract and body mixed | Search abstract for overview, body for details |
| Q&A matching | Question semantics lost in answer | Match question-to-question separately |

## Implementation

```python
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer

class MultiVectorSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.dim = 1024
        self.collection_name = "multi_vector_search"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("title", DataType.VARCHAR, max_length=1024)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("title_embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("content_embedding", DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="title_embedding", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="content_embedding", index_type="AUTOINDEX", metric_type="COSINE")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add(self, items: list):
        """Add data
        items: [{"title": "...", "content": "..."}]
        """
        titles = [item["title"] for item in items]
        contents = [item["content"] for item in items]

        title_embeddings = self.model.encode(titles).tolist()
        content_embeddings = self.model.encode(contents).tolist()

        data = []
        for item, title_emb, content_emb in zip(items, title_embeddings, content_embeddings):
            data.append({
                "title": item["title"],
                "content": item["content"],
                "title_embedding": title_emb,
                "content_embedding": content_emb
            })

        self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query: str, limit: int = 10, mode: str = "balanced"):
        """Multi-vector search
        mode: "balanced" | "title" | "content"
        """
        query_embedding = self.model.encode(query).tolist()

        # Title vector search
        title_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="title_embedding",
            param={"metric_type": "COSINE"},
            limit=limit * 2
        )

        # Content vector search
        content_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="content_embedding",
            param={"metric_type": "COSINE"},
            limit=limit * 2
        )

        # RRF fusion
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[title_req, content_req],
            ranker=RRFRanker(k=60),
            limit=limit,
            output_fields=["title", "content"]
        )

        return [{"title": hit["entity"]["title"],
                 "content": hit["entity"]["content"][:200] + "...",
                 "score": hit["distance"]} for hit in results[0]]

    def search_title_only(self, query: str, limit: int = 10):
        """Fast search on title only"""
        embedding = self.model.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            anns_field="title_embedding",
            limit=limit,
            output_fields=["title", "content"]
        )
        return [{"title": hit["entity"]["title"], "score": hit["distance"]} for hit in results[0]]

# Usage
search = MultiVectorSearch()

# Add articles
search.add([
    {
        "title": "Milvus Vector Database Getting Started",
        "content": "Milvus is an open-source vector database designed for AI applications..."
    },
    {
        "title": "RAG System Architecture Design",
        "content": "Retrieval-Augmented Generation combines retrieval and generation..."
    }
])

# Multi-vector search
results = search.search("how to use vector database")

# Title-only search (faster)
results = search.search_title_only("Milvus")
```

## Weight Strategies

### Scenario-Based Weights

| Scenario | Title | Description | Reviews | Why |
|----------|-------|-------------|---------|-----|
| **Product Lookup** | 0.5 | 0.3 | 0.2 | Users often know product name |
| **Feature Search** | 0.3 | 0.5 | 0.2 | Looking for specific features |
| **Review-Based** | 0.2 | 0.3 | 0.5 | "What do people say about X" |
| **Academic Papers** | 0.4 | 0.4 | 0.2 | Title and abstract equally important |

### Dynamic Weight Selection

```python
def get_weights_for_query(query: str) -> tuple:
    """Automatically select weights based on query characteristics."""

    query_lower = query.lower()
    word_count = len(query.split())

    # Short queries → likely looking for specific item by name
    if word_count <= 3:
        return (0.7, 0.3)  # Favor title

    # Questions → need detailed content
    if query_lower.startswith(('how', 'what', 'why', 'when')):
        return (0.3, 0.7)  # Favor content

    # Default balanced
    return (0.5, 0.5)
```

### Using Weighted Ranker

```python
from pymilvus import WeightedRanker

# Instead of RRF, use weighted fusion
ranker = WeightedRanker(0.6, 0.4)  # 60% title, 40% content

results = self.client.hybrid_search(
    collection_name=self.collection_name,
    reqs=[title_req, content_req],
    ranker=ranker,  # Weighted instead of RRF
    limit=limit,
    output_fields=["title", "content"]
)
```

## Performance Optimization

### Adaptive Search Strategy

```python
def smart_search(self, query: str, mode: str = "auto"):
    """Automatically choose search strategy."""

    if mode == "auto":
        # Short query → title only (faster)
        if len(query.split()) <= 2:
            return self.search_title_only(query)
        # Long query → full multi-vector
        else:
            return self.search(query)
    elif mode == "title":
        return self.search_title_only(query)
    else:
        return self.search(query)
```

### Embedding Caching

```python
from functools import lru_cache

class OptimizedMultiVectorSearch(MultiVectorSearch):
    @lru_cache(maxsize=1000)
    def _get_embedding(self, text: str) -> tuple:
        """Cache embeddings for repeated queries."""
        return tuple(self.model.encode(text).tolist())

    def search(self, query: str, limit: int = 10):
        query_embedding = list(self._get_embedding(query))
        # ... rest of search logic
```

## Common Pitfalls

### ❌ Pitfall 1: Too Many Vector Fields

**Problem**: Created 10 vector fields, search is slow

**Why**: Each field requires a separate ANN search

**Fix**: Limit to 2-4 most important fields
```python
# BAD - too many fields
schema.add_field("title_vec", ...)
schema.add_field("subtitle_vec", ...)
schema.add_field("description_vec", ...)
schema.add_field("summary_vec", ...)
schema.add_field("tags_vec", ...)
schema.add_field("category_vec", ...)

# GOOD - consolidated
schema.add_field("title_vec", ...)  # Title + subtitle
schema.add_field("content_vec", ...)  # Description + summary
```

### ❌ Pitfall 2: Same Embedding for Different Length Texts

**Problem**: Title vector and body vector have similar embeddings

**Why**: Model truncates long text, short text embeds fully

**Fix**: Consider different models or chunk long content
```python
# For very long content, consider chunking
chunks = [content[i:i+500] for i in range(0, len(content), 500)]
chunk_embeddings = self.model.encode(chunks)
# Store best chunk or average
```

### ❌ Pitfall 3: Ignoring Empty Fields

**Problem**: Items with empty descriptions cause errors

**Why**: Empty string produces invalid embedding

**Fix**: Handle empty fields
```python
def _safe_embed(self, text: str) -> list:
    if not text or not text.strip():
        return [0.0] * self.dim  # Zero vector for empty
    return self.model.encode(text).tolist()
```

### ❌ Pitfall 4: Mismatched Weights

**Problem**: Weights don't sum to 1.0

**Why**: Can cause unexpected score scaling

**Fix**: Always normalize weights
```python
def normalize_weights(*weights):
    total = sum(weights)
    return tuple(w / total for w in weights)

weights = normalize_weights(0.6, 0.3, 0.2)  # (0.55, 0.27, 0.18)
```

## When to Level Up

| Need | Upgrade To |
|------|------------|
| Add keyword matching | Combine with `hybrid-search` |
| Filter by attributes | Add filter parameter |
| Different modalities (image + text) | `multimodal-retrieval` |
| Better precision | Add `core:rerank` step |

## References

- Fusion strategies: `hybrid-search/references/fusion-strategies.md`
- Embedding models: `core:embedding`
- Vertical guides: `verticals/`
