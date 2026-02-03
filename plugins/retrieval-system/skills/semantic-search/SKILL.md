---
name: semantic-search
description: "Use when user wants to build semantic/text search. Triggers on: semantic search, text search, full-text search, natural language search, find similar text, vector search, meaning-based search, conceptual search."
---

# Semantic Search

Build vector-based semantic search systems that understand meaning, not just keywords.

## When to Activate

Activate this skill when:
- User wants to search text by **meaning** rather than exact keywords
- User mentions "find similar", "semantic", "natural language search"
- User has a collection of documents/texts to make searchable
- User's search queries should understand synonyms and related concepts

**Do NOT activate** when:
- User needs **exact keyword matching** → use `hybrid-search`
- User has **multiple data types** (text + image) → use `multimodal-retrieval`
- User needs **filtering by attributes** → use `filtered-search`

## Interactive Flow

### Step 1: Understand the Use Case

"What type of content will users search?"

A) **Short texts** (titles, product names, questions)
   - Typically < 200 characters
   - Users expect quick, precise matches

B) **Long documents** (articles, papers, documentation)
   - Need chunking strategy
   - May need to return specific passages

C) **Conversational queries** (customer support, FAQ)
   - Queries are natural language questions
   - Answers should be semantically relevant

Which describes your use case? (A/B/C)

### Step 2: Clarify Scale and Latency

"What's your expected scale?"

| Scale | Documents | Latency Target |
|-------|-----------|----------------|
| Small | < 100K | < 50ms |
| Medium | 100K - 10M | < 100ms |
| Large | > 10M | < 200ms |

"For your scale, I'll configure appropriate index parameters."

### Step 3: Confirm Before Implementation

"Based on your requirements:

- **Embedding model**: `BAAI/bge-large-en-v1.5` (1024 dim)
- **Index type**: AUTOINDEX
- **Metric**: COSINE similarity

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Library Catalog

Think of semantic search like a **smart librarian**:
- Traditional search = looking for exact words in book titles
- Semantic search = understanding "I want books about cooking" includes recipes, cuisine, culinary arts

```
┌─────────────────────────────────────────────────────────┐
│                    Semantic Search                       │
│                                                          │
│  Query: "affordable laptop"                              │
│                    │                                     │
│                    ▼                                     │
│           ┌───────────────┐                             │
│           │   Embedding   │  Convert text to vector     │
│           │     Model     │  (1024 dimensions)          │
│           └───────┬───────┘                             │
│                   │                                     │
│                   ▼                                     │
│           [0.12, -0.45, 0.78, ...]                      │
│                   │                                     │
│                   ▼                                     │
│      ┌────────────────────────┐                        │
│      │      Vector Index      │  Find nearest vectors   │
│      │        (Milvus)        │  in high-dim space      │
│      └────────────┬───────────┘                        │
│                   │                                     │
│                   ▼                                     │
│  Results: "budget-friendly notebook", "cheap computer"  │
│  (semantically similar, different keywords)             │
└─────────────────────────────────────────────────────────┘
```

### Why Vectors Work

| Concept | Explanation |
|---------|-------------|
| **Embedding** | Text → High-dimensional vector that captures meaning |
| **Similarity** | Vectors close together = similar meaning |
| **COSINE** | Measures angle between vectors (0-1, higher = more similar) |

## Why Semantic Search Over Alternatives

| Need | Solution | Why |
|------|----------|-----|
| "Find documents about X" | ✅ Semantic Search | Understands meaning |
| "Find documents containing 'X'" | ❌ Use keyword search | Exact match needed |
| "Find documents about X in category Y" | ⚠️ Consider filtered-search | Need attribute filtering |
| "Find documents matching both keywords AND meaning" | ⚠️ Consider hybrid-search | Both precision and recall |

### Limitations to Know

1. **No keyword precision**: "iPhone 15" query might return "Samsung Galaxy" (semantically similar)
2. **Language dependency**: Models trained on specific languages work best for those
3. **Domain shift**: General models may miss domain-specific terminology

## Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, collection_name: str = "semantic_search", uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.dim = 1024
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add(self, texts: list):
        """Add documents"""
        embeddings = self.model.encode(texts).tolist()
        data = [{"text": text, "embedding": emb} for text, emb in zip(texts, embeddings)]
        self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query: str, limit: int = 10):
        """Search"""
        query_embedding = self.model.encode([query]).tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            data=query_embedding,
            limit=limit,
            output_fields=["text"]
        )

        return [
            {"text": hit["entity"]["text"], "score": hit["distance"]}
            for hit in results[0]
        ]

# Usage
search = SemanticSearch()
search.add(["Python is a programming language", "Java is also a programming language", "Machine learning is popular"])
results = search.search("what is programming")
for r in results:
    print(f"{r['score']:.3f}: {r['text']}")
```

## Configuration Guide

### Embedding Model Selection

| Language | Model | Dimensions | Quality | Speed |
|----------|-------|------------|---------|-------|
| English | `BAAI/bge-large-en-v1.5` | 1024 | ★★★★★ | ★★★ |
| English | `BAAI/bge-base-en-v1.5` | 768 | ★★★★ | ★★★★ |
| Chinese | `BAAI/bge-large-zh-v1.5` | 1024 | ★★★★★ | ★★★ |
| Multilingual | `BAAI/bge-m3` | 1024 | ★★★★ | ★★★ |
| API-based | `text-embedding-3-small` | 1536 | ★★★★ | ★★★★★ |

### Similarity Threshold Guidelines

| Similarity Score | Interpretation | Action |
|-----------------|----------------|--------|
| > 0.9 | Near identical | High confidence match |
| 0.7 - 0.9 | Strong match | Good results |
| 0.5 - 0.7 | Related | May need verification |
| < 0.5 | Weak match | Likely irrelevant |

## Common Pitfalls

### ❌ Pitfall 1: Expecting Keyword Precision

**Problem**: User searches "iPhone 15 Pro Max" but gets "Samsung Galaxy S24"

**Why**: Semantic search finds conceptually similar items, not exact matches

**Fix**: Use `hybrid-search` to combine keyword + semantic matching

### ❌ Pitfall 2: Not Chunking Long Documents

**Problem**: Searching a 10-page document returns nothing relevant

**Why**: Embedding models have token limits; long text gets truncated

**Fix**:
```python
# Split into chunks before indexing
chunks = [doc[i:i+500] for i in range(0, len(doc), 500)]
search.add(chunks)
```

### ❌ Pitfall 3: Wrong Language Model

**Problem**: Chinese queries return poor results

**Why**: Using English-trained model for Chinese text

**Fix**: Use language-appropriate model (e.g., `bge-large-zh-v1.5` for Chinese)

### ❌ Pitfall 4: Too Many Results

**Problem**: Returning 100 results when user needs top 3

**Why**: No relevance threshold filtering

**Fix**:
```python
# Filter by similarity score
results = [r for r in results if r["score"] > 0.7][:3]
```

## When to Level Up

Consider upgrading when you need:

| Need | Upgrade To |
|------|------------|
| Keyword + semantic matching | `hybrid-search` |
| Filter by category/price/date | `filtered-search` |
| Search across title + description | `multi-vector-search` |
| Higher precision results | Add `core:rerank` |

## References

- Chunking strategies: `core:chunking`
- Embedding model details: `core:embedding`
- Index configuration: `core:indexing`
- Similarity metrics comparison: `references/similarity-metrics.md`
