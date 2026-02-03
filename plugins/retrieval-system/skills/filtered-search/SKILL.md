---
name: filtered-search
description: "Use when user needs vector search with scalar field filtering. Triggers on: filtered search, filter by category, metadata filter, faceted search, conditional search, attribute filtering, search with constraints."
---

# Filtered Search

Vector semantic search combined with scalar field filtering — find semantically relevant results that also match specific criteria.

## When to Activate

Activate this skill when:
- User needs to **constrain search results** by attributes (category, price, date, etc.)
- User mentions "filter by", "only show", "within range", "where category is"
- User has **structured metadata** alongside text content
- User wants faceted search like e-commerce product filtering

**Do NOT activate** when:
- User only needs pure semantic search → use `semantic-search`
- User needs keyword + semantic fusion → use `hybrid-search`
- User has no metadata fields to filter on

## Interactive Flow

### Step 1: Identify Filter Fields

"What attributes do you need to filter by?"

| Common Filter Types | Examples |
|---------------------|----------|
| **Category/Type** | category = "electronics", status = "active" |
| **Numeric Range** | price between 100-500, rating >= 4.0 |
| **Date/Time** | created_at > "2024-01-01", within last 7 days |
| **Tags/Arrays** | tags contains "new", skills includes "python" |
| **Boolean** | in_stock = true, is_verified = true |

Which filter types do you need? (can select multiple)

### Step 2: Understand Filter Cardinality

"For each filter field, how many unique values exist?"

| Cardinality | Example | Index Strategy |
|-------------|---------|----------------|
| Low (< 100) | category, status | TRIE index |
| Medium (100-10K) | brand, city | INVERTED index |
| High (> 10K) | user_id, timestamp | STL_SORT or no index |

This affects index design and query performance.

### Step 3: Confirm Schema Design

"Based on your requirements, here's the proposed schema:

```python
# Your schema
schema.add_field('category', DataType.VARCHAR, max_length=256)  # TRIE index
schema.add_field('price', DataType.FLOAT)  # STL_SORT index
schema.add_field('tags', DataType.ARRAY, element_type=DataType.VARCHAR)  # INVERTED
```

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Department Store

Think of filtered search as a **department store with sections**:
- Pure semantic search = "Find me something comfortable" (searches everywhere)
- Filtered search = "Find me something comfortable **in the shoe department, under $100**"

```
┌─────────────────────────────────────────────────────────┐
│                    Filtered Search                       │
│                                                          │
│  Query: "comfortable work shoes"                         │
│  Filters: category="shoes", price<=100                   │
│                                                          │
│           ┌─────────────────────────────┐               │
│           │     Step 1: Filter First     │               │
│           │                              │               │
│           │  Full Collection (1M items)  │               │
│           │           │                  │               │
│           │           ▼                  │               │
│           │  category="shoes" (50K)      │               │
│           │           │                  │               │
│           │           ▼                  │               │
│           │  price <= 100 (10K)          │               │
│           └──────────┬──────────────────┘               │
│                      │                                   │
│                      ▼                                   │
│           ┌─────────────────────────────┐               │
│           │  Step 2: Vector Search       │               │
│           │  on 10K filtered items       │               │
│           │                              │               │
│           │  → Semantically match        │               │
│           │    "comfortable work shoes"  │               │
│           └──────────┬──────────────────┘               │
│                      │                                   │
│                      ▼                                   │
│           Top 10 relevant results                        │
└─────────────────────────────────────────────────────────┘
```

### Filter vs Post-Filter

| Approach | When Applied | Performance |
|----------|--------------|-------------|
| **Pre-filter** | Before vector search | ✅ Efficient (searches fewer vectors) |
| **Post-filter** | After vector search | ⚠️ May miss results if limit is small |

Milvus uses **pre-filtering** by default — filters are applied before ANN search.

## Why Filtered Search

| Scenario | Without Filtering | With Filtering |
|----------|-------------------|----------------|
| E-commerce: "laptop under $1000" | Returns $2000 laptops too | Only budget options |
| Job search: "Python developer in NYC" | Returns SF jobs too | Location-specific |
| Content: "AI news from this week" | Returns old articles | Recent only |

### When NOT to Use Filtering

- **Filter reduces results too much**: If filter leaves < 100 items, vector search adds little value
- **Filter is the only criteria**: Just use scalar query, no need for vectors
- **Dynamic filters change frequently**: Consider separate indexes

## Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

class FilteredSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.collection_name = "filtered_search"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("category", DataType.VARCHAR, max_length=256)
        schema.add_field("price", DataType.FLOAT)
        schema.add_field("tags", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20, max_length=64)
        schema.add_field("created_at", DataType.INT64)  # Timestamp
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        # Scalar indexes (accelerate filtering)
        index_params.add_index(field_name="category", index_type="TRIE")
        index_params.add_index(field_name="price", index_type="STL_SORT")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add(self, items: list):
        """Add data
        items: [{"text": "...", "category": "...", "price": 100, "tags": [...], "created_at": 1234567890}]
        """
        texts = [item["text"] for item in items]
        embeddings = self.model.encode(texts).tolist()
        for item, emb in zip(items, embeddings):
            item["embedding"] = emb
        self.client.insert(collection_name=self.collection_name, data=items)

    def search(self, query: str, filters: dict = None, limit: int = 10):
        """Search with filtering
        filters: {
            "category": "electronics",
            "price_min": 100,
            "price_max": 1000,
            "tags": ["new"],
            "created_after": 1234567890
        }
        """
        embedding = self.model.encode(query).tolist()

        # Build filter expression
        expr_parts = []
        if filters:
            if "category" in filters:
                expr_parts.append(f'category == "{filters["category"]}"')
            if "price_min" in filters:
                expr_parts.append(f'price >= {filters["price_min"]}')
            if "price_max" in filters:
                expr_parts.append(f'price <= {filters["price_max"]}')
            if "tags" in filters:
                for tag in filters["tags"]:
                    expr_parts.append(f'array_contains(tags, "{tag}")')
            if "created_after" in filters:
                expr_parts.append(f'created_at >= {filters["created_after"]}')

        filter_expr = " and ".join(expr_parts) if expr_parts else ""

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            filter=filter_expr,
            limit=limit,
            output_fields=["text", "category", "price", "tags"]
        )

        return [{"text": hit["entity"]["text"],
                 "category": hit["entity"]["category"],
                 "price": hit["entity"]["price"],
                 "score": hit["distance"]} for hit in results[0]]

# Usage
search = FilteredSearch()

# Add products
search.add([
    {"text": "iPhone 15 Pro 256GB", "category": "phones", "price": 999, "tags": ["new", "5G"], "created_at": 1704067200},
    {"text": "MacBook Air M3", "category": "laptops", "price": 1099, "tags": ["new", "lightweight"], "created_at": 1704067200},
    {"text": "AirPods Pro 2", "category": "headphones", "price": 249, "tags": ["noise-canceling"], "created_at": 1704067200},
])

# Search: phones category, price 500-1500
results = search.search(
    query="Apple smartphone",
    filters={"category": "phones", "price_min": 500, "price_max": 1500}
)
```

## Filter Expression Syntax

```python
# Comparison operators
'price == 100'              # Equals
'price != 100'              # Not equals
'price > 100'               # Greater than
'price >= 100'              # Greater or equal
'price < 100'               # Less than
'price <= 100'              # Less or equal

# String matching
'category == "electronics"' # Exact match
'title like "iPhone%"'      # Prefix match
'title like "%Pro%"'        # Contains

# IN operator
'category in ["phones", "laptops", "tablets"]'

# Array operations
'array_contains(tags, "new")'           # Array contains value
'array_contains_all(tags, ["a", "b"])'  # Contains all
'array_contains_any(tags, ["a", "b"])'  # Contains any

# Logical operators
'price >= 100 and price <= 1000'        # AND
'category == "phones" or category == "tablets"'  # OR
'not (price > 1000)'                     # NOT

# Complex expression
'category == "phones" and price >= 500 and array_contains(tags, "5G")'
```

## Index Strategy Guide

| Field Type | Cardinality | Index Type | Use Case |
|-----------|-------------|------------|----------|
| VARCHAR | Low (< 100) | **TRIE** | Category, status, type |
| VARCHAR | High | **INVERTED** | Tags, keywords |
| INT/FLOAT | Any | **STL_SORT** | Numeric ranges |
| ARRAY | Any | **INVERTED** | Array contains |
| BOOL | 2 | None needed | Boolean flags |

### Index Creation Example

```python
index_params = self.client.prepare_index_params()

# Vector index (required)
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

# Scalar indexes (optional but recommended for frequently filtered fields)
index_params.add_index("category", index_type="TRIE")      # Low cardinality string
index_params.add_index("price", index_type="STL_SORT")     # Numeric range
index_params.add_index("tags", index_type="INVERTED")      # Array contains
```

## Common Pitfalls

### ❌ Pitfall 1: Over-Filtering

**Problem**: Filter returns 0 results

**Why**: Filter conditions too restrictive

**Fix**: Check filter cardinality before searching
```python
# First check how many items match the filter
count = client.query(
    collection_name="products",
    filter='category == "rare_category" and price < 10',
    output_fields=["count(*)"]
)
# If count is 0, relax filters
```

### ❌ Pitfall 2: Missing Scalar Index

**Problem**: Filtered search is slow

**Why**: No index on frequently filtered field

**Fix**: Add appropriate index type
```python
# Add index for the filtered field
index_params.add_index("category", index_type="TRIE")
```

### ❌ Pitfall 3: Wrong Index Type

**Problem**: Index doesn't improve performance

**Why**: Using TRIE for high-cardinality field

**Fix**: Match index type to cardinality
- Low cardinality → TRIE
- High cardinality → INVERTED
- Numeric range → STL_SORT

### ❌ Pitfall 4: SQL Injection in Filters

**Problem**: User input directly in filter expression

**Why**: Security vulnerability

**Fix**: Validate and sanitize user input
```python
# BAD - vulnerable to injection
filter_expr = f'category == "{user_input}"'

# GOOD - validate input
VALID_CATEGORIES = ["phones", "laptops", "tablets"]
if user_input in VALID_CATEGORIES:
    filter_expr = f'category == "{user_input}"'
```

## When to Level Up

| Need | Upgrade To |
|------|------------|
| Keyword matching + filters | `hybrid-search` with filters |
| Multiple text fields | `multi-vector-search` with filters |
| Complex multi-hop queries | `agentic-rag` with tool-based filtering |

## References

- Filter expression syntax: `references/filter-optimization.md`
- Index configuration: `core:indexing`
- Vertical guides: `verticals/`
