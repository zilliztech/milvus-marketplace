---
name: filtered-search
description: "Use when user needs vector search with scalar field filtering. Triggers on: filtered search, filter by category, metadata filter, faceted search, conditional search, attribute filtering."
---

# Filtered Search - Vector Search with Filtering

Vector semantic search + scalar field filtering, filter before search or simultaneously.

## Use Cases

- E-commerce: Search after filtering by price/category/brand
- Recruitment: Match after filtering by location/salary/experience
- Real Estate: Recommend after filtering by area/price/layout
- Content Platforms: Search after filtering by time/tags/author

## Architecture

```
Query + Filter Conditions → Scalar Filtering → Vector Search → Results
```

## Complete Implementation

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
        index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 256})
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
# Equals
'category == "electronics"'

# Comparison
'price >= 100'
'price < 1000'

# Range
'price >= 100 and price <= 1000'

# IN
'category in ["phones", "laptops", "tablets"]'

# Array contains
'array_contains(tags, "new")'

# Combination
'category == "phones" and price >= 1000 and array_contains(tags, "5G")'

# String matching
'text like "iPhone%"'  # Prefix match
```

## Index Strategies

| Field Type | Index Type | Use Case |
|-----------|-----------|----------|
| VARCHAR (low cardinality) | TRIE | Categories, status |
| VARCHAR (high cardinality) | INVERTED | Tags, keywords |
| INT/FLOAT | STL_SORT | Numeric range queries |
| ARRAY | INVERTED | Array contains queries |

## Vertical Applications

See detailed guides in `verticals/` directory:
- `ecommerce.md` - E-commerce (category/price/brand)
- `recruitment.md` - Recruitment (location/salary/experience)
- `real-estate.md` - Real Estate (area/price/layout)

## Related Tools

- Data processing orchestration: `core:ray`
- Vectorization: `core:embedding`
- Indexing: `core:indexing`
