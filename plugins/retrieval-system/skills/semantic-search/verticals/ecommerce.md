# E-commerce Product Search

## Data Characteristics

- Short product titles (usually < 100 characters)
- Rich attributes (brand, price, category, etc.)
- Diverse search queries (colloquial, abbreviations, typos)
- High real-time requirements

## Recommended Configuration

| Config Item | Recommended Value | Notes |
|-------------|-------------------|-------|
| Embedding Model | `BAAI/bge-large-zh-v1.5` | Chinese e-commerce |
| | `GanymedeNil/text2vec-large-chinese` | Alternative |
| Index Type | IVF_PQ | Large data volume |
| | HNSW | High accuracy requirements |
| nlist (IVF) | sqrt(n) | n is data volume |

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

# Key filter fields
schema.add_field("category_l1", DataType.VARCHAR, max_length=64)   # Level 1 category
schema.add_field("category_l2", DataType.VARCHAR, max_length=64)   # Level 2 category
schema.add_field("brand", DataType.VARCHAR, max_length=128)
schema.add_field("price", DataType.FLOAT)
schema.add_field("sales", DataType.INT32)                          # Sales volume
schema.add_field("rating", DataType.FLOAT)                         # Rating
schema.add_field("in_stock", DataType.BOOL)                        # In stock
schema.add_field("shop_id", DataType.VARCHAR, max_length=64)

# Index
index_params.add_index("embedding", index_type="IVF_PQ", metric_type="IP",
                       params={"nlist": 1024, "m": 16, "nbits": 8})
index_params.add_index("category_l1", index_type="TRIE")
index_params.add_index("brand", index_type="TRIE")
index_params.add_index("price", index_type="STL_SORT")
index_params.add_index("sales", index_type="STL_SORT")
```

## Search Optimization

### 1. Query Rewriting

```python
def rewrite_query(query: str) -> str:
    """Handle colloquial queries"""
    # Abbreviation expansion
    abbreviations = {
        "power bank": "portable charger",
        "bluetooth earbuds": "wireless bluetooth earphones",
        "phone case": "mobile phone protective case",
    }

    for abbr, full in abbreviations.items():
        if abbr in query:
            query = query.replace(abbr, full)

    return query
```

### 2. Category Prediction + Filtering

```python
def search_with_category(query: str, limit: int = 20):
    """Search with category prediction and filtering"""
    # Simple rule-based category prediction
    category_rules = {
        "phone": "Electronics/Mobile",
        "dress": "Women's Fashion/Dresses",
        "running shoes": "Sports/Footwear",
    }

    predicted_category = None
    for keyword, category in category_rules.items():
        if keyword in query:
            predicted_category = category
            break

    # Build filter condition
    filter_expr = 'in_stock == true'
    if predicted_category:
        filter_expr += f' and category_l1 == "{predicted_category.split("/")[0]}"'

    return client.search(
        collection_name="products",
        data=[embed(query)],
        filter=filter_expr,
        limit=limit,
        output_fields=["title", "price", "sales", "rating"]
    )
```

### 3. Sales/Rating Weighted Ranking

```python
def search_with_ranking(query: str, limit: int = 20):
    """Comprehensive score ranking"""
    results = client.search(
        collection_name="products",
        data=[embed(query)],
        filter='in_stock == true',
        limit=limit * 3,  # Fetch more for re-ranking
        output_fields=["title", "price", "sales", "rating"]
    )

    # Composite score = similarity * 0.5 + normalized_sales * 0.3 + normalized_rating * 0.2
    max_sales = max(r["entity"]["sales"] for r in results[0]) or 1

    for r in results[0]:
        similarity_score = r["distance"]
        sales_score = r["entity"]["sales"] / max_sales
        rating_score = r["entity"]["rating"] / 5.0

        r["final_score"] = similarity_score * 0.5 + sales_score * 0.3 + rating_score * 0.2

    results[0].sort(key=lambda x: x["final_score"], reverse=True)
    return results[0][:limit]
```

## Example Queries

```python
# Basic search
results = search.search("wireless bluetooth earphones noise cancelling")

# With price filter
results = search.search(
    "running shoes",
    filter='price >= 200 and price <= 500 and in_stock == true'
)

# Brand search
results = search.search(
    "Nike basketball shoes",
    filter='brand == "Nike"'
)
```

## Performance Optimization

1. **Hot Product Caching**: Cache results for top 1000 popular products
2. **Partitioning Strategy**: Partition by level 1 category to reduce search scope
3. **Vector Quantization**: Use IVF_PQ to reduce memory usage
