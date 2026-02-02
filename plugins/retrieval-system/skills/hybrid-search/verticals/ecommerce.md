# E-commerce Hybrid Search

## Why Hybrid Search is Needed

In e-commerce search scenarios:
- **Keywords are important**: Brand names, model numbers, SKUs need exact matching
- **Semantics are also important**: Users may describe the same product differently
- **Single method is insufficient**: Pure keywords miss semantics, pure vectors miss exact terms

## Recommended Configuration

| Config | Recommended Value | Description |
|--------|------------------|-------------|
| Dense Vector Model | `BAAI/bge-large-en-v1.5` | Semantic understanding |
| Sparse Vector | BM25 / SPLADE | Keyword matching |
| Fusion Method | RRF | Stable and effective |
| RRF k Value | 60 | Default, adjustable |
| Vector Weight | 0.6 | Slightly favor semantics |
| Keyword Weight | 0.4 | Preserve exact matching |

## Typical Scenarios

### 1. Brand + Description Hybrid

```python
# User searches: "iPhone 15 Pro Max phone case transparent"
# - "iPhone 15 Pro Max" needs exact matching
# - "phone case transparent" can expand semantically to "protective case", "clear case"

query = "iPhone 15 Pro Max phone case transparent"

# Hybrid search
results = client.hybrid_search(
    collection_name="products",
    reqs=[
        AnnSearchRequest(data=[dense_embed(query)], anns_field="dense_vector", limit=50),
        AnnSearchRequest(data=[sparse_embed(query)], anns_field="sparse_vector", limit=50)
    ],
    ranker=RRFRanker(k=60),
    filter='in_stock == true',
    limit=20,
    output_fields=["title", "price", "brand"]
)
```

### 2. Model Number Query

```python
# User searches: "A2894" (a product model number)
# Pure semantic search may not find it, needs keywords

query = "A2894"

# Keyword weight higher in this case
results = client.hybrid_search(
    collection_name="products",
    reqs=[
        AnnSearchRequest(data=[dense_embed(query)], anns_field="dense_vector",
                        limit=50, params={"weight": 0.3}),
        AnnSearchRequest(data=[sparse_embed(query)], anns_field="sparse_vector",
                        limit=50, params={"weight": 0.7})
    ],
    ranker=WeightedRanker(0.3, 0.7),
    limit=20
)
```

### 3. Long-tail Query

```python
# User searches: "gift for programmer boyfriend"
# Pure keywords can't find it, needs semantic understanding

query = "gift suitable for programmer boyfriend"

# Semantic weight higher
results = client.hybrid_search(
    collection_name="products",
    reqs=[
        AnnSearchRequest(data=[dense_embed(query)], anns_field="dense_vector",
                        limit=50, params={"weight": 0.8}),
        AnnSearchRequest(data=[sparse_embed(query)], anns_field="sparse_vector",
                        limit=50, params={"weight": 0.2})
    ],
    ranker=WeightedRanker(0.8, 0.2),
    limit=20
)
```

## Dynamic Weight Strategy

```python
def get_search_weights(query: str) -> tuple:
    """Dynamically adjust weights based on query characteristics"""
    # Detect if contains model/SKU (letter+number combination)
    import re
    has_sku = bool(re.search(r'[A-Za-z]+\d+|\d+[A-Za-z]+', query))

    # Detect if contains brand
    brands = ["Apple", "Samsung", "Nike", "Adidas", "Sony", "LG"]
    has_brand = any(brand.lower() in query.lower() for brand in brands)

    # Query length
    is_long_query = len(query) > 20

    # Dynamic weights
    if has_sku:
        return (0.2, 0.8)  # SKU query, keyword priority
    elif has_brand and not is_long_query:
        return (0.4, 0.6)  # Brand + short query, balanced toward keywords
    elif is_long_query:
        return (0.8, 0.2)  # Long query, semantic priority
    else:
        return (0.6, 0.4)  # Default, slightly favor semantics
```

## Performance Optimization

```python
# 1. Partitioning strategy: partition by L1 category
client.create_partition(collection_name="products", partition_name="electronics")
client.create_partition(collection_name="products", partition_name="clothing")

# 2. Specify partition when searching
results = client.hybrid_search(
    collection_name="products",
    reqs=[...],
    partition_names=["electronics"],  # Only search electronics partition
    limit=20
)

# 3. Pre-filter to reduce search scope
results = client.hybrid_search(
    collection_name="products",
    reqs=[...],
    filter='category_l1 == "electronics" and price >= 100 and price <= 1000',
    limit=20
)
```

## Examples

```python
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

client = MilvusClient("./milvus.db")

def ecommerce_hybrid_search(query: str, category: str = None,
                            price_range: tuple = None, limit: int = 20):
    """E-commerce hybrid search"""
    # Dynamic weights
    dense_weight, sparse_weight = get_search_weights(query)

    # Build filter conditions
    filters = ['in_stock == true']
    if category:
        filters.append(f'category_l1 == "{category}"')
    if price_range:
        filters.append(f'price >= {price_range[0]} and price <= {price_range[1]}')

    filter_expr = ' and '.join(filters)

    # Hybrid search
    results = client.hybrid_search(
        collection_name="products",
        reqs=[
            AnnSearchRequest(
                data=[dense_embed(query)],
                anns_field="dense_vector",
                limit=50,
                param={"metric_type": "COSINE"}
            ),
            AnnSearchRequest(
                data=[sparse_embed(query)],
                anns_field="sparse_vector",
                limit=50,
                param={"metric_type": "IP"}
            )
        ],
        ranker=WeightedRanker(dense_weight, sparse_weight),
        filter=filter_expr,
        limit=limit,
        output_fields=["title", "price", "brand", "image_url"]
    )

    return results

# Usage
results = ecommerce_hybrid_search("iPhone 15 phone case", category="accessories")
```
