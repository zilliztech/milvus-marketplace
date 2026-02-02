# E-commerce Filtered Search

## Use Case

E-commerce search needs to satisfy both:
- Semantic relevance (find relevant products)
- Hard constraints (price, stock, brand, etc.)

## Common Filter Conditions

| Field | Type | Filter Method |
|-------|------|---------------|
| Price | FLOAT | Range filter |
| Brand | VARCHAR | Exact match |
| Category | VARCHAR | Exact/hierarchical match |
| Stock | BOOL | In stock |
| Rating | FLOAT | Minimum rating |
| Sales | INT | Minimum sales |
| Shop | VARCHAR | Shop filter |
| Shipping | VARCHAR | Region filter |

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

# Filter fields (need indexing)
schema.add_field("category_l1", DataType.VARCHAR, max_length=64)
schema.add_field("category_l2", DataType.VARCHAR, max_length=64)
schema.add_field("category_l3", DataType.VARCHAR, max_length=64)
schema.add_field("brand", DataType.VARCHAR, max_length=128)
schema.add_field("price", DataType.FLOAT)
schema.add_field("original_price", DataType.FLOAT)
schema.add_field("discount", DataType.FLOAT)                        # Discount rate
schema.add_field("rating", DataType.FLOAT)
schema.add_field("review_count", DataType.INT32)
schema.add_field("sales_30d", DataType.INT32)                       # 30-day sales
schema.add_field("in_stock", DataType.BOOL)
schema.add_field("shop_id", DataType.VARCHAR, max_length=64)
schema.add_field("shop_rating", DataType.FLOAT)
schema.add_field("shipping_from", DataType.VARCHAR, max_length=64)  # Shipping location
schema.add_field("free_shipping", DataType.BOOL)
schema.add_field("has_coupon", DataType.BOOL)

# Index configuration
index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
# Scalar indexes
index_params.add_index("category_l1", index_type="TRIE")
index_params.add_index("category_l2", index_type="TRIE")
index_params.add_index("brand", index_type="TRIE")
index_params.add_index("price", index_type="STL_SORT")
index_params.add_index("rating", index_type="STL_SORT")
index_params.add_index("sales_30d", index_type="STL_SORT")
```

## Implementation

```python
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

class EcommerceFilteredSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    def search(self, query: str,
               # Filter conditions
               category: str = None,
               brands: list = None,
               price_min: float = None,
               price_max: float = None,
               min_rating: float = None,
               min_sales: int = None,
               in_stock_only: bool = True,
               free_shipping: bool = False,
               shipping_from: str = None,
               # Sorting
               sort_by: str = None,  # price_asc, price_desc, sales, rating
               limit: int = 20) -> list:
        """Search with filter conditions"""

        # 1. Build filter expression
        filter_parts = []

        if in_stock_only:
            filter_parts.append('in_stock == true')

        if category:
            # Support multi-level categories
            if '/' in category:
                cats = category.split('/')
                if len(cats) >= 1:
                    filter_parts.append(f'category_l1 == "{cats[0]}"')
                if len(cats) >= 2:
                    filter_parts.append(f'category_l2 == "{cats[1]}"')
                if len(cats) >= 3:
                    filter_parts.append(f'category_l3 == "{cats[2]}"')
            else:
                filter_parts.append(f'category_l1 == "{category}"')

        if brands:
            brand_expr = ' or '.join([f'brand == "{b}"' for b in brands])
            filter_parts.append(f'({brand_expr})')

        if price_min is not None:
            filter_parts.append(f'price >= {price_min}')

        if price_max is not None:
            filter_parts.append(f'price <= {price_max}')

        if min_rating is not None:
            filter_parts.append(f'rating >= {min_rating}')

        if min_sales is not None:
            filter_parts.append(f'sales_30d >= {min_sales}')

        if free_shipping:
            filter_parts.append('free_shipping == true')

        if shipping_from:
            filter_parts.append(f'shipping_from == "{shipping_from}"')

        filter_expr = ' and '.join(filter_parts) if filter_parts else ""

        # 2. Vector search
        embedding = self.model.encode(query).tolist()

        results = self.client.search(
            collection_name="products",
            data=[embedding],
            filter=filter_expr,
            limit=limit * 2 if sort_by else limit,  # Get more if sorting needed
            output_fields=["title", "price", "rating", "sales_30d", "brand",
                          "category_l1", "image_url", "shop_id"]
        )

        # 3. Post-processing sort (if specified)
        items = results[0]

        if sort_by:
            if sort_by == 'price_asc':
                items.sort(key=lambda x: x["entity"]["price"])
            elif sort_by == 'price_desc':
                items.sort(key=lambda x: x["entity"]["price"], reverse=True)
            elif sort_by == 'sales':
                items.sort(key=lambda x: x["entity"]["sales_30d"], reverse=True)
            elif sort_by == 'rating':
                items.sort(key=lambda x: x["entity"]["rating"], reverse=True)

            items = items[:limit]

        return [{
            "id": item["id"],
            "title": item["entity"]["title"],
            "price": item["entity"]["price"],
            "rating": item["entity"]["rating"],
            "sales": item["entity"]["sales_30d"],
            "brand": item["entity"]["brand"],
            "category": item["entity"]["category_l1"],
            "image_url": item["entity"]["image_url"],
            "relevance_score": item["distance"]
        } for item in items]

    def get_filters(self, query: str) -> dict:
        """Get available filter options (for frontend display)"""
        # Search without filters first, get distribution stats
        embedding = self.model.encode(query).tolist()

        results = self.client.search(
            collection_name="products",
            data=[embedding],
            limit=500,  # Get more for statistics
            output_fields=["category_l1", "brand", "price", "shipping_from"]
        )

        # Statistics
        from collections import Counter

        categories = Counter(r["entity"]["category_l1"] for r in results[0])
        brands = Counter(r["entity"]["brand"] for r in results[0])
        locations = Counter(r["entity"]["shipping_from"] for r in results[0])
        prices = [r["entity"]["price"] for r in results[0]]

        return {
            "categories": [{"name": k, "count": v} for k, v in categories.most_common(10)],
            "brands": [{"name": k, "count": v} for k, v in brands.most_common(20)],
            "shipping_locations": [{"name": k, "count": v} for k, v in locations.most_common(10)],
            "price_range": {
                "min": min(prices) if prices else 0,
                "max": max(prices) if prices else 0,
                "avg": sum(prices) / len(prices) if prices else 0
            }
        }
```

## Examples

```python
search = EcommerceFilteredSearch()

# Basic search
results = search.search("wireless bluetooth headphones")

# With price range
results = search.search(
    "wireless bluetooth headphones",
    price_min=50,
    price_max=200
)

# Specify brands
results = search.search(
    "wireless bluetooth headphones",
    brands=["Sony", "Apple", "Samsung"],
    min_rating=4.5
)

# Complete filtering
results = search.search(
    "wireless bluetooth headphones",
    category="electronics/headphones",
    brands=["Sony", "Apple"],
    price_min=100,
    price_max=500,
    min_rating=4.0,
    in_stock_only=True,
    free_shipping=True,
    sort_by="sales"
)

# Get filter options
filters = search.get_filters("wireless bluetooth headphones")
print("Available brands:", [b["name"] for b in filters["brands"][:5]])
print("Price range:", filters["price_range"])
```

## Performance Optimization

### 1. Partitioning Strategy

```python
# Partition by top-level category
client.create_partition(collection_name="products", partition_name="electronics")
client.create_partition(collection_name="products", partition_name="clothing")

# Search specific partition
results = client.search(
    collection_name="products",
    partition_names=["electronics"],
    data=[embedding],
    filter='price >= 100 and price <= 500',
    limit=20
)
```

### 2. Pre-compute Popular Combinations

```python
# Cache results for popular search + filter combinations
cache_key = f"{query}:{category}:{price_range}"
if cache_key in cache:
    return cache[cache_key]
```

### 3. Filter Condition Order Optimization

```python
# Put highly selective conditions first
# Example: brand (high selectivity) > price range > stock (low selectivity)
filter_expr = 'brand == "Apple" and price >= 100 and in_stock == true'
```
