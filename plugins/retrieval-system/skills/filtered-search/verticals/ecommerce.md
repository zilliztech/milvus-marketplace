# E-commerce Filtered Search

> Search products with semantic relevance and hard constraints (price, brand, stock).

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Product Catalog Language

<ask_user>
What language are your product titles and descriptions in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** | English models |
| **Chinese** | Chinese models |
| **Multilingual** | Multilingual models |
</ask_user>

### 2. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download required |
</ask_user>

### 3. Local Model Selection (if local)

<ask_user>
**For English products:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast, good for titles |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Better quality |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3GB | Best quality |

**For Chinese products:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-base-zh-v1.5` | 768 | 400MB | Good balance |
| `BAAI/bge-large-zh-v1.5` | 1024 | 1.3GB | Best quality |

**For Multilingual:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-m3` | 1024 | 2.2GB | Best multilingual |
</ask_user>

### 4. Data Scale

<ask_user>
How many products do you have?

- Each product = 1 vector

| Product Count | Recommended Milvus |
|---------------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 5. Project Setup

<ask_user>
| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production projects |
| **pip** | Quick prototypes |
</ask_user>

---

## Dependencies

```bash
uv init ecommerce-search
cd ecommerce-search
uv add pymilvus openai
# Or for local embedding:
uv add pymilvus sentence-transformers
```

---

## End-to-End Implementation

### Step 1: Configure Embedding

```python
# === Choose ONE embedding approach ===

# Option A: OpenAI API
from openai import OpenAI
client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

# Option B: Local Model
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("BAAI/bge-large-en-v1.5")
# def embed(texts): return model.encode(texts, normalize_embeddings=True).tolist()
# DIMENSION = 1024
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("products.db")

schema = milvus.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("product_id", DataType.VARCHAR, max_length=64)
schema.add_field("title", DataType.VARCHAR, max_length=512)

# Filter fields
schema.add_field("category_l1", DataType.VARCHAR, max_length=64)
schema.add_field("category_l2", DataType.VARCHAR, max_length=64)
schema.add_field("brand", DataType.VARCHAR, max_length=128)
schema.add_field("price", DataType.FLOAT)
schema.add_field("rating", DataType.FLOAT)
schema.add_field("sales_30d", DataType.INT32)
schema.add_field("in_stock", DataType.BOOL)
schema.add_field("free_shipping", DataType.BOOL)

# Index configuration
index_params = milvus.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index("category_l1", index_type="TRIE")
index_params.add_index("brand", index_type="TRIE")
index_params.add_index("price", index_type="STL_SORT")

milvus.create_collection("products", schema=schema, index_params=index_params)
```

### Step 3: Index Products

```python
def index_products(products: list[dict]):
    """Index products with embeddings.

    products: [{"product_id": "...", "title": "...", "category_l1": "...", ...}]
    """
    titles = [p["title"] for p in products]
    embeddings = embed(titles)

    data = [
        {
            "embedding": emb,
            "product_id": p["product_id"],
            "title": p["title"],
            "category_l1": p.get("category_l1", ""),
            "category_l2": p.get("category_l2", ""),
            "brand": p.get("brand", ""),
            "price": p.get("price", 0.0),
            "rating": p.get("rating", 0.0),
            "sales_30d": p.get("sales_30d", 0),
            "in_stock": p.get("in_stock", True),
            "free_shipping": p.get("free_shipping", False)
        }
        for p, emb in zip(products, embeddings)
    ]

    milvus.insert(collection_name="products", data=data)
    print(f"Indexed {len(products)} products")
```

### Step 4: Filtered Search

```python
def search_products(query: str,
                    category: str = None,
                    brands: list = None,
                    price_min: float = None,
                    price_max: float = None,
                    min_rating: float = None,
                    in_stock_only: bool = True,
                    free_shipping: bool = False,
                    sort_by: str = None,
                    top_k: int = 20):
    """Search products with filters."""
    query_embedding = embed([query])[0]

    # Build filter expression
    filters = []

    if in_stock_only:
        filters.append('in_stock == true')

    if category:
        filters.append(f'category_l1 == "{category}"')

    if brands:
        brand_expr = ' or '.join([f'brand == "{b}"' for b in brands])
        filters.append(f'({brand_expr})')

    if price_min is not None:
        filters.append(f'price >= {price_min}')

    if price_max is not None:
        filters.append(f'price <= {price_max}')

    if min_rating is not None:
        filters.append(f'rating >= {min_rating}')

    if free_shipping:
        filters.append('free_shipping == true')

    filter_expr = ' and '.join(filters) if filters else None

    results = milvus.search(
        collection_name="products",
        data=[query_embedding],
        filter=filter_expr,
        limit=top_k * 2 if sort_by else top_k,
        output_fields=["product_id", "title", "price", "rating", "sales_30d", "brand"]
    )

    items = results[0]

    # Post-sort if specified
    if sort_by:
        if sort_by == 'price_asc':
            items.sort(key=lambda x: x["entity"]["price"])
        elif sort_by == 'price_desc':
            items.sort(key=lambda x: x["entity"]["price"], reverse=True)
        elif sort_by == 'sales':
            items.sort(key=lambda x: x["entity"]["sales_30d"], reverse=True)
        elif sort_by == 'rating':
            items.sort(key=lambda x: x["entity"]["rating"], reverse=True)
        items = items[:top_k]

    return [{
        "product_id": item["entity"]["product_id"],
        "title": item["entity"]["title"],
        "price": item["entity"]["price"],
        "rating": item["entity"]["rating"],
        "brand": item["entity"]["brand"],
        "relevance": item["distance"]
    } for item in items]
```

---

## Run Example

```python
# Index products
products = [
    {
        "product_id": "SKU001",
        "title": "Sony WH-1000XM5 Wireless Headphones",
        "category_l1": "Electronics",
        "category_l2": "Headphones",
        "brand": "Sony",
        "price": 349.99,
        "rating": 4.8,
        "sales_30d": 1500,
        "in_stock": True,
        "free_shipping": True
    },
    # ... more products
]
index_products(products)

# Search with filters
results = search_products(
    "wireless bluetooth headphones",
    category="Electronics",
    brands=["Sony", "Apple", "Samsung"],
    price_min=100,
    price_max=500,
    min_rating=4.0,
    in_stock_only=True,
    free_shipping=True,
    sort_by="sales"
)

for r in results:
    print(f"{r['title']} - ${r['price']} ({r['brand']})")
```
