# E-commerce Product Search

> Search products by natural language queries with category prediction and sales ranking.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Product Language

<ask_user>
What language are your product titles/descriptions in?

| Option | Model Recommendation |
|--------|---------------------|
| **Chinese** | Chinese-optimized models |
| **English** | English-optimized models |
| **Multilingual** | Multilingual models |
</ask_user>

### 2. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality, fast | Requires API key, costs money |
| **Local Model** | Free, offline | Model download required |
</ask_user>

### 3. Local Model Selection (if local)

<ask_user>
**For Chinese products:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-small-zh-v1.5` | 512 | 95MB | Fast, CPU-friendly |
| `BAAI/bge-base-zh-v1.5` | 768 | 400MB | Balanced |
| `BAAI/bge-large-zh-v1.5` | 1024 | 1.3GB | Best quality |

**For English products:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Higher quality |
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
Choose project management:

| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production projects |
| **pip** | Quick prototypes |
</ask_user>

---

## Dependencies

### OpenAI + uv
```bash
uv init ecommerce-search
cd ecommerce-search
uv add pymilvus openai
```

### Local Model + uv
```bash
uv init ecommerce-search
cd ecommerce-search
uv add pymilvus sentence-transformers
```

---

## End-to-End Implementation

### Step 1: Configure Embedding

```python
# === Choose ONE ===

# Option A: OpenAI API
from openai import OpenAI
client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

# Option B: Local Model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-zh-v1.5")  # or bge-base-en

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()

DIMENSION = 768
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("products.db")  # Milvus Lite

schema = client.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("category_l1", DataType.VARCHAR, max_length=64)
schema.add_field("category_l2", DataType.VARCHAR, max_length=64)
schema.add_field("brand", DataType.VARCHAR, max_length=128)
schema.add_field("price", DataType.FLOAT)
schema.add_field("sales", DataType.INT32)
schema.add_field("rating", DataType.FLOAT)
schema.add_field("in_stock", DataType.BOOL)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("products", schema=schema, index_params=index_params)
```

### Step 3: Index Products

```python
def index_products(products: list[dict], batch_size: int = 100):
    """Index products in batches.

    products: [{"title": "...", "category_l1": "...", "price": ..., ...}]
    """
    for i in range(0, len(products), batch_size):
        batch = products[i:i+batch_size]
        titles = [p["title"] for p in batch]
        embeddings = embed(titles)

        data = [
            {
                "title": p["title"],
                "embedding": emb,
                "category_l1": p.get("category_l1", ""),
                "category_l2": p.get("category_l2", ""),
                "brand": p.get("brand", ""),
                "price": p.get("price", 0.0),
                "sales": p.get("sales", 0),
                "rating": p.get("rating", 0.0),
                "in_stock": p.get("in_stock", True)
            }
            for p, emb in zip(batch, embeddings)
        ]

        client.insert(collection_name="products", data=data)

    print(f"Indexed {len(products)} products")
```

### Step 4: Search with Ranking

```python
def rewrite_query(query: str) -> str:
    """Handle colloquial queries and abbreviations."""
    abbreviations = {
        "phone case": "mobile phone protective case",
        "power bank": "portable charger",
        "earbuds": "wireless bluetooth earphones",
    }
    for abbr, full in abbreviations.items():
        if abbr in query.lower():
            query = query.lower().replace(abbr, full)
    return query

def search_products(query: str, top_k: int = 20,
                   category: str = None,
                   min_price: float = None,
                   max_price: float = None):
    """Search products with filters and ranking."""
    query = rewrite_query(query)
    query_embedding = embed([query])[0]

    # Build filter
    filters = ["in_stock == true"]
    if category:
        filters.append(f'category_l1 == "{category}"')
    if min_price is not None:
        filters.append(f'price >= {min_price}')
    if max_price is not None:
        filters.append(f'price <= {max_price}')

    filter_expr = " and ".join(filters)

    # Search with extra results for re-ranking
    results = client.search(
        collection_name="products",
        data=[query_embedding],
        filter=filter_expr,
        limit=top_k * 2,
        output_fields=["title", "price", "sales", "rating", "brand", "category_l1"]
    )

    if not results[0]:
        return []

    # Composite ranking: similarity * 0.5 + sales * 0.3 + rating * 0.2
    max_sales = max(r["entity"]["sales"] for r in results[0]) or 1

    for r in results[0]:
        similarity = r["distance"]
        sales_score = r["entity"]["sales"] / max_sales
        rating_score = r["entity"]["rating"] / 5.0

        r["final_score"] = similarity * 0.5 + sales_score * 0.3 + rating_score * 0.2

    results[0].sort(key=lambda x: x["final_score"], reverse=True)
    return results[0][:top_k]

def print_results(results):
    for i, r in enumerate(results, 1):
        e = r["entity"]
        print(f"\n#{i} {e['title'][:50]}...")
        print(f"    Price: ${e['price']:.2f} | Sales: {e['sales']} | Rating: {e['rating']:.1f}")
        print(f"    Brand: {e['brand']} | Category: {e['category_l1']}")
        print(f"    Score: {r['final_score']:.3f}")
```

### Step 5: Category Prediction Search

```python
def search_with_category_prediction(query: str, top_k: int = 20):
    """Search with automatic category prediction."""
    query_embedding = embed([query])[0]

    # First search without category filter
    initial = client.search(
        collection_name="products",
        data=[query_embedding],
        filter="in_stock == true",
        limit=50,
        output_fields=["category_l1"]
    )

    # Count category distribution
    from collections import Counter
    categories = [r["entity"]["category_l1"] for r in initial[0][:10]]
    most_common = Counter(categories).most_common(1)

    # If dominant category found, re-search with filter
    if most_common and most_common[0][1] >= 5:
        predicted = most_common[0][0]
        return search_products(query, top_k=top_k, category=predicted)

    return search_products(query, top_k=top_k)
```

---

## Run Example

```python
# Sample products
products = [
    {
        "title": "Wireless Bluetooth Earphones Noise Cancelling",
        "category_l1": "Electronics",
        "category_l2": "Audio",
        "brand": "Sony",
        "price": 199.99,
        "sales": 5000,
        "rating": 4.5,
        "in_stock": True
    },
    # ... more products
]

# Index
index_products(products)

# Basic search
results = search_products("wireless earphones")
print_results(results)

# With filters
results = search_products("running shoes", min_price=50, max_price=200)
print_results(results)

# With category prediction
results = search_with_category_prediction("Nike basketball shoes")
print_results(results)
```
