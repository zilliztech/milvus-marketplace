# E-commerce Hybrid Search

> Combine semantic search with keyword matching for better product discovery.

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

### 2. Dense Embedding Method

<ask_user>
Choose your dense embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download required |
</ask_user>

### 3. Local Model Selection (if local)

<ask_user>
**For English:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Good balance |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3GB | Best quality |

**For Chinese:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-base-zh-v1.5` | 768 | 400MB | Good balance |
| `BAAI/bge-large-zh-v1.5` | 1024 | 1.3GB | Best quality |
</ask_user>

### 4. Sparse Embedding

<ask_user>
Choose sparse embedding for keyword matching:

| Method | Notes |
|--------|-------|
| **BM25** | Classic, works well |
| **SPLADE** | Neural sparse, better quality |
</ask_user>

### 5. Data Scale

<ask_user>
How many products do you have?

| Product Count | Recommended Milvus |
|---------------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 6. Project Setup

<ask_user>
| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production projects |
| **pip** | Quick prototypes |
</ask_user>

---

## Dependencies

```bash
uv init ecommerce-hybrid
cd ecommerce-hybrid
uv add pymilvus sentence-transformers
# For BM25:
uv add pymilvus[model]
```

---

## End-to-End Implementation

### Step 1: Configure Models

```python
from pymilvus import model
from sentence_transformers import SentenceTransformer

# Dense embedding
dense_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
DENSE_DIM = 1024

# Sparse embedding (BM25)
bm25_ef = model.sparse.BM25EmbeddingFunction(
    analyzer="english",
    k1=1.5,  # Term frequency saturation
    b=0.75   # Length normalization
)

def dense_embed(texts: list[str]) -> list[list[float]]:
    return dense_model.encode(texts, normalize_embeddings=True).tolist()

def sparse_embed(texts: list[str]):
    return bm25_ef.encode_documents(texts) if isinstance(texts, list) else bm25_ef.encode_queries(texts)
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("products.db")

schema = client.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DENSE_DIM)
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
schema.add_field("product_id", DataType.VARCHAR, max_length=64)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("category", DataType.VARCHAR, max_length=64)
schema.add_field("brand", DataType.VARCHAR, max_length=128)
schema.add_field("price", DataType.FLOAT)
schema.add_field("in_stock", DataType.BOOL)

index_params = client.prepare_index_params()
index_params.add_index("dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

client.create_collection("products", schema=schema, index_params=index_params)
```

### Step 3: Index Products

```python
def index_products(products: list[dict]):
    """Index products with dense and sparse vectors."""
    titles = [p["title"] for p in products]

    # Fit BM25 on corpus (do this once with full corpus)
    bm25_ef.fit(titles)

    dense_vectors = dense_embed(titles)
    sparse_vectors = bm25_ef.encode_documents(titles)

    data = [
        {
            "dense_vector": dense,
            "sparse_vector": sparse,
            "product_id": p["product_id"],
            "title": p["title"],
            "category": p.get("category", ""),
            "brand": p.get("brand", ""),
            "price": p.get("price", 0.0),
            "in_stock": p.get("in_stock", True)
        }
        for p, dense, sparse in zip(products, dense_vectors, sparse_vectors)
    ]

    client.insert(collection_name="products", data=data)
```

### Step 4: Hybrid Search

```python
from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

def get_search_weights(query: str) -> tuple:
    """Dynamically adjust weights based on query type."""
    import re

    # SKU/model number pattern
    has_sku = bool(re.search(r'[A-Za-z]+\d+|\d+[A-Za-z]+', query))

    # Known brand names
    brands = ["Apple", "Samsung", "Nike", "Sony", "iPhone", "Galaxy"]
    has_brand = any(b.lower() in query.lower() for b in brands)

    is_long_query = len(query) > 20

    if has_sku:
        return (0.2, 0.8)  # Keyword priority for SKU
    elif has_brand and not is_long_query:
        return (0.4, 0.6)  # Balanced with keyword lean
    elif is_long_query:
        return (0.8, 0.2)  # Semantic priority for descriptions
    else:
        return (0.6, 0.4)  # Default: favor semantics

def hybrid_search(query: str, category: str = None, top_k: int = 20):
    """Hybrid search combining dense and sparse vectors."""
    dense_weight, sparse_weight = get_search_weights(query)

    dense_vec = dense_embed([query])[0]
    sparse_vec = bm25_ef.encode_queries([query])[0]

    # Build filter
    filter_expr = 'in_stock == true'
    if category:
        filter_expr += f' and category == "{category}"'

    results = client.hybrid_search(
        collection_name="products",
        reqs=[
            AnnSearchRequest(
                data=[dense_vec],
                anns_field="dense_vector",
                param={"metric_type": "COSINE"},
                limit=50
            ),
            AnnSearchRequest(
                data=[sparse_vec],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=50
            )
        ],
        ranker=WeightedRanker(dense_weight, sparse_weight),
        filter=filter_expr,
        limit=top_k,
        output_fields=["product_id", "title", "price", "brand"]
    )

    return [{
        "product_id": r["entity"]["product_id"],
        "title": r["entity"]["title"],
        "price": r["entity"]["price"],
        "brand": r["entity"]["brand"],
        "score": r["distance"]
    } for r in results]
```

---

## Run Example

```python
# Index products
products = [
    {"product_id": "A2894", "title": "Apple iPhone 15 Pro Max 256GB", "category": "Phones", "brand": "Apple", "price": 1199},
    {"product_id": "B7721", "title": "Samsung Galaxy S24 Ultra", "category": "Phones", "brand": "Samsung", "price": 1099},
]
index_products(products)

# SKU search (keyword priority)
results = hybrid_search("A2894")

# Descriptive search (semantic priority)
results = hybrid_search("gift for programmer boyfriend")

# Brand + feature search (balanced)
results = hybrid_search("iPhone phone case transparent", category="Accessories")

for r in results:
    print(f"{r['title']} - ${r['price']}")
```
