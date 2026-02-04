# E-commerce Similar Product Recommendations

> Find similar products for "You may also like" and alternatives.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Product Catalog Language

<ask_user>
What language are your products in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** | English models |
| **Chinese** | Chinese models |
| **Multilingual** | Multilingual models |
</ask_user>

### 2. Feature Type

<ask_user>
What features to use for similarity?

| Mode | Description |
|------|-------------|
| **Text only** | Title + description |
| **Image only** | Product images |
| **Multimodal** (recommended) | Text + image combined |
</ask_user>

### 3. Text Embedding

<ask_user>
Choose text embedding:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download |

Local options:
- `BAAI/bge-large-en-v1.5` (1024d, English)
- `BAAI/bge-large-zh-v1.5` (1024d, Chinese)
</ask_user>

### 4. Data Scale

<ask_user>
How many products do you have?

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
| **uv + pyproject.toml** (recommended) | Production |
| **pip** | Quick prototype |
</ask_user>

---

## Dependencies

```bash
uv init similar-products
cd similar-products
uv add pymilvus sentence-transformers
# For image features:
uv add transformers torch Pillow
```

---

## End-to-End Implementation

### Step 1: Configure Embedding

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
DIMENSION = 1024

def create_product_embedding(product: dict) -> list:
    """Create embedding from product text."""
    text = f"{product['title']} {product.get('brand', '')} {product.get('category', '')}"

    if product.get("attributes"):
        attrs = " ".join([f"{k}:{v}" for k, v in product["attributes"].items()])
        text += f" {attrs}"

    return model.encode(text, normalize_embeddings=True).tolist()
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("products.db")

schema = milvus.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("product_id", DataType.VARCHAR, max_length=64)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("category_l1", DataType.VARCHAR, max_length=64)
schema.add_field("category_l2", DataType.VARCHAR, max_length=64)
schema.add_field("brand", DataType.VARCHAR, max_length=128)
schema.add_field("price", DataType.FLOAT)
schema.add_field("price_tier", DataType.VARCHAR, max_length=16)  # low/mid/high
schema.add_field("in_stock", DataType.BOOL)
schema.add_field("shop_id", DataType.VARCHAR, max_length=64)
schema.add_field("image_url", DataType.VARCHAR, max_length=512)

index_params = milvus.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

milvus.create_collection("products", schema=schema, index_params=index_params)
```

### Step 3: Index Products

```python
def index_products(products: list[dict]):
    """Index products with embeddings."""
    data = []
    for p in products:
        embedding = create_product_embedding(p)
        data.append({
            "embedding": embedding,
            "product_id": p["product_id"],
            "title": p["title"],
            "category_l1": p.get("category_l1", ""),
            "category_l2": p.get("category_l2", ""),
            "brand": p.get("brand", ""),
            "price": p.get("price", 0.0),
            "price_tier": p.get("price_tier", "mid"),
            "in_stock": p.get("in_stock", True),
            "shop_id": p.get("shop_id", ""),
            "image_url": p.get("image_url", "")
        })

    milvus.insert(collection_name="products", data=data)
```

### Step 4: Similar Products

```python
def get_similar_products(product_id: str, top_k: int = 10,
                         same_category: bool = True,
                         same_price_tier: bool = True,
                         exclude_same_shop: bool = True):
    """Get similar products."""
    # Get current product
    product = milvus.query(
        collection_name="products",
        filter=f'product_id == "{product_id}"',
        output_fields=["embedding", "category_l1", "price_tier", "shop_id"],
        limit=1
    )

    if not product:
        return []

    p = product[0]

    # Build filters
    filters = [
        f'product_id != "{product_id}"',
        'in_stock == true'
    ]

    if same_category:
        filters.append(f'category_l1 == "{p["category_l1"]}"')

    if same_price_tier:
        filters.append(f'price_tier == "{p["price_tier"]}"')

    if exclude_same_shop:
        filters.append(f'shop_id != "{p["shop_id"]}"')

    filter_expr = ' and '.join(filters)

    results = milvus.search(
        collection_name="products",
        data=[p["embedding"]],
        filter=filter_expr,
        limit=top_k,
        output_fields=["product_id", "title", "price", "brand", "image_url"]
    )

    return [{
        "product_id": r["entity"]["product_id"],
        "title": r["entity"]["title"],
        "price": r["entity"]["price"],
        "brand": r["entity"]["brand"],
        "image_url": r["entity"]["image_url"],
        "similarity": r["distance"]
    } for r in results[0]]

def get_alternatives(product_id: str, top_k: int = 5):
    """Get alternatives (for out-of-stock)."""
    product = milvus.query(
        collection_name="products",
        filter=f'product_id == "{product_id}"',
        output_fields=["embedding", "category_l2", "price"],
        limit=1
    )

    if not product:
        return []

    p = product[0]
    price = p["price"]

    # Same L2 category, similar price (±20%)
    filter_expr = f'''
        product_id != "{product_id}"
        and in_stock == true
        and category_l2 == "{p["category_l2"]}"
        and price >= {price * 0.8}
        and price <= {price * 1.2}
    '''

    results = milvus.search(
        collection_name="products",
        data=[p["embedding"]],
        filter=filter_expr,
        limit=top_k,
        output_fields=["product_id", "title", "price"]
    )

    return results[0]
```

---

## Run Example

```python
# Index products
products = [
    {
        "product_id": "iphone15_001",
        "title": "Apple iPhone 15 Pro Max 256GB",
        "category_l1": "Electronics",
        "category_l2": "Smartphones",
        "brand": "Apple",
        "price": 1199,
        "price_tier": "high",
        "in_stock": True,
        "shop_id": "shop_001"
    },
    {
        "product_id": "samsung24_001",
        "title": "Samsung Galaxy S24 Ultra 256GB",
        "category_l1": "Electronics",
        "category_l2": "Smartphones",
        "brand": "Samsung",
        "price": 1099,
        "price_tier": "high",
        "in_stock": True,
        "shop_id": "shop_002"
    },
]
index_products(products)

# Get similar products
similar = get_similar_products(
    product_id="iphone15_001",
    same_category=True,
    exclude_same_shop=True
)

print("Similar products:")
for p in similar:
    print(f"  {p['title']} - ${p['price']} (similarity: {p['similarity']:.2f})")

# Get alternatives (for out-of-stock)
alternatives = get_alternatives("iphone15_001")
```
