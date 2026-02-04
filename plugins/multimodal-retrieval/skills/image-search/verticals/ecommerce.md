# E-commerce Visual Search

> Search products by uploading an image to find similar items.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Search Mode

<ask_user>
How do you want to search products?

| Mode | Description |
|------|-------------|
| **Image-to-Image** | Upload photo, find similar products |
| **Text-to-Image** | Describe "red dress", find matching |
| **Both** (recommended) | Support both modes |
</ask_user>

### 2. Image Embedding Model

<ask_user>
Choose an image embedding model:

| Model | Size | Notes |
|-------|------|-------|
| `openai/clip-vit-base-patch32` (recommended) | 600MB | Good balance |
| `openai/clip-vit-large-patch14` | 1.7GB | Higher quality |
| `CN-CLIP` | 600MB | Chinese products |
</ask_user>

### 3. Data Scale

<ask_user>
How many product images do you have?

- Each image = 1 vector

| Image Count | Recommended Milvus |
|-------------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 4. Project Setup

<ask_user>
Choose project management:

| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production projects |
| **pip** | Quick prototypes |
</ask_user>

---

## Dependencies

```bash
uv init visual-search
cd visual-search
uv add pymilvus transformers torch Pillow
```

---

## End-to-End Implementation

### Step 1: Configure CLIP Model

```python
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

DIMENSION = 768  # 512 for base, 768 for large

def embed_images(images: list[Image.Image]) -> list[list[float]]:
    """Embed images using CLIP."""
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()

def embed_text(texts: list[str]) -> list[list[float]]:
    """Embed text using CLIP."""
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("products.db")  # Milvus Lite

schema = client.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("product_id", DataType.VARCHAR, max_length=64)
schema.add_field("image_url", DataType.VARCHAR, max_length=512)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("category", DataType.VARCHAR, max_length=64)
schema.add_field("brand", DataType.VARCHAR, max_length=128)
schema.add_field("price", DataType.FLOAT)
schema.add_field("in_stock", DataType.BOOL)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("product_images", schema=schema, index_params=index_params)
```

### Step 3: Preprocess & Index Images

```python
def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess user-uploaded image."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    max_size = 800
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    return image

def index_products(products: list[dict], batch_size: int = 32):
    """Index product images.

    products: [{"image_path": "...", "product_id": "...", "title": "...", ...}]
    """
    for i in range(0, len(products), batch_size):
        batch = products[i:i+batch_size]

        images = []
        for p in batch:
            img = Image.open(p["image_path"])
            images.append(preprocess_image(img))

        embeddings = embed_images(images)

        data = [
            {
                "embedding": emb,
                "product_id": p["product_id"],
                "image_url": p.get("image_url", ""),
                "title": p.get("title", ""),
                "category": p.get("category", ""),
                "brand": p.get("brand", ""),
                "price": p.get("price", 0.0),
                "in_stock": p.get("in_stock", True)
            }
            for p, emb in zip(batch, embeddings)
        ]

        client.insert(collection_name="product_images", data=data)
        print(f"Indexed {i + len(batch)}/{len(products)}")
```

### Step 4: Search

```python
def search_by_image(image_path: str, top_k: int = 20,
                    category: str = None, max_price: float = None):
    """Search by uploading an image."""
    img = Image.open(image_path)
    img = preprocess_image(img)
    query_embedding = embed_images([img])[0]

    filters = ["in_stock == true"]
    if category:
        filters.append(f'category == "{category}"')
    if max_price:
        filters.append(f'price <= {max_price}')

    results = client.search(
        collection_name="product_images",
        data=[query_embedding],
        filter=" and ".join(filters),
        limit=top_k,
        output_fields=["product_id", "title", "price", "image_url", "brand"]
    )
    return results[0]

def search_by_text(query: str, top_k: int = 20):
    """Search by text description."""
    query_embedding = embed_text([query])[0]

    results = client.search(
        collection_name="product_images",
        data=[query_embedding],
        filter="in_stock == true",
        limit=top_k,
        output_fields=["product_id", "title", "price", "image_url", "brand"]
    )
    return results[0]

def print_results(results):
    for i, r in enumerate(results, 1):
        e = r["entity"]
        print(f"\n#{i} {e['title'][:50]}...")
        print(f"    Price: ${e['price']:.2f} | Brand: {e['brand']}")
        print(f"    Score: {r['distance']:.3f}")
```

### Step 5: Category Prediction Search

```python
def search_with_category_prediction(image_path: str, top_k: int = 20):
    """Search with automatic category prediction."""
    img = Image.open(image_path)
    img = preprocess_image(img)
    query_embedding = embed_images([img])[0]

    # First search without filter
    initial = client.search(
        collection_name="product_images",
        data=[query_embedding],
        filter="in_stock == true",
        limit=50,
        output_fields=["category"]
    )

    # Predict category from top results
    from collections import Counter
    categories = [r["entity"]["category"] for r in initial[0][:10]]
    most_common = Counter(categories).most_common(1)

    if most_common and most_common[0][1] >= 5:
        predicted = most_common[0][0]
        return search_by_image(image_path, top_k=top_k, category=predicted)

    return initial[0][:top_k]
```

---

## Run Example

```python
# Index products
products = [
    {
        "image_path": "./images/dress_001.jpg",
        "product_id": "SKU001",
        "title": "Red Summer Dress",
        "category": "Dresses",
        "brand": "Zara",
        "price": 59.99,
        "in_stock": True
    },
    # ... more products
]
index_products(products)

# Search by image
results = search_by_image("./user_upload.jpg")
print_results(results)

# Search by text
results = search_by_text("black leather jacket")
print_results(results)

# With price filter
results = search_by_image("./photo.jpg", max_price=100)
print_results(results)
```
