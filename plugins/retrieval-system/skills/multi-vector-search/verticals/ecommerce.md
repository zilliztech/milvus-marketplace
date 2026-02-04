# E-commerce Multi-Vector Search

> Search products using multiple vectors for title, description, and image.

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

### 2. Search Modes Needed

<ask_user>
Which search modes do you need?

| Mode | Description |
|------|-------------|
| **Text-to-Product** | Search by text query |
| **Image-to-Product** | Search by uploading image |
| **Both** (recommended) | Support both modes |
</ask_user>

### 3. Text Embedding

<ask_user>
Choose text embedding:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download |

Local model options:
- `BAAI/bge-large-en-v1.5` (English, 1024d)
- `BAAI/bge-large-zh-v1.5` (Chinese, 1024d)
</ask_user>

### 4. Image Embedding

<ask_user>
Choose image embedding model:

| Model | Size | Notes |
|-------|------|-------|
| `openai/clip-vit-base-patch32` | 600MB | Good balance |
| `openai/clip-vit-large-patch14` | 1.7GB | Higher quality |
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
| **uv + pyproject.toml** (recommended) | Production |
| **pip** | Quick prototype |
</ask_user>

---

## Dependencies

```bash
uv init multi-vector-search
cd multi-vector-search
uv add pymilvus sentence-transformers transformers torch Pillow
```

---

## End-to-End Implementation

### Step 1: Configure Models

```python
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Text embedding
text_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
TEXT_DIM = 1024

# Image embedding (CLIP)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device).eval()
IMAGE_DIM = 512

def embed_text(texts: list[str]) -> list[list[float]]:
    return text_model.encode(texts, normalize_embeddings=True).tolist()

def embed_images(images: list[Image.Image]) -> list[list[float]]:
    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()

def embed_text_for_image(texts: list[str]) -> list[list[float]]:
    """Embed text using CLIP for image search."""
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("products.db")

schema = client.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("title_vector", DataType.FLOAT_VECTOR, dim=TEXT_DIM)
schema.add_field("desc_vector", DataType.FLOAT_VECTOR, dim=TEXT_DIM)
schema.add_field("image_vector", DataType.FLOAT_VECTOR, dim=IMAGE_DIM)
schema.add_field("product_id", DataType.VARCHAR, max_length=64)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("description", DataType.VARCHAR, max_length=65535)
schema.add_field("image_url", DataType.VARCHAR, max_length=512)
schema.add_field("category", DataType.VARCHAR, max_length=64)
schema.add_field("brand", DataType.VARCHAR, max_length=128)
schema.add_field("price", DataType.FLOAT)
schema.add_field("in_stock", DataType.BOOL)

index_params = client.prepare_index_params()
index_params.add_index("title_vector", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index("desc_vector", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index("image_vector", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("products", schema=schema, index_params=index_params)
```

### Step 3: Index Products

```python
def index_product(product: dict):
    """Index a single product with multiple vectors."""
    # Text vectors
    title_vec = embed_text([product["title"]])[0]
    desc_vec = embed_text([product["description"]])[0]

    # Image vector
    image = Image.open(product["image_path"]).convert("RGB")
    image_vec = embed_images([image])[0]

    client.insert(
        collection_name="products",
        data=[{
            "title_vector": title_vec,
            "desc_vector": desc_vec,
            "image_vector": image_vec,
            "product_id": product["product_id"],
            "title": product["title"],
            "description": product["description"],
            "image_url": product.get("image_url", ""),
            "category": product.get("category", ""),
            "brand": product.get("brand", ""),
            "price": product.get("price", 0.0),
            "in_stock": product.get("in_stock", True)
        }]
    )
```

### Step 4: Multi-Vector Search

```python
from pymilvus import AnnSearchRequest, WeightedRanker

def search_products(query: str, mode: str = "balanced", top_k: int = 20):
    """Search products using multiple vectors.

    mode: balanced, title, description, visual
    """
    text_vec = embed_text([query])[0]
    clip_text_vec = embed_text_for_image([query])[0]

    # Set weights based on mode
    weights = {
        "balanced": [0.4, 0.4, 0.2],
        "title": [0.7, 0.2, 0.1],
        "description": [0.2, 0.7, 0.1],
        "visual": [0.2, 0.2, 0.6]
    }[mode]

    results = client.hybrid_search(
        collection_name="products",
        reqs=[
            AnnSearchRequest(data=[text_vec], anns_field="title_vector",
                           param={"metric_type": "COSINE"}, limit=50),
            AnnSearchRequest(data=[text_vec], anns_field="desc_vector",
                           param={"metric_type": "COSINE"}, limit=50),
            AnnSearchRequest(data=[clip_text_vec], anns_field="image_vector",
                           param={"metric_type": "COSINE"}, limit=50)
        ],
        ranker=WeightedRanker(*weights),
        filter='in_stock == true',
        limit=top_k,
        output_fields=["product_id", "title", "price", "brand", "image_url"]
    )

    return [{
        "product_id": r["entity"]["product_id"],
        "title": r["entity"]["title"],
        "price": r["entity"]["price"],
        "brand": r["entity"]["brand"],
        "image_url": r["entity"]["image_url"],
        "score": r["distance"]
    } for r in results]

def search_by_image(image_path: str, top_k: int = 20):
    """Search products by image."""
    image = Image.open(image_path).convert("RGB")
    image_vec = embed_images([image])[0]

    results = client.search(
        collection_name="products",
        data=[image_vec],
        anns_field="image_vector",
        filter='in_stock == true',
        limit=top_k,
        output_fields=["product_id", "title", "price", "image_url"]
    )

    return results[0]
```

---

## Run Example

```python
# Index product
index_product({
    "product_id": "SKU001",
    "title": "Apple iPhone 15 Pro Max 256GB Natural Titanium",
    "description": "Titanium design, A17 Pro chip, 48MP main camera...",
    "image_path": "iphone15.jpg",
    "category": "Phones",
    "brand": "Apple",
    "price": 1199
})

# Text search - balanced
results = search_products("apple phone good camera", mode="balanced")

# Text search - title focused
results = search_products("iPhone 15 Pro Max", mode="title")

# Text search - visual focused
results = search_products("silver metallic phone", mode="visual")

# Image search
results = search_by_image("user_upload.jpg")

for r in results:
    print(f"{r['title']} - ${r['price']}")
```
