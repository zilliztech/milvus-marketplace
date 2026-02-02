# E-commerce Multi-Vector Search

## Use Case

Products have multiple searchable dimensions:
- **Title**: Product name, keywords
- **Description**: Detailed description, selling points
- **Image**: Product appearance
- **Attributes**: Structured features

A single vector cannot fully capture product information.

## Multi-Vector Strategy

| Vector Field | Source | Purpose |
|-------------|--------|---------|
| title_vector | Title | Keyword matching |
| desc_vector | Description | Semantic understanding |
| image_vector | Image | Visual similarity |
| attr_vector | Attributes | Structured matching |

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)

# Multiple vector fields
schema.add_field("title_vector", DataType.FLOAT_VECTOR, dim=1024)
schema.add_field("desc_vector", DataType.FLOAT_VECTOR, dim=1024)
schema.add_field("image_vector", DataType.FLOAT_VECTOR, dim=512)

# Raw data
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("description", DataType.VARCHAR, max_length=65535)
schema.add_field("image_url", DataType.VARCHAR, max_length=512)

# Filter fields
schema.add_field("category", DataType.VARCHAR, max_length=64)
schema.add_field("brand", DataType.VARCHAR, max_length=128)
schema.add_field("price", DataType.FLOAT)
schema.add_field("in_stock", DataType.BOOL)

# Multi-vector indexes
index_params = client.prepare_index_params()
index_params.add_index("title_vector", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index("desc_vector", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index("image_vector", index_type="AUTOINDEX", metric_type="COSINE")
```

## Implementation

```python
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch

class MultiVectorProductSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.text_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def add_product(self, product: dict):
        """Add product"""
        # Generate vectors
        title_vec = self.text_model.encode(product["title"]).tolist()
        desc_vec = self.text_model.encode(product["description"]).tolist()

        # Image vector
        image = self._load_image(product["image_url"])
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_vec = self.clip_model.get_image_features(**inputs)
            image_vec = (image_vec / image_vec.norm(dim=-1, keepdim=True))[0].tolist()

        self.client.insert(
            collection_name="products",
            data=[{
                "id": product["id"],
                "title_vector": title_vec,
                "desc_vector": desc_vec,
                "image_vector": image_vec,
                "title": product["title"],
                "description": product["description"],
                "image_url": product["image_url"],
                "category": product.get("category", ""),
                "brand": product.get("brand", ""),
                "price": product.get("price", 0),
                "in_stock": product.get("in_stock", True)
            }]
        )

    def search(self, query: str,
               mode: str = "balanced",  # balanced/title/description/visual
               category: str = None,
               limit: int = 20) -> list:
        """Multi-vector search"""
        # Generate query vectors
        text_vec = self.text_model.encode(query).tolist()

        # Image query vector (using CLIP text encoding)
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            clip_text_vec = self.clip_model.get_text_features(**inputs)
            clip_text_vec = (clip_text_vec / clip_text_vec.norm(dim=-1, keepdim=True))[0].tolist()

        # Build filter condition
        filter_expr = 'in_stock == true'
        if category:
            filter_expr += f' and category == "{category}"'

        # Set weights based on mode
        if mode == "title":
            weights = [0.7, 0.2, 0.1]
        elif mode == "description":
            weights = [0.2, 0.7, 0.1]
        elif mode == "visual":
            weights = [0.2, 0.2, 0.6]
        else:  # balanced
            weights = [0.4, 0.4, 0.2]

        # Multi-vector search
        results = self.client.hybrid_search(
            collection_name="products",
            reqs=[
                AnnSearchRequest(
                    data=[text_vec],
                    anns_field="title_vector",
                    param={"metric_type": "COSINE"},
                    limit=50
                ),
                AnnSearchRequest(
                    data=[text_vec],
                    anns_field="desc_vector",
                    param={"metric_type": "COSINE"},
                    limit=50
                ),
                AnnSearchRequest(
                    data=[clip_text_vec],
                    anns_field="image_vector",
                    param={"metric_type": "COSINE"},
                    limit=50
                )
            ],
            ranker=WeightedRanker(*weights),
            filter=filter_expr,
            limit=limit,
            output_fields=["title", "description", "image_url", "price", "brand"]
        )

        return [{
            "id": r["id"],
            "title": r["entity"]["title"],
            "description": r["entity"]["description"][:100] + "...",
            "image_url": r["entity"]["image_url"],
            "price": r["entity"]["price"],
            "brand": r["entity"]["brand"],
            "score": r["distance"]
        } for r in results]

    def search_by_image(self, image_path: str, limit: int = 20) -> list:
        """Image-to-image search"""
        image = self._load_image(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_vec = self.clip_model.get_image_features(**inputs)
            image_vec = (image_vec / image_vec.norm(dim=-1, keepdim=True))[0].tolist()

        results = self.client.search(
            collection_name="products",
            data=[image_vec],
            anns_field="image_vector",
            filter='in_stock == true',
            limit=limit,
            output_fields=["title", "image_url", "price"]
        )

        return results[0]

    def search_similar(self, product_id: str, mode: str = "balanced", limit: int = 10) -> list:
        """Similar product recommendations"""
        product = self.client.get(
            collection_name="products",
            ids=[product_id],
            output_fields=["title_vector", "desc_vector", "image_vector", "category"]
        )

        if not product:
            return []

        p = product[0]

        # Multi-vector search for similar products
        results = self.client.hybrid_search(
            collection_name="products",
            reqs=[
                AnnSearchRequest(
                    data=[p["title_vector"]],
                    anns_field="title_vector",
                    param={"metric_type": "COSINE"},
                    limit=30
                ),
                AnnSearchRequest(
                    data=[p["desc_vector"]],
                    anns_field="desc_vector",
                    param={"metric_type": "COSINE"},
                    limit=30
                ),
                AnnSearchRequest(
                    data=[p["image_vector"]],
                    anns_field="image_vector",
                    param={"metric_type": "COSINE"},
                    limit=30
                )
            ],
            ranker=RRFRanker(k=60),
            filter=f'id != "{product_id}" and in_stock == true',
            limit=limit,
            output_fields=["title", "image_url", "price"]
        )

        return results

    def _load_image(self, path_or_url: str):
        """Load image"""
        from PIL import Image
        import requests
        from io import BytesIO

        if path_or_url.startswith(('http://', 'https://')):
            response = requests.get(path_or_url)
            return Image.open(BytesIO(response.content))
        else:
            return Image.open(path_or_url)
```

## Examples

```python
search = MultiVectorProductSearch()

# Add product
search.add_product({
    "id": "prod_001",
    "title": "Apple iPhone 15 Pro Max 256GB Natural Titanium",
    "description": "Titanium design, A17 Pro chip, 48MP main camera...",
    "image_url": "iphone15.jpg",
    "category": "phones",
    "brand": "Apple",
    "price": 1199
})

# Balanced search
results = search.search("apple phone good camera", mode="balanced")

# Title priority (precise lookup)
results = search.search("iPhone 15 Pro Max", mode="title")

# Description priority (feature search)
results = search.search("phone with great camera quality", mode="description")

# Visual priority
results = search.search("silver metallic phone", mode="visual")

# Image-to-image search
results = search.search_by_image("user_upload.jpg")

# Similar products
results = search.search_similar("prod_001")
```

## Dynamic Weight Strategy

```python
def get_dynamic_weights(query: str) -> tuple:
    """Automatically adjust weights based on query"""
    # Detect query type
    is_specific = any(word in query.lower() for word in ["model", "sku", "part number", "iphone", "samsung"])
    is_descriptive = len(query) > 20 or any(word in query.lower() for word in ["how", "what kind", "feature"])
    is_visual = any(word in query.lower() for word in ["color", "appearance", "look", "design"])

    if is_specific:
        return (0.7, 0.2, 0.1)  # Title priority
    elif is_visual:
        return (0.2, 0.2, 0.6)  # Visual priority
    elif is_descriptive:
        return (0.2, 0.6, 0.2)  # Description priority
    else:
        return (0.4, 0.4, 0.2)  # Balanced
```
