# E-commerce Visual Search / Photo Shopping

## Data Characteristics

- Product image quality varies
- User-uploaded images may have noise (background, lighting)
- Need to identify product main subject
- Strong category filtering requirements

## Recommended Configuration

| Config Item | Recommended Value | Notes |
|-------------|-------------------|-------|
| Embedding Model | `openai/clip-vit-large-patch14` | High accuracy |
| | `openai/clip-vit-base-patch32` | Speed priority |
| | `CN-CLIP` | Chinese products |
| Index Type | IVF_PQ | Large data volume |
| | HNSW | High accuracy |
| Vector Dimension | 512/768 | Depends on model |

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("product_id", DataType.VARCHAR, max_length=64)
schema.add_field("image_url", DataType.VARCHAR, max_length=512)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=512)

# Product information
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("category_l1", DataType.VARCHAR, max_length=64)
schema.add_field("category_l2", DataType.VARCHAR, max_length=64)
schema.add_field("brand", DataType.VARCHAR, max_length=128)
schema.add_field("price", DataType.FLOAT)
schema.add_field("in_stock", DataType.BOOL)

# Image information
schema.add_field("image_type", DataType.VARCHAR, max_length=32)    # main/detail/user
schema.add_field("is_primary", DataType.BOOL)                       # Whether primary image
```

## Image Preprocessing

```python
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class EcommerceImageSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess user-uploaded image"""
        # 1. Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 2. Resize (maintain aspect ratio)
        max_size = 800
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        # 3. Center crop (optional, for removing background)
        # image = self.center_crop(image)

        return image

    def extract_main_object(self, image: Image.Image) -> Image.Image:
        """Extract product main subject (using segmentation model)"""
        # Can integrate SAM or other segmentation models here
        # from segment_anything import SamPredictor
        # Simplified version: return original image
        return image

    def encode_image(self, image: Image.Image) -> list:
        """Image vectorization"""
        image = self.preprocess_image(image)
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        return features[0].tolist()
```

## Search Strategies

### 1. Category Prediction + Filtering

```python
def search_with_category_prediction(self, image: Image.Image, limit: int = 20):
    """Search with category prediction and filtering"""
    embedding = self.encode_image(image)

    # First search without filtering to get top results
    initial_results = self.client.search(
        collection_name="product_images",
        data=[embedding],
        filter='is_primary == true and in_stock == true',
        limit=50,
        output_fields=["category_l1"]
    )

    # Calculate category distribution
    from collections import Counter
    categories = [r["entity"]["category_l1"] for r in initial_results[0][:10]]
    most_common = Counter(categories).most_common(1)

    # If there's a dominant category, re-search with filter
    if most_common and most_common[0][1] >= 5:
        predicted_category = most_common[0][0]
        return self.client.search(
            collection_name="product_images",
            data=[embedding],
            filter=f'category_l1 == "{predicted_category}" and is_primary == true and in_stock == true',
            limit=limit,
            output_fields=["product_id", "title", "price", "image_url"]
        )

    return initial_results
```

### 2. Multi-Image Fusion Search

```python
def search_with_multiple_images(self, images: list, limit: int = 20):
    """Multi-image fusion search (user uploads multiple images)"""
    embeddings = [self.encode_image(img) for img in images]

    # Average fusion
    import numpy as np
    fused_embedding = np.mean(embeddings, axis=0).tolist()

    return self.client.search(
        collection_name="product_images",
        data=[fused_embedding],
        filter='is_primary == true and in_stock == true',
        limit=limit,
        output_fields=["product_id", "title", "price", "image_url"]
    )
```

### 3. Price Range Filtering

```python
def search_with_price_range(self, image: Image.Image, min_price: float = None,
                            max_price: float = None, limit: int = 20):
    """Search with price filtering"""
    embedding = self.encode_image(image)

    filter_parts = ['is_primary == true', 'in_stock == true']
    if min_price is not None:
        filter_parts.append(f'price >= {min_price}')
    if max_price is not None:
        filter_parts.append(f'price <= {max_price}')

    filter_expr = ' and '.join(filter_parts)

    return self.client.search(
        collection_name="product_images",
        data=[embedding],
        filter=filter_expr,
        limit=limit,
        output_fields=["product_id", "title", "price", "image_url"]
    )
```

## Data Import

```python
def add_product_images(self, products: list):
    """Batch add product images
    products: [{"product_id": "...", "images": [...], "title": "...", ...}]
    """
    data = []

    for product in products:
        for i, img_url in enumerate(product["images"]):
            # Download image
            image = download_image(img_url)
            embedding = self.encode_image(image)

            data.append({
                "id": f"{product['product_id']}_{i}",
                "product_id": product["product_id"],
                "image_url": img_url,
                "embedding": embedding,
                "title": product["title"],
                "category_l1": product["category_l1"],
                "category_l2": product.get("category_l2", ""),
                "brand": product.get("brand", ""),
                "price": product["price"],
                "in_stock": product.get("in_stock", True),
                "image_type": "main" if i == 0 else "detail",
                "is_primary": i == 0
            })

    self.client.insert(collection_name="product_images", data=data)
```

## Examples

```python
search = EcommerceImageSearch()

# User uploads image for search
user_image = Image.open("user_upload.jpg")
results = search.search_with_category_prediction(user_image)

# With price filtering
results = search.search_with_price_range(user_image, min_price=100, max_price=500)

# Multi-image search
images = [Image.open(f"upload_{i}.jpg") for i in range(3)]
results = search.search_with_multiple_images(images)
```
