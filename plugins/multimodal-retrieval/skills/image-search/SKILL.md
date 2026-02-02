---
name: image-search
description: "Use when user wants to build image search or similar image finding. Triggers on: image search, similar image, visual search, image retrieval, CLIP, reverse image search, image matching."
---

# Image Search

Build image-to-image or text-to-image search systems.

## Data Processing

For batch image processing, use Ray orchestration (see `core:ray`).

**Key Steps**:

1. **Load images**: PIL/OpenCV loading
2. **Vectorize**: CLIP encodes images
3. **Write to Milvus**: Batch insert

**Model Selection**:

| Model | Dimensions | Features |
|-------|------------|----------|
| clip-ViT-B-32 | 512 | Fast, general |
| clip-ViT-L-14 | 768 | High accuracy |
| chinese-clip | 512 | Chinese optimized |

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("image_path", DataType.VARCHAR, max_length=512)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=512)

index_params.add_index("embedding", index_type="HNSW", metric_type="COSINE",
                       params={"M": 16, "efConstruction": 256})
```

## Search Implementation

```python
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from PIL import Image

class ImageSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('clip-ViT-B-32')

    def search_by_image(self, image_path: str, limit: int = 10):
        """Image-to-image search"""
        image = Image.open(image_path)
        embedding = self.model.encode(image).tolist()

        results = self.client.search(
            collection_name="image_search",
            data=[embedding],
            limit=limit,
            output_fields=["image_path"]
        )

        return [{"path": hit["entity"]["image_path"], "score": hit["distance"]}
                for hit in results[0]]

    def search_by_text(self, text: str, limit: int = 10):
        """Text-to-image search"""
        embedding = self.model.encode(text).tolist()

        results = self.client.search(
            collection_name="image_search",
            data=[embedding],
            limit=limit,
            output_fields=["image_path"]
        )

        return [{"path": hit["entity"]["image_path"], "score": hit["distance"]}
                for hit in results[0]]

# Usage
search = ImageSearch()
results = search.search_by_image("query.jpg")
results = search.search_by_text("a cat sitting on a sofa")
```

## Model Selection

| Model | Dimensions | Features |
|-------|------------|----------|
| clip-ViT-B-32 | 512 | Fast, general |
| clip-ViT-L-14 | 768 | High accuracy |
| chinese-clip | 512 | Chinese optimized |

### Using Chinese-CLIP

```python
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

# Image encoding
inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs)

# Text encoding
inputs = processor(text="a cat", return_tensors="pt")
text_features = model.get_text_features(**inputs)
```

## Image Preprocessing

```python
from PIL import Image
import io

def preprocess_image(image_path: str, max_size: int = 512):
    """Standardize image format and size"""
    image = Image.open(image_path)

    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Limit size
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))

    return image
```

## Image Search with Tags

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="./milvus.db")

# Create schema with tags field
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="tags", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20, max_length=64)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=512)

index_params = client.prepare_index_params()
index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 256})

client.create_collection(collection_name="image_search_tags", schema=schema, index_params=index_params)

# Filter during search
results = client.search(
    collection_name="image_search_tags",
    data=[embedding],
    limit=10,
    filter='array_contains(tags, "cat")',  # Filter images with cat tag
    output_fields=["image_path", "tags"]
)
```

## Related Tools

- Data processing orchestration: `core:ray`
- Vectorization: `core:embedding`
- Index optimization: `core:indexing`
- Text-to-image search: `multimodal-retrieval:text-to-image-search`
