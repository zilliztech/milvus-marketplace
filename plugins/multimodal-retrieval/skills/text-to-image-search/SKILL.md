---
name: text-to-image-search
description: "Use when user needs to search images using natural language descriptions. Triggers on: text to image, describe and find, natural language image search, image caption search, find image by description."
---

# Text-to-Image Search - Search Images with Text

Search images using natural language descriptions, supporting complex semantics (e.g., "red car turning right").

## Use Cases

- Stock image search (designers finding assets)
- Surveillance video retrieval (finding specific scenes)
- Medical image retrieval (finding cases by description)
- E-commerce (finding product images by natural language)

## Architecture

### Option A: CLIP Direct Search (Simple)

```
Text Description → CLIP Text Encoding → Vector Search(CLIP Image Vectors) → Results
```

### Option B: VLM Description + Text Search (Complex Semantics)

```
Image → VLM Generate Caption → Text Embedding → Storage
Text Description → Text Embedding → Vector Search → Results
```

## Data Processing

Batch image processing is recommended using Ray orchestration (see `core:ray`).

**Option A (CLIP) Key Steps**:

1. Load image (PIL)
2. CLIP encode image
3. Write to Milvus

**Option B (VLM) Key Steps**:

1. Load image (PIL)
2. VLM generate caption (GPT-4o/Qwen-VL)
3. BGE encode caption
4. Write to Milvus

## Schema Design

### Option A: CLIP

```python
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("image_path", DataType.VARCHAR, max_length=512)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=512)
```

### Option B: VLM

```python
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("image_path", DataType.VARCHAR, max_length=512)
schema.add_field("caption", DataType.VARCHAR, max_length=4096)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)
```

## Search Implementation

```python
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

class TextToImageSearch:
    def __init__(self, uri: str = "./milvus.db", use_vlm: bool = False):
        self.client = MilvusClient(uri=uri)
        self.use_vlm = use_vlm

        if use_vlm:
            self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            self.collection = "vlm_image_search"
        else:
            self.model = SentenceTransformer('clip-ViT-B-32')
            self.collection = "clip_image_search"

    def search(self, text_query: str, limit: int = 10):
        """Search images with text"""
        embedding = self.model.encode(text_query).tolist()

        output_fields = ["image_path"]
        if self.use_vlm:
            output_fields.append("caption")

        results = self.client.search(
            collection_name=self.collection,
            data=[embedding],
            limit=limit,
            output_fields=output_fields
        )

        return [{"path": hit["entity"]["image_path"],
                 "caption": hit["entity"].get("caption", ""),
                 "score": hit["distance"]} for hit in results[0]]

# Usage
search = TextToImageSearch(use_vlm=True)
results = search.search("a red car turning right at an intersection")
```

## Option Comparison

| Option | Pros | Cons | Use Cases |
|--------|------|------|-----------|
| CLIP Direct | Simple, fast, no API cost | Weak complex semantics | General image search |
| VLM + Text | Strong complex semantics | Slow indexing, API cost | Surveillance, medical, specialized |

## VLM Selection

| Model | Features | API/Local |
|-------|----------|-----------|
| GPT-4o | Best quality, expensive | API |
| Claude 3 | Good quality | API |
| Qwen-VL | Good for Chinese, affordable | API |
| LLaVA | Open source, local | Local |

## Vertical Applications

See detailed guides in `verticals/` directory:
- `stock-image.md` - Stock image search
- `surveillance.md` - Surveillance video retrieval
- `medical-imaging.md` - Medical imaging

## Related Tools

- Data processing orchestration: `core:ray`
- Vectorization: `core:embedding`
- Image-to-image search: `multimodal-retrieval:image-search`
