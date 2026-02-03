---
name: image-search
description: "Use when user wants to build image search or similar image finding. Triggers on: image search, similar image, visual search, image retrieval, CLIP, reverse image search, image matching, find similar photos."
---

# Image Search

Build image-to-image search systems that find visually similar images using deep learning embeddings.

## When to Activate

Activate this skill when:
- User wants to **find similar images** given an input image
- User mentions "reverse image search", "visual similarity", "find similar"
- User has an **image database** to search through
- User wants to build product visual search, face matching, or duplicate detection

**Do NOT activate** when:
- User wants to search images **using text** → use `text-to-image-search`
- User has **mixed text and image documents** → use `multimodal-rag`
- User needs video content search → use `video-search`

## Interactive Flow

### Step 1: Understand the Search Type

"What kind of image search do you need?"

A) **Image-to-Image**: Upload image, find similar ones
   - Product visual search (find similar products)
   - Duplicate/near-duplicate detection
   - Face recognition

B) **Text-to-Image**: Describe what you're looking for
   - Stock photo search
   - Descriptive queries ("red car on highway")

C) **Both**: Support both search modalities
   - E-commerce (upload photo OR describe)
   - Content management systems

Which do you need? (A/B/C)

### Step 2: Determine Image Types

"What types of images are in your database?"

| Type | Characteristics | Model Recommendation |
|------|-----------------|----------------------|
| **General photos** | Diverse subjects | CLIP ViT-B-32 |
| **Product images** | Clean backgrounds | CLIP ViT-L-14 |
| **Faces** | Portrait photos | FaceNet or CLIP |
| **Domain-specific** | Medical, satellite, etc. | Domain fine-tuned |

### Step 3: Confirm Configuration

"Based on your requirements:

- **Model**: CLIP ViT-B-32 (512 dim)
- **Index**: AUTOINDEX with COSINE
- **Search type**: Image-to-image

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Visual Fingerprint

Think of image embeddings as a **visual fingerprint**:
- Each image gets a unique 512-dimensional "fingerprint"
- Similar-looking images have similar fingerprints
- Search = find images with closest fingerprints

```
┌─────────────────────────────────────────────────────────┐
│                    Image Search                          │
│                                                          │
│  Query Image: [🐱 cat photo]                            │
│                    │                                     │
│                    ▼                                     │
│           ┌───────────────┐                             │
│           │     CLIP      │  Extract visual features    │
│           │   Encoder     │  (colors, shapes, objects)  │
│           └───────┬───────┘                             │
│                   │                                     │
│                   ▼                                     │
│           [0.23, -0.45, 0.12, ...]                      │
│           (512-dimensional vector)                      │
│                   │                                     │
│                   ▼                                     │
│      ┌────────────────────────┐                        │
│      │      Vector Index      │  Find similar vectors   │
│      │    (1M image vectors)  │  in milliseconds        │
│      └────────────┬───────────┘                        │
│                   │                                     │
│                   ▼                                     │
│  Results: [🐱] [🐱] [🐈] [🐕]                          │
│  (ranked by visual similarity)                          │
└─────────────────────────────────────────────────────────┘
```

### How CLIP Works

CLIP (Contrastive Language-Image Pre-training):
- Trained on 400M image-text pairs
- Learns to align images and text in same vector space
- **Image encoder**: Image → 512-dim vector
- **Text encoder**: Text → 512-dim vector (same space!)

This enables both image-to-image AND text-to-image search with one model.

## Why Image Search Over Alternatives

| Need | Best Solution |
|------|---------------|
| Find visually similar images | ✅ Image Search (this skill) |
| Search images by description | `text-to-image-search` |
| Q&A on documents with images | `multimodal-rag` |
| Search video content | `video-search` |

## Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from PIL import Image

class ImageSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('clip-ViT-B-32')
        self.dim = 512
        self.collection_name = "image_search"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("image_path", DataType.VARCHAR, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add_images(self, image_paths: list):
        """Add images to index"""
        images = [Image.open(p).convert('RGB') for p in image_paths]
        embeddings = self.model.encode(images).tolist()

        data = [{"image_path": path, "embedding": emb}
                for path, emb in zip(image_paths, embeddings)]

        self.client.insert(collection_name=self.collection_name, data=data)

    def search_by_image(self, image_path: str, limit: int = 10):
        """Image-to-image search"""
        image = Image.open(image_path).convert('RGB')
        embedding = self.model.encode(image).tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=limit,
            output_fields=["image_path"]
        )

        return [{"path": hit["entity"]["image_path"], "score": hit["distance"]}
                for hit in results[0]]

    def search_by_text(self, text: str, limit: int = 10):
        """Text-to-image search (CLIP enables this!)"""
        embedding = self.model.encode(text).tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=limit,
            output_fields=["image_path"]
        )

        return [{"path": hit["entity"]["image_path"], "score": hit["distance"]}
                for hit in results[0]]

# Usage
search = ImageSearch()

# Index images
search.add_images(["cat1.jpg", "cat2.jpg", "dog1.jpg"])

# Search by image
results = search.search_by_image("query.jpg")

# Search by text
results = search.search_by_text("a fluffy orange cat")
```

## Model Selection Guide

| Model | Dimensions | Speed | Quality | Best For |
|-------|------------|-------|---------|----------|
| **clip-ViT-B-32** | 512 | ★★★★★ | ★★★ | General, fast |
| **clip-ViT-L-14** | 768 | ★★★ | ★★★★★ | High accuracy |
| **clip-ViT-B-16** | 512 | ★★★★ | ★★★★ | Balanced |
| **chinese-clip** | 512 | ★★★★ | ★★★★ | Chinese text queries |
| **SigLIP** | 768 | ★★★ | ★★★★★ | Latest, best quality |

### Using Chinese-CLIP

```python
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch

class ChineseImageSearch:
    def __init__(self):
        self.model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        self.processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features[0].numpy().tolist()

    def encode_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features[0].numpy().tolist()
```

## Image Preprocessing

```python
from PIL import Image

def preprocess_image(image_path: str, max_size: int = 512) -> Image:
    """Standardize image for consistent embeddings."""
    image = Image.open(image_path)

    # Convert to RGB (handle PNG with transparency, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize if too large (saves memory, minimal quality impact)
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)

    return image
```

## Common Pitfalls

### ❌ Pitfall 1: Not Converting Image Mode

**Problem**: Error when encoding PNG images with transparency

**Why**: CLIP expects RGB, but PNG may be RGBA

**Fix**: Always convert to RGB
```python
image = Image.open(path).convert('RGB')
```

### ❌ Pitfall 2: Memory Issues with Large Images

**Problem**: Out of memory when processing high-resolution images

**Why**: CLIP resizes internally, but PIL loads full image

**Fix**: Resize before encoding
```python
if max(image.size) > 1024:
    image.thumbnail((1024, 1024))
```

### ❌ Pitfall 3: Expecting Text-Quality Results

**Problem**: Text search results seem worse than image search

**Why**: CLIP text-image alignment isn't perfect

**Fix**: For text-heavy use cases, consider `text-to-image-search` with VLM captions

### ❌ Pitfall 4: Searching with Very Different Images

**Problem**: Product photos don't match user uploads

**Why**: Different lighting, angles, backgrounds

**Fix**: Consider data augmentation during indexing or use filtered search to narrow domain

## When to Level Up

| Need | Upgrade To |
|------|------------|
| Better text search results | `text-to-image-search` with VLM |
| Filter by metadata (category, date) | Add `filtered-search` pattern |
| Q&A on image content | `multimodal-rag` |
| Video content search | `video-search` |

## References

- Image embedding models: `references/image-embeddings.md`
- Batch processing: `core:ray`
- Index configuration: `core:indexing`
