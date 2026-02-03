---
name: text-to-image-search
description: "Use when user needs to search images using natural language descriptions. Triggers on: text to image, describe and find, natural language image search, image caption search, find image by description, describe to find."
---

# Text-to-Image Search

Search images using natural language descriptions — find visuals by describing what you're looking for.

## When to Activate

Activate this skill when:
- User wants to **find images by describing them** in natural language
- User mentions "find image of", "search for pictures of", "describe and find"
- User has **complex visual queries** ("red car turning right at an intersection")
- User's queries are **descriptive sentences**, not just keywords

**Do NOT activate** when:
- User has an **image to find similar ones** → use `image-search`
- User wants Q&A on documents with images → use `multimodal-rag`
- User needs video search → use `video-search`

## Interactive Flow

### Step 1: Assess Query Complexity

"What type of text queries will users make?"

A) **Simple queries** ("cat", "sunset beach", "red car")
   - CLIP direct encoding works well
   - Fast, no extra API costs

B) **Complex queries** ("a red car turning right at an intersection at night")
   - May need VLM-generated captions
   - Better semantic understanding

C) **Domain-specific** ("tumor in left lung lobe", "fault line in seismic data")
   - May need specialized models or VLM descriptions
   - Domain vocabulary matters

Which describes your queries? (A/B/C)

### Step 2: Choose Architecture

"Based on query complexity, here are your options:"

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| **A: CLIP Direct** | Text → CLIP → Search | Fast, free | Weak on complex queries |
| **B: VLM Captions** | Image → VLM → Caption → Text embedding | Better semantics | Slow indexing, API cost |

### Step 3: Confirm Configuration

"Based on your requirements:

- **Architecture**: [CLIP Direct / VLM Captions]
- **Model**: [clip-ViT-B-32 / BGE + GPT-4o]
- **Index**: AUTOINDEX with COSINE

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Two Approaches

Think of text-to-image search as **two different libraries**:

**Option A: CLIP (Shared Language)**
- Both images and text are translated to the same "language" (vector space)
- Works because CLIP was trained on image-text pairs
- Like having a bilingual librarian who speaks both "image" and "text"

**Option B: VLM Captions (Description Matching)**
- Each image gets a detailed text description
- Search matches query to descriptions
- Like having someone describe every image in words first

```
┌─────────────────────────────────────────────────────────────┐
│                    Option A: CLIP Direct                     │
│                                                              │
│  Indexing:                      Search:                      │
│  Image → CLIP → Vector          Text → CLIP → Vector         │
│                                           │                  │
│           Same vector space! ────────────→│                  │
│                                           ▼                  │
│                                      Find similar            │
│                                         vectors              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                Option B: VLM + Text Embedding                │
│                                                              │
│  Indexing:                                                   │
│  Image → VLM → "A red car..." → BGE → Vector                │
│                                                              │
│  Search:                                                     │
│  Query: "red vehicle" → BGE → Vector → Find similar         │
│                                                              │
│  Matching happens in text embedding space                    │
└─────────────────────────────────────────────────────────────┘
```

### When to Use Each

| Scenario | Best Option | Why |
|----------|-------------|-----|
| Simple queries, high volume | CLIP Direct | Fast, no API cost |
| Complex descriptions | VLM Captions | Better understanding |
| Domain-specific (medical, legal) | VLM Captions | Can prompt for domain terms |
| Budget constrained | CLIP Direct | Free |
| Quality critical | VLM Captions | More accurate |

## Implementation

### Option A: CLIP Direct

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from PIL import Image

class CLIPTextToImageSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('clip-ViT-B-32')
        self.dim = 512
        self.collection_name = "clip_image_search"
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
        """Index images with CLIP embeddings."""
        images = [Image.open(p).convert('RGB') for p in image_paths]
        embeddings = self.model.encode(images).tolist()

        data = [{"image_path": path, "embedding": emb}
                for path, emb in zip(image_paths, embeddings)]

        self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, text_query: str, limit: int = 10):
        """Search images with text description."""
        # CLIP encodes text into same space as images
        embedding = self.model.encode(text_query).tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=limit,
            output_fields=["image_path"]
        )

        return [{"path": hit["entity"]["image_path"], "score": hit["distance"]}
                for hit in results[0]]

# Usage
search = CLIPTextToImageSearch()
search.add_images(["beach.jpg", "city.jpg", "forest.jpg"])
results = search.search("sunset over the ocean")
```

### Option B: VLM Captions

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import base64

class VLMTextToImageSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.text_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.openai = OpenAI()
        self.dim = 1024
        self.collection_name = "vlm_image_search"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("image_path", DataType.VARCHAR, max_length=512)
        schema.add_field("caption", DataType.VARCHAR, max_length=4096)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def _generate_caption(self, image_path: str) -> str:
        """Generate detailed caption using VLM."""
        with open(image_path, "rb") as f:
            b64_image = base64.standard_b64encode(f.read()).decode()

        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail. Include objects, actions, colors, setting, and any text visible."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }],
            max_tokens=500
        )
        return response.choices[0].message.content

    def add_images(self, image_paths: list):
        """Index images with VLM-generated captions."""
        data = []
        for path in image_paths:
            caption = self._generate_caption(path)
            embedding = self.text_model.encode(caption).tolist()
            data.append({
                "image_path": path,
                "caption": caption,
                "embedding": embedding
            })

        self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, text_query: str, limit: int = 10):
        """Search images with text description."""
        embedding = self.text_model.encode(text_query).tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=limit,
            output_fields=["image_path", "caption"]
        )

        return [{
            "path": hit["entity"]["image_path"],
            "caption": hit["entity"]["caption"],
            "score": hit["distance"]
        } for hit in results[0]]

# Usage
search = VLMTextToImageSearch()
search.add_images(["traffic.jpg"])
results = search.search("a red car turning right at an intersection")
```

## Comparison Table

| Aspect | CLIP Direct | VLM Captions |
|--------|-------------|--------------|
| **Indexing speed** | Fast (ms per image) | Slow (seconds per image) |
| **Query speed** | Fast | Fast |
| **API cost** | Free | ~$0.01 per image |
| **Simple queries** | ★★★★ | ★★★★★ |
| **Complex queries** | ★★★ | ★★★★★ |
| **Domain-specific** | ★★ | ★★★★ |
| **Storage** | 512d vector only | 1024d vector + text |

## Common Pitfalls

### ❌ Pitfall 1: Expecting CLIP to Understand Complex Queries

**Problem**: "red car turning right at night" returns random cars

**Why**: CLIP wasn't trained on such specific scene descriptions

**Fix**: Use VLM captions for complex queries

### ❌ Pitfall 2: VLM Captions Too Generic

**Problem**: All captions say "This is an image of..."

**Why**: Default prompts generate generic descriptions

**Fix**: Use specific prompts
```python
prompt = """Describe this image with:
1. Main objects and their colors
2. Actions or movements
3. Setting/environment
4. Time of day if visible
5. Any text in the image"""
```

### ❌ Pitfall 3: Mixing Vector Spaces

**Problem**: Search returns nothing

**Why**: Used BGE to embed query but CLIP for images

**Fix**: Use same model for query as for indexing
```python
# CLIP indexing → CLIP query
# BGE caption indexing → BGE query
# Never mix!
```

### ❌ Pitfall 4: High VLM Costs

**Problem**: $100+ API bill for 10K images

**Why**: Using GPT-4o for everything

**Fix**: Use cheaper models or local VLMs
```python
# Cheaper options:
# - gpt-4o-mini (~$0.003 per image)
# - Local LLaVA (free, requires GPU)
# - Qwen-VL API (cheaper for Chinese)
```

## When to Level Up

| Need | Upgrade To |
|------|------------|
| Also search by image | Add `image-search` capability |
| Q&A on image content | `multimodal-rag` |
| Filter by metadata | Add `filtered-search` pattern |
| Video content | `video-search` |

## References

- VLM model comparison: See OpenAI, Anthropic, Google docs
- CLIP variants: `image-search/references/image-embeddings.md`
- Batch processing: `core:ray`
