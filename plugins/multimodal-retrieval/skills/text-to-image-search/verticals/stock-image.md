# Stock Image Search

> Search stock images using text descriptions with VLM-generated captions.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Image Description Language

<ask_user>
What language do you want to search in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** | English models |
| **Chinese** | Chinese models |
| **Multilingual** | Multilingual models |
</ask_user>

### 2. Image Description Method

<ask_user>
Choose how to generate image descriptions:

| Method | Pros | Cons |
|--------|------|------|
| **VLM (GPT-4o/Qwen-VL)** | Rich, detailed descriptions | Higher cost |
| **CLIP Direct** | Fast, no preprocessing | Weaker complex semantics |
| **Hybrid** (recommended) | VLM for indexing, text search | Best of both |
</ask_user>

### 3. Text Embedding

<ask_user>
Choose text embedding for search:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download |

Local options:
- `all-MiniLM-L6-v2` (384d, 80MB)
- `BAAI/bge-base-en-v1.5` (768d, 440MB)
</ask_user>

### 4. Data Scale

<ask_user>
How many images do you have?

- Each image = 1 vector

| Image Count | Recommended Milvus |
|-------------|-------------------|
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
uv init stock-image-search
cd stock-image-search
uv add pymilvus openai Pillow
```

---

## End-to-End Implementation

### Step 1: Configure Models

```python
from openai import OpenAI
from PIL import Image
import base64

client = OpenAI()

def embed_text(texts: list[str]) -> list[list[float]]:
    """Embed text using OpenAI."""
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

def describe_image(image_path: str) -> dict:
    """Analyze image with VLM."""
    with open(image_path, "rb") as f:
        base64_image = base64.standard_b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": """Analyze this image and return JSON:
{
  "description": "Detailed description of image content",
  "tags": ["tag1", "tag2", ...],
  "colors": ["dominant_color1", "dominant_color2"],
  "style": "realistic/illustration/3d/vintage",
  "mood": "happy/serious/calm/energetic"
}"""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }],
        response_format={"type": "json_object"},
        max_tokens=500
    )

    import json
    return json.loads(response.choices[0].message.content)
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("stock_images.db")

schema = milvus.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("image_id", DataType.VARCHAR, max_length=64)
schema.add_field("image_url", DataType.VARCHAR, max_length=512)
schema.add_field("description", DataType.VARCHAR, max_length=65535)
schema.add_field("tags", DataType.VARCHAR, max_length=512)
schema.add_field("colors", DataType.VARCHAR, max_length=128)
schema.add_field("style", DataType.VARCHAR, max_length=64)
schema.add_field("category", DataType.VARCHAR, max_length=64)
schema.add_field("width", DataType.INT32)
schema.add_field("height", DataType.INT32)
schema.add_field("orientation", DataType.VARCHAR, max_length=16)

index_params = milvus.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

milvus.create_collection("stock_images", schema=schema, index_params=index_params)
```

### Step 3: Index Images

```python
def get_image_info(image_path: str) -> dict:
    """Get image dimensions and orientation."""
    img = Image.open(image_path)
    width, height = img.size

    if width > height * 1.2:
        orientation = "landscape"
    elif height > width * 1.2:
        orientation = "portrait"
    else:
        orientation = "square"

    return {"width": width, "height": height, "orientation": orientation}

def add_image(image_path: str, image_id: str, category: str = ""):
    """Add image to library."""
    # VLM analysis
    analysis = describe_image(image_path)

    # Image info
    info = get_image_info(image_path)

    # Embed description
    description = analysis["description"]
    embedding = embed_text([description])[0]

    milvus.insert(
        collection_name="stock_images",
        data=[{
            "embedding": embedding,
            "image_id": image_id,
            "image_url": image_path,
            "description": description,
            "tags": ",".join(analysis.get("tags", [])),
            "colors": ",".join(analysis.get("colors", [])),
            "style": analysis.get("style", ""),
            "category": category,
            "width": info["width"],
            "height": info["height"],
            "orientation": info["orientation"]
        }]
    )

    return analysis
```

### Step 4: Search

```python
def search_images(query: str,
                  category: str = None,
                  orientation: str = None,
                  style: str = None,
                  min_width: int = None,
                  top_k: int = 20):
    """Search images by text description."""
    query_embedding = embed_text([query])[0]

    filters = []
    if category:
        filters.append(f'category == "{category}"')
    if orientation:
        filters.append(f'orientation == "{orientation}"')
    if style:
        filters.append(f'style == "{style}"')
    if min_width:
        filters.append(f'width >= {min_width}')

    filter_expr = ' and '.join(filters) if filters else None

    results = milvus.search(
        collection_name="stock_images",
        data=[query_embedding],
        filter=filter_expr,
        limit=top_k,
        output_fields=["image_id", "image_url", "description", "tags", "style"]
    )

    return [{
        "image_id": r["entity"]["image_id"],
        "url": r["entity"]["image_url"],
        "description": r["entity"]["description"],
        "tags": r["entity"]["tags"].split(","),
        "style": r["entity"]["style"],
        "relevance": r["distance"]
    } for r in results[0]]

def search_similar(image_id: str, top_k: int = 10):
    """Find similar images."""
    image = milvus.get(
        collection_name="stock_images",
        ids=[image_id],
        output_fields=["embedding", "style"]
    )

    if not image:
        return []

    results = milvus.search(
        collection_name="stock_images",
        data=[image[0]["embedding"]],
        filter=f'image_id != "{image_id}" and style == "{image[0]["style"]}"',
        limit=top_k,
        output_fields=["image_id", "image_url", "description"]
    )

    return results[0]
```

---

## Run Example

```python
# Add images
analysis = add_image(
    image_path="business_meeting.jpg",
    image_id="img_001",
    category="business"
)
print(f"Analysis: {analysis}")

# Text search
results = search_images(
    "business professionals discussing project in meeting room",
    category="business",
    orientation="landscape",
    min_width=1920
)

for r in results:
    print(f"- {r['description'][:50]}...")
    print(f"  Tags: {', '.join(r['tags'][:5])}")

# Color-based search
results = search_images(
    "blue sky white clouds natural landscape",
    style="realistic"
)

# Similar images
similar = search_similar("img_001")
```
