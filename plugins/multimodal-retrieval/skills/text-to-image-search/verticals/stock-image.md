# Stock Image Search

## Use Cases

- Stock image library search
- Design resource retrieval
- News illustration lookup
- Advertising creative assets

## Two Approaches Comparison

| Approach | Use Case | Pros | Cons |
|----------|----------|------|------|
| CLIP Direct Search | Simple descriptions | Fast, no preprocessing needed | Weak complex semantics |
| VLM Description + Text Search | Complex queries | Strong semantic understanding | Requires preprocessing, higher cost |

## Recommended Strategy

For stock image scenarios, **dual-path combination** is recommended:
1. Indexing: Use VLM to generate image descriptions, store description text vectors
2. Search: User query directly vectorized for text search

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("image_url", DataType.VARCHAR, max_length=512)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

# VLM generated description
schema.add_field("description", DataType.VARCHAR, max_length=65535)
schema.add_field("tags", DataType.VARCHAR, max_length=512)          # Auto tags
schema.add_field("colors", DataType.VARCHAR, max_length=128)        # Dominant colors
schema.add_field("style", DataType.VARCHAR, max_length=64)          # Style

# Image properties
schema.add_field("width", DataType.INT32)
schema.add_field("height", DataType.INT32)
schema.add_field("orientation", DataType.VARCHAR, max_length=16)    # portrait/landscape/square
schema.add_field("format", DataType.VARCHAR, max_length=16)         # jpg/png/svg

# Usage information
schema.add_field("category", DataType.VARCHAR, max_length=64)       # Category
schema.add_field("license", DataType.VARCHAR, max_length=32)        # License type
schema.add_field("download_count", DataType.INT32)
```

## Implementation

```python
from pymilvus import MilvusClient, DataType
from openai import OpenAI
from PIL import Image
import base64

class StockImageSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()
        self._init_collection()

    def _embed(self, text: str) -> list:
        """Generate embedding using OpenAI API"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

    def _describe_image(self, image_path: str) -> dict:
        """Analyze image with VLM"""
        with open(image_path, "rb") as f:
            base64_image = base64.standard_b64encode(f.read()).decode()

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": """Please analyze this image and return JSON format:
{
  "description": "Detailed description of image content, scene, people, actions, etc.",
  "tags": ["tag1", "tag2", ...],
  "colors": ["dominant_color1", "dominant_color2"],
  "style": "Photo style: realistic/illustration/3d/flat/vintage etc.",
  "mood": "Emotional atmosphere: happy/serious/calm/energetic etc."
}"""},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            response_format={"type": "json_object"},
            max_tokens=500
        )

        import json
        return json.loads(response.choices[0].message.content)

    def _get_image_info(self, image_path: str) -> dict:
        """Get basic image information"""
        img = Image.open(image_path)
        width, height = img.size

        if width > height * 1.2:
            orientation = "landscape"
        elif height > width * 1.2:
            orientation = "portrait"
        else:
            orientation = "square"

        return {
            "width": width,
            "height": height,
            "orientation": orientation,
            "format": img.format.lower() if img.format else "unknown"
        }

    def add_image(self, image_path: str, image_id: str, category: str = "",
                  license_type: str = "standard"):
        """Add image to library"""
        # 1. VLM analysis
        analysis = self._describe_image(image_path)

        # 2. Get image info
        info = self._get_image_info(image_path)

        # 3. Vectorize description
        description = analysis["description"]
        embedding = self._embed(description).tolist()

        # 4. Store
        self.client.insert(
            collection_name="stock_images",
            data=[{
                "id": image_id,
                "image_url": image_path,
                "embedding": embedding,
                "description": description,
                "tags": ",".join(analysis.get("tags", [])),
                "colors": ",".join(analysis.get("colors", [])),
                "style": analysis.get("style", ""),
                "width": info["width"],
                "height": info["height"],
                "orientation": info["orientation"],
                "format": info["format"],
                "category": category,
                "license": license_type,
                "download_count": 0
            }]
        )

        return analysis

    def search(self, query: str,
               # Filter conditions
               category: str = None,
               orientation: str = None,
               style: str = None,
               colors: list = None,
               min_width: int = None,
               min_height: int = None,
               # Sorting
               sort_by: str = None,  # relevance/downloads
               limit: int = 20) -> list:
        """Search images with text"""
        # 1. Vectorize query
        embedding = self._embed(query).tolist()

        # 2. Build filter conditions
        filters = []

        if category:
            filters.append(f'category == "{category}"')

        if orientation:
            filters.append(f'orientation == "{orientation}"')

        if style:
            filters.append(f'style == "{style}"')

        if colors:
            # Color matching (any color)
            color_conds = [f'colors like "%{c}%"' for c in colors]
            filters.append(f'({" or ".join(color_conds)})')

        if min_width:
            filters.append(f'width >= {min_width}')

        if min_height:
            filters.append(f'height >= {min_height}')

        filter_expr = ' and '.join(filters) if filters else ""

        # 3. Search
        results = self.client.search(
            collection_name="stock_images",
            data=[embedding],
            filter=filter_expr,
            limit=limit * 2 if sort_by == "downloads" else limit,
            output_fields=["image_url", "description", "tags", "colors",
                          "style", "width", "height", "download_count"]
        )

        items = results[0]

        # 4. Sort
        if sort_by == "downloads":
            items.sort(key=lambda x: x["entity"]["download_count"], reverse=True)
            items = items[:limit]

        return [{
            "id": item["id"],
            "url": item["entity"]["image_url"],
            "description": item["entity"]["description"],
            "tags": item["entity"]["tags"].split(","),
            "colors": item["entity"]["colors"].split(","),
            "style": item["entity"]["style"],
            "dimensions": f"{item['entity']['width']}x{item['entity']['height']}",
            "downloads": item["entity"]["download_count"],
            "relevance": item["distance"]
        } for item in items]

    def search_similar(self, image_id: str, limit: int = 10) -> list:
        """Search similar images"""
        image = self.client.get(
            collection_name="stock_images",
            ids=[image_id],
            output_fields=["embedding", "style"]
        )

        if not image:
            return []

        # Same style + vector similarity
        results = self.client.search(
            collection_name="stock_images",
            data=[image[0]["embedding"]],
            filter=f'id != "{image_id}" and style == "{image[0]["style"]}"',
            limit=limit,
            output_fields=["image_url", "description"]
        )

        return results[0]

    def get_suggestions(self, query: str, limit: int = 10) -> list:
        """Search suggestions (autocomplete)"""
        # Autocomplete based on existing tags
        embedding = self._embed(query).tolist()

        results = self.client.search(
            collection_name="stock_images",
            data=[embedding],
            limit=50,
            output_fields=["tags"]
        )

        # Extract popular tags
        from collections import Counter
        all_tags = []
        for r in results[0]:
            all_tags.extend(r["entity"]["tags"].split(","))

        # Filter tags matching query
        matching = [t for t in all_tags if query.lower() in t.lower()]
        tag_counts = Counter(matching)

        return [{"tag": t, "count": c} for t, c in tag_counts.most_common(limit)]
```

## Examples

```python
search = StockImageSearch()

# Add image
analysis = search.add_image(
    image_path="business_meeting.jpg",
    image_id="img_001",
    category="business"
)
print(f"Image analysis: {analysis}")

# Text search
results = search.search(
    "business professionals discussing project in meeting room",
    category="business",
    orientation="landscape",
    min_width=1920
)

print("Search results:")
for r in results:
    print(f"  - {r['description'][:50]}...")
    print(f"    Tags: {', '.join(r['tags'][:5])}")
    print(f"    Dimensions: {r['dimensions']}")

# Color search
results = search.search(
    "blue sky white clouds natural landscape",
    colors=["blue", "white"],
    style="realistic"
)

# Search suggestions
suggestions = search.get_suggestions("bus")
print("Suggestions:", [s["tag"] for s in suggestions])
```

## Batch Processing

```python
def batch_add_images(self, image_dir: str, category: str = ""):
    """Batch add images"""
    import os
    from concurrent.futures import ThreadPoolExecutor

    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def process(image_file):
        image_path = os.path.join(image_dir, image_file)
        image_id = os.path.splitext(image_file)[0]
        try:
            return self.add_image(image_path, image_id, category)
        except Exception as e:
            print(f"Processing failed: {image_file}, {e}")
            return None

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process, images))

    success = sum(1 for r in results if r is not None)
    print(f"Processing complete: {success}/{len(images)}")
```
