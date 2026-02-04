# GIF Search

> Search for GIFs by uploading a similar GIF or describing with text.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Search Mode

<ask_user>
How do you want to search for GIFs?

| Mode | Description |
|------|-------------|
| **GIF-to-GIF** | Upload a GIF, find similar animations |
| **Text-to-GIF** | Describe like "cat dancing", find matching GIFs |
| **Both** (recommended) | Support both search modes |
</ask_user>

### 2. Data Source

<ask_user>
Where are your GIFs?

| Source | Notes |
|--------|-------|
| **Local folder** | Your own GIF collection |
| **GIPHY API** | Need free API key from developers.giphy.com |
| **Tenor API** | Need free API key from tenor.com |
</ask_user>

### 3. GIF Embedding Strategy

<ask_user>
How to embed GIFs (which have multiple frames)?

| Strategy | Pros | Cons |
|----------|------|------|
| **Middle frame** (recommended) | Fast, simple | May miss animation context |
| **Multi-frame average** | Captures motion | Slower, may blur features |
| **Key frames** | Best quality | Slower, more storage |
</ask_user>

### 4. Data Scale

<ask_user>
How many GIFs do you have?

- Each GIF = 1-3 vectors (depending on strategy)

| GIF Count | Recommended Milvus |
|-----------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 5. Project Setup

<ask_user>
Choose project management:

| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production projects |
| **pip** | Quick prototypes |
</ask_user>

---

## Dependencies

### uv
```bash
uv init gif-search
cd gif-search
uv add pymilvus transformers torch Pillow requests
```

### pip
```bash
pip install pymilvus transformers torch Pillow requests
```

---

## End-to-End Implementation

### Step 1: Configure CLIP Model

```python
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

DIMENSION = 512

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

### Step 2: Extract Frames from GIF

```python
from PIL import Image, ImageSequence
import os

def extract_gif_frames(gif_path: str, strategy: str = "middle") -> list[Image.Image]:
    """Extract representative frames from a GIF."""
    gif = Image.open(gif_path)
    frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(gif)]

    if len(frames) == 0:
        return []

    if strategy == "middle":
        # Return just the middle frame
        return [frames[len(frames) // 2]]

    elif strategy == "multi":
        # Return first, middle, last
        indices = [0, len(frames) // 2, len(frames) - 1]
        return [frames[i] for i in indices if i < len(frames)]

    elif strategy == "keyframes":
        # Sample every N frames
        n = max(1, len(frames) // 5)
        return frames[::n][:5]  # Max 5 frames

    return [frames[0]]

def load_gifs_from_folder(folder: str, strategy: str = "middle") -> list[dict]:
    """Load all GIFs from a folder."""
    gifs = []

    for file in os.listdir(folder):
        if file.lower().endswith(".gif"):
            path = os.path.join(folder, file)
            try:
                frames = extract_gif_frames(path, strategy)
                if frames:
                    gifs.append({
                        "path": path,
                        "filename": file,
                        "frames": frames
                    })
            except Exception as e:
                print(f"Error loading {path}: {e}")

    return gifs

gifs = load_gifs_from_folder("./my_gifs", strategy="middle")
```

### Step 3: Download from GIPHY (optional)

```python
import requests

GIPHY_API_KEY = "your_api_key"  # Get free key from developers.giphy.com

def search_giphy(query: str, limit: int = 50) -> list[dict]:
    """Search GIFs from GIPHY API."""
    url = "https://api.giphy.com/v1/gifs/search"
    params = {
        "api_key": GIPHY_API_KEY,
        "q": query,
        "limit": limit
    }

    resp = requests.get(url, params=params)
    data = resp.json()

    gifs = []
    for item in data.get("data", []):
        gifs.append({
            "id": item["id"],
            "title": item.get("title", ""),
            "url": item["images"]["fixed_height"]["url"],
            "preview_url": item["images"]["preview_gif"]["url"]
        })

    return gifs

def download_gif(url: str, output_path: str) -> str:
    """Download a GIF from URL."""
    resp = requests.get(url, timeout=30)
    with open(output_path, "wb") as f:
        f.write(resp.content)
    return output_path

def fetch_giphy_dataset(queries: list[str], per_query: int = 100, output_dir: str = "giphy_gifs"):
    """Build a GIF dataset from GIPHY."""
    os.makedirs(output_dir, exist_ok=True)
    all_gifs = []

    for query in queries:
        print(f"Fetching GIFs for: {query}")
        results = search_giphy(query, limit=per_query)

        for gif in results:
            try:
                path = os.path.join(output_dir, f"{gif['id']}.gif")
                if not os.path.exists(path):
                    download_gif(gif["url"], path)
                all_gifs.append({
                    "path": path,
                    "filename": f"{gif['id']}.gif",
                    "title": gif["title"]
                })
            except Exception as e:
                print(f"Error downloading {gif['id']}: {e}")

    return all_gifs

# Build dataset with popular categories
queries = ["cat", "dog", "reaction", "funny", "dance", "anime"]
gifs = fetch_giphy_dataset(queries, per_query=50)
```

### Step 4: Index into Milvus

```python
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("gifs.db")  # Milvus Lite
# client = MilvusClient(uri="http://localhost:19530")  # Standalone

client.create_collection(
    collection_name="gifs",
    dimension=DIMENSION,
    auto_id=True
)

def index_gifs(gifs: list[dict], batch_size: int = 32):
    """Embed and index GIFs."""
    data = []

    for i, gif in enumerate(gifs):
        # Load frames if not already loaded
        if "frames" not in gif:
            frames = extract_gif_frames(gif["path"], strategy="middle")
            if not frames:
                continue
            gif["frames"] = frames

        # Embed frames
        embeddings = embed_images(gif["frames"])

        if len(embeddings) == 1:
            # Single frame strategy
            data.append({
                "vector": embeddings[0],
                "path": gif["path"],
                "filename": gif["filename"],
                "title": gif.get("title", "")
            })
        else:
            # Multi-frame: average the embeddings
            avg_embedding = np.mean(embeddings, axis=0).tolist()
            data.append({
                "vector": avg_embedding,
                "path": gif["path"],
                "filename": gif["filename"],
                "title": gif.get("title", "")
            })

        if len(data) >= batch_size:
            client.insert(collection_name="gifs", data=data)
            print(f"Indexed {i + 1}/{len(gifs)}")
            data = []

    # Insert remaining
    if data:
        client.insert(collection_name="gifs", data=data)

    print(f"Indexed {len(gifs)} GIFs")

index_gifs(gifs)
```

### Step 5: Search

```python
def search_by_gif(gif_path: str, top_k: int = 5):
    """Search by uploading a GIF."""
    frames = extract_gif_frames(gif_path, strategy="middle")
    if not frames:
        print("Could not extract frames from GIF")
        return []

    embeddings = embed_images(frames)
    query_vector = embeddings[0] if len(embeddings) == 1 else np.mean(embeddings, axis=0).tolist()

    results = client.search(
        collection_name="gifs",
        data=[query_vector],
        limit=top_k,
        output_fields=["path", "filename", "title"]
    )
    return results[0]

def search_by_text(query: str, top_k: int = 5):
    """Search by text description."""
    query_vector = embed_text([query])[0]

    results = client.search(
        collection_name="gifs",
        data=[query_vector],
        limit=top_k,
        output_fields=["path", "filename", "title"]
    )
    return results[0]

def display_results(results):
    """Display search results."""
    for i, hit in enumerate(results, 1):
        entity = hit["entity"]
        print(f"#{i} Score: {hit['distance']:.3f}")
        print(f"    Title: {entity['title']}")
        print(f"    File: {entity['filename']}")
        print()
```

---

## Run Example

```python
# Index local GIFs
gifs = load_gifs_from_folder("./my_gifs")
index_gifs(gifs)

# Search by text
results = search_by_text("cat jumping")
display_results(results)

# Search by GIF
results = search_by_gif("./query.gif")
display_results(results)
```

---

## Advanced: Tenor API Alternative

```python
TENOR_API_KEY = "your_api_key"  # Get from tenor.com

def search_tenor(query: str, limit: int = 50) -> list[dict]:
    """Search GIFs from Tenor API."""
    url = "https://tenor.googleapis.com/v2/search"
    params = {
        "key": TENOR_API_KEY,
        "q": query,
        "limit": limit,
        "media_filter": "gif"
    }

    resp = requests.get(url, params=params)
    data = resp.json()

    gifs = []
    for item in data.get("results", []):
        gifs.append({
            "id": item["id"],
            "title": item.get("content_description", ""),
            "url": item["media_formats"]["gif"]["url"]
        })

    return gifs
```
