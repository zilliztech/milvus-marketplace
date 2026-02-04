# Fashion Image Search

> Search for similar clothing items by uploading an image or describing with text.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Search Mode

<ask_user>
How do you want to search for fashion items?

| Mode | Description |
|------|-------------|
| **Image-to-Image** | Upload a photo, find similar items |
| **Text-to-Image** | Describe "red dress", find matching items |
| **Both** (recommended) | Support both search modes |
</ask_user>

### 2. Image Embedding Model

<ask_user>
Choose an image embedding model:

| Model | Size | Notes |
|-------|------|-------|
| `openai/clip-vit-base-patch32` (recommended) | 600MB | General CLIP, good balance |
| `openai/clip-vit-large-patch14` | 1.7GB | Higher quality, needs more VRAM |
| `patrickjohncyh/fashion-clip` | 600MB | Fine-tuned for fashion |
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 600MB | Trained on larger dataset |
</ask_user>

### 3. Data Source

<ask_user>
Where are your fashion images?

| Source | Notes |
|--------|-------|
| **Local folder** | Point to image directory |
| **DeepFashion dataset** | Download subset from Kaggle/HuggingFace |
| **E-commerce API** | Crawl from product pages |
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
uv init fashion-search
cd fashion-search
uv add pymilvus transformers torch Pillow
```

### pip
```bash
pip install pymilvus transformers torch Pillow
```

### Optional: Download DeepFashion subset
```bash
# From HuggingFace datasets
uv add datasets
```

---

## End-to-End Implementation

### Step 1: Configure CLIP Model

```python
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load CLIP model
model_name = "openai/clip-vit-base-patch32"  # Change based on your choice
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

DIMENSION = 512  # CLIP base: 512, large: 768

def embed_images(images: list[Image.Image]) -> list[list[float]]:
    """Embed images using CLIP."""
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)  # Normalize

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

### Step 2: Load Images

```python
import os
from pathlib import Path

def load_images_from_folder(folder: str, extensions: set = {".jpg", ".jpeg", ".png", ".webp"}):
    """Load all images from a folder."""
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                path = os.path.join(root, file)
                try:
                    img = Image.open(path).convert("RGB")
                    images.append({
                        "path": path,
                        "filename": file,
                        "image": img
                    })
                except Exception as e:
                    print(f"Error loading {path}: {e}")
    return images

# Example: Load from local folder
images = load_images_from_folder("./fashion_images")
```

### Step 3: Load DeepFashion (optional)

```python
from datasets import load_dataset

def load_deepfashion_subset(max_samples: int = 5000):
    """Load DeepFashion subset from HuggingFace."""
    # Note: This uses a community dataset, availability may vary
    ds = load_dataset("logasanjeev/DeepFashion", split="train")

    images = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        images.append({
            "path": f"deepfashion_{i}",
            "filename": f"item_{i}.jpg",
            "image": item["image"].convert("RGB"),
            "category": item.get("label", "unknown")
        })
    return images

# Load subset
images = load_deepfashion_subset(max_samples=5000)
```

### Step 4: Index into Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient("fashion.db")  # Milvus Lite
# client = MilvusClient(uri="http://localhost:19530")  # Standalone

client.create_collection(
    collection_name="fashion",
    dimension=DIMENSION,
    auto_id=True
)

def index_images(images: list[dict], batch_size: int = 32):
    """Embed and index images in batches."""
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        pil_images = [item["image"] for item in batch]
        vectors = embed_images(pil_images)

        data = [
            {
                "vector": vec,
                "path": item["path"],
                "filename": item["filename"],
                "category": item.get("category", "")
            }
            for vec, item in zip(vectors, batch)
        ]
        client.insert(collection_name="fashion", data=data)
        print(f"Indexed {i + len(batch)}/{len(images)}")

# Index all images
index_images(images)
```

### Step 5: Search

```python
def search_by_image(image_path: str, top_k: int = 5):
    """Search by uploading an image."""
    img = Image.open(image_path).convert("RGB")
    query_vector = embed_images([img])[0]

    results = client.search(
        collection_name="fashion",
        data=[query_vector],
        limit=top_k,
        output_fields=["path", "filename", "category"]
    )
    return results[0]

def search_by_text(query: str, top_k: int = 5):
    """Search by text description."""
    query_vector = embed_text([query])[0]

    results = client.search(
        collection_name="fashion",
        data=[query_vector],
        limit=top_k,
        output_fields=["path", "filename", "category"]
    )
    return results[0]

def display_results(results):
    """Display search results."""
    for i, hit in enumerate(results, 1):
        entity = hit["entity"]
        print(f"#{i} Score: {hit['distance']:.3f}")
        print(f"    File: {entity['filename']}")
        print(f"    Category: {entity['category']}")
        print(f"    Path: {entity['path']}")
        print()
```

---

## Run Example

```python
# Index images
images = load_images_from_folder("./fashion_images")
index_images(images)

# Search by image
results = search_by_image("./query_dress.jpg")
display_results(results)

# Search by text
results = search_by_text("red summer dress with floral pattern")
display_results(results)

results = search_by_text("black leather jacket")
display_results(results)
```

---

## Advanced: Filter by Category

```python
def search_with_filter(query: str, category: str, top_k: int = 5):
    """Search with category filter."""
    query_vector = embed_text([query])[0]

    results = client.search(
        collection_name="fashion",
        data=[query_vector],
        limit=top_k,
        filter=f'category == "{category}"',
        output_fields=["path", "filename", "category"]
    )
    return results[0]

# Search only in "dress" category
results = search_with_filter("elegant evening wear", category="dress")
```
