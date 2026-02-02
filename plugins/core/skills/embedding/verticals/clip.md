# CLIP Series (Image Embedding)

OpenAI's open-source cross-modal image-text model, the go-to choice for image search.

## Model Versions

| Model | Dimensions | Features |
|-------|-----------|----------|
| **clip-ViT-L-14** | 768 | Highest accuracy, recommended |
| clip-ViT-B-32 | 512 | Fast, balanced |
| clip-ViT-B-16 | 512 | Medium |
| **Chinese-CLIP-ViT-H** | 1024 | Chinese optimized, largest |
| Chinese-CLIP-ViT-L | 768 | Chinese optimized |
| Chinese-CLIP-ViT-B | 512 | Chinese optimized, lightweight |

## Installation

```bash
pip install sentence-transformers
# Or
pip install transformers pillow
```

## Code Examples

### Basic Usage (sentence-transformers)

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

# Load model
model = SentenceTransformer('clip-ViT-B-32')

# Encode image
image = Image.open("cat.jpg")
image_embedding = model.encode(image)
print(f"Dimensions: {len(image_embedding)}")  # 512

# Encode text (for image-text matching)
text = "a photo of a cat"
text_embedding = model.encode(text)

# Batch images
images = [Image.open(f"img_{i}.jpg") for i in range(10)]
image_embeddings = model.encode(images, batch_size=8)
```

### High-Accuracy Model

```python
# Use ViT-L-14 (higher accuracy)
model = SentenceTransformer('clip-ViT-L-14')

image = Image.open("image.jpg")
embedding = model.encode(image)
print(f"Dimensions: {len(embedding)}")  # 768
```

### Chinese-CLIP (Chinese)

```python
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image

# Load model
model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14")
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14")

# Encode image
image = Image.open("cat.jpg")
inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs)
image_embedding = image_features[0].detach().numpy()

# Encode Chinese text
text = "a cat"
inputs = processor(text=text, return_tensors="pt")
text_features = model.get_text_features(**inputs)
text_embedding = text_features[0].detach().numpy()
```

### Chinese-CLIP Wrapper Class

```python
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image
import torch
from typing import List, Union
import numpy as np

class ChineseCLIPEmbedding:
    def __init__(self, model_name: str = "OFA-Sys/chinese-clip-vit-large-patch14", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChineseCLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = ChineseCLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def encode_images(self, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Encode images"""
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                all_embeddings.append(features.cpu().numpy())

        return np.vstack(all_embeddings)

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                all_embeddings.append(features.cpu().numpy())

        return np.vstack(all_embeddings)

# Usage
clip = ChineseCLIPEmbedding()

# Encode images
images = [Image.open(f"img_{i}.jpg") for i in range(10)]
image_embeddings = clip.encode_images(images)

# Encode Chinese texts
texts = ["a cat", "a dog", "a car"]
text_embeddings = clip.encode_texts(texts)
```

### GPU Acceleration

```python
from sentence_transformers import SentenceTransformer

# Use GPU
model = SentenceTransformer('clip-ViT-B-32', device='cuda')

# Batch processing
images = [Image.open(f"img_{i}.jpg") for i in range(100)]
embeddings = model.encode(
    images,
    batch_size=32,  # Larger batch for GPU
    show_progress_bar=True
)
```

## Milvus Integration

### Image-to-Image Search

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from PIL import Image
import os

# Initialize
client = MilvusClient(uri="./milvus.db")
model = SentenceTransformer('clip-ViT-B-32')

# Create Collection
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("image_path", DataType.VARCHAR, max_length=512)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=512)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("image_search", schema=schema, index_params=index_params)

# Batch add images
image_dir = "./images"
paths = []
images = []
for f in os.listdir(image_dir):
    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(image_dir, f)
        paths.append(path)
        images.append(Image.open(path))

embeddings = model.encode(images, batch_size=32).tolist()
data = [{"image_path": p, "embedding": e} for p, e in zip(paths, embeddings)]
client.insert("image_search", data)

# Image-to-image search
query_image = Image.open("query.jpg")
query_embedding = model.encode(query_image).tolist()

results = client.search(
    collection_name="image_search",
    data=[query_embedding],
    limit=10,
    output_fields=["image_path"]
)

for hit in results[0]:
    print(f"Similar image: {hit['entity']['image_path']}, Similarity: {hit['distance']:.3f}")
```

### Text-to-Image Search

```python
# Search images with text
query_text = "a cat sitting on a sofa"
text_embedding = model.encode(query_text).tolist()

results = client.search(
    collection_name="image_search",
    data=[text_embedding],
    limit=10,
    output_fields=["image_path"]
)
```

## Image Preprocessing

```python
from PIL import Image

def preprocess_image(image_path: str, max_size: int = 512) -> Image.Image:
    """Standardize image format"""
    image = Image.open(image_path)

    # Convert to RGB (handle PNG transparency)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Limit size
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))

    return image

# Usage
image = preprocess_image("image.png")
embedding = model.encode(image)
```

## Model Selection Recommendations

| Scenario | Recommended Model | Reason |
|----------|------------------|--------|
| General image search | clip-ViT-B-32 | Fast, good enough |
| High-accuracy needs | clip-ViT-L-14 | Highest accuracy |
| Chinese image-text | Chinese-CLIP-ViT-L | Better Chinese understanding |
| Resource-constrained | clip-ViT-B-32 | Lightest |

## Limits and Notes

1. **Image resolution**: CLIP internally resizes to 224x224, pre-scaling large images speeds up processing
2. **Text length**: CLIP max 77 tokens (about 40 Chinese characters)
3. **Image-text matching**: Image and text embeddings from the same model can be compared
4. **Chinese support**: Original CLIP has average Chinese performance, recommend Chinese-CLIP

## Model Download (Offline Use)

```bash
# sentence-transformers model
huggingface-cli download sentence-transformers/clip-ViT-B-32 --local-dir ./clip-vit-b-32

# Chinese-CLIP
huggingface-cli download OFA-Sys/chinese-clip-vit-large-patch14 --local-dir ./chinese-clip

# Use local model
model = SentenceTransformer('./clip-vit-b-32')
```

## China Mirror

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('clip-ViT-B-32')
```
