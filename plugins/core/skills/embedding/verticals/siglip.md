# SigLIP (Google's Next-Gen Image Model)

Google's open-source next-generation vision-language model, an improved version of CLIP with better performance.

## Model Versions

| Model | Dimensions | Resolution | Size | Features |
|-------|-----------|-----------|------|----------|
| **siglip-so400m-patch14-384** | 1152 | 384 | 878MB | Recommended, best performance |
| siglip-base-patch16-256 | 768 | 256 | 380MB | Base version |
| siglip-base-patch16-384 | 768 | 384 | High resolution |
| siglip-large-patch16-384 | 1024 | 384 | 652MB | Large model |

## SigLIP vs CLIP

| Comparison | SigLIP | CLIP |
|-----------|--------|------|
| Training method | Sigmoid Loss | Contrastive Loss |
| Zero-shot performance | Better | Baseline |
| Chinese capability | Average | Chinese-CLIP better |
| Inference speed | Similar | Similar |

## Installation

```bash
pip install transformers pillow torch
# Or use sentence-transformers
pip install sentence-transformers
```

## Code Examples

### Basic Usage (transformers)

```python
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

# Load model
model_name = "google/siglip-so400m-patch14-384"
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Encode image
image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    image_features = model.get_image_features(**inputs)
    image_embedding = image_features[0].numpy()

print(f"Dimensions: {len(image_embedding)}")  # 1152

# Encode text
text = "a photo of a cat"
inputs = processor(text=text, return_tensors="pt", padding=True)

with torch.no_grad():
    text_features = model.get_text_features(**inputs)
    text_embedding = text_features[0].numpy()
```

### Batch Encoding

```python
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

def encode_images(images: list, batch_size: int = 16):
    """Batch encode images"""
    all_embeddings = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt")

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.numpy())

    import numpy as np
    return np.vstack(all_embeddings)

# Usage
images = [Image.open(f"img_{i}.jpg") for i in range(100)]
embeddings = encode_images(images)
```

### GPU Acceleration

```python
import torch
from transformers import AutoProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# Move to GPU when encoding
image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    features = model.get_image_features(**inputs)
    embedding = features[0].cpu().numpy()
```

### Wrapper Class

```python
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import numpy as np
from typing import List, Union

class SigLIPEmbedding:
    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
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
siglip = SigLIPEmbedding()

# Encode images
images = [Image.open(f"img_{i}.jpg") for i in range(10)]
image_embeddings = siglip.encode_images(images)

# Encode texts
texts = ["a cat", "a dog", "a car"]
text_embeddings = siglip.encode_texts(texts)
```

## Milvus Integration

```python
from pymilvus import MilvusClient, DataType
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import os

# Initialize
client = MilvusClient(uri="./milvus.db")
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# Create Collection (SigLIP so400m has 1152 dimensions)
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("image_path", DataType.VARCHAR, max_length=512)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1152)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("siglip_images", schema=schema, index_params=index_params)

# Insert images
image_dir = "./images"
paths = []
embeddings = []

for f in os.listdir(image_dir):
    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(image_dir, f)
        image = Image.open(path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            embedding = features[0].numpy().tolist()

        paths.append(path)
        embeddings.append(embedding)

data = [{"image_path": p, "embedding": e} for p, e in zip(paths, embeddings)]
client.insert("siglip_images", data)

# Image-to-image search
query_image = Image.open("query.jpg").convert("RGB")
inputs = processor(images=query_image, return_tensors="pt")
with torch.no_grad():
    features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    query_embedding = features[0].numpy().tolist()

results = client.search(
    collection_name="siglip_images",
    data=[query_embedding],
    limit=10,
    output_fields=["image_path"]
)

# Text-to-image search
query_text = "a cute cat"
inputs = processor(text=query_text, return_tensors="pt", padding=True)
with torch.no_grad():
    features = model.get_text_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    text_embedding = features[0].numpy().tolist()

results = client.search(
    collection_name="siglip_images",
    data=[text_embedding],
    limit=10,
    output_fields=["image_path"]
)
```

## Image Preprocessing

```python
from PIL import Image

def preprocess_image(image_path: str) -> Image.Image:
    """Standardize image format"""
    image = Image.open(image_path)

    # Convert to RGB (handle PNG transparency, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

# Usage
image = preprocess_image("image.png")
inputs = processor(images=image, return_tensors="pt")
```

## Model Download (Offline Use)

```bash
# Download model
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir ./siglip

# Use local model
model = AutoModel.from_pretrained("./siglip")
processor = AutoProcessor.from_pretrained("./siglip")
```

## China Mirror

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, AutoModel
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
```

## Limits and Notes

| Limit | Value |
|-------|-------|
| Input resolution | 256/384 (depends on model) |
| Text length | 64 tokens |
| Language | Primarily English |

**Notes**:
- Chinese performance not as good as Chinese-CLIP
- Need to convert images to RGB format
- Text and images are in the same vector space, enabling cross-modal retrieval

## Selection Recommendations

| Scenario | Recommended Model |
|----------|------------------|
| Highest accuracy | siglip-so400m-patch14-384 |
| Balanced | siglip-large-patch16-384 |
| Lightweight | siglip-base-patch16-256 |
| Chinese image-text | Chinese-CLIP is better |

## SigLIP vs CLIP: How to Choose

- **Choose SigLIP**: Pursuing highest accuracy, English scenarios
- **Choose CLIP**: Chinese scenarios (use Chinese-CLIP), more community support
