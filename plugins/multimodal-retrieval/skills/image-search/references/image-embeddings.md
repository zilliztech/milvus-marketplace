# Image Embedding Models Comparison

Guide to selecting the right image embedding model for your use case.

## Model Overview

| Model | Dimensions | Training Data | Best For |
|-------|------------|---------------|----------|
| CLIP ViT-B-32 | 512 | 400M image-text pairs | General, fast |
| CLIP ViT-L-14 | 768 | 400M image-text pairs | High accuracy |
| Chinese-CLIP | 512 | Chinese image-text | Chinese queries |
| SigLIP | 768 | Curated dataset | Latest, best quality |
| EVA-CLIP | 1024 | Extended training | Research, large-scale |

## CLIP Family

### CLIP ViT-B-32

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('clip-ViT-B-32')
# Image: 512 dim, Text: 512 dim (shared space)
```

**Pros**:
- Fastest inference
- Widely supported
- Good balance of speed/quality

**Cons**:
- Lower accuracy than larger models
- May miss fine details

**Best for**: Prototyping, high-throughput systems

### CLIP ViT-L-14

```python
model = SentenceTransformer('clip-ViT-L-14')
# Image: 768 dim, Text: 768 dim
```

**Pros**:
- Best accuracy in CLIP family
- Better fine-grained recognition

**Cons**:
- Slower (2-3x ViT-B-32)
- Larger memory footprint

**Best for**: Production with accuracy requirements

### CLIP ViT-B-16

```python
model = SentenceTransformer('clip-ViT-B-16')
# Image: 512 dim, Text: 512 dim
```

**Pros**:
- Better accuracy than B-32
- Faster than L-14

**Cons**:
- Middle ground, not best at anything

**Best for**: Balanced requirements

## Chinese-CLIP

For Chinese text-to-image search:

```python
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch

model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

# Encode image
image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

# Encode Chinese text
inputs = processor(text="一只橙色的猫", return_tensors="pt", padding=True)
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
```

**Variants**:
- `chinese-clip-vit-base-patch16` (512 dim) - Balanced
- `chinese-clip-vit-large-patch14` (768 dim) - Higher quality

## SigLIP (Recommended for New Projects)

Google's improved CLIP variant:

```python
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# Encode image
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_features = model.get_image_features(**inputs)
```

**Pros**:
- Better training objective (sigmoid instead of softmax)
- Improved zero-shot performance
- Better multilingual support

**Cons**:
- Newer, less community tooling
- May require more setup

## Performance Benchmarks

*Tested on ImageNet validation set*

| Model | Top-1 Accuracy | Inference Time | Memory |
|-------|---------------|----------------|--------|
| ViT-B-32 | 63.2% | 5ms | 350MB |
| ViT-B-16 | 68.3% | 8ms | 350MB |
| ViT-L-14 | 75.5% | 15ms | 890MB |
| SigLIP-B | 73.8% | 8ms | 400MB |

## Dimension vs Quality Trade-off

```
Quality
    ↑
    │     ┌─────────────────┐
    │     │   ViT-L-14      │ (768d)
    │     │   SigLIP-L      │
    │     └─────────────────┘
    │
    │  ┌─────────────────┐
    │  │   ViT-B-16      │ (512d)
    │  │   SigLIP-B      │
    │  └─────────────────┘
    │
    │  ┌─────────────────┐
    │  │   ViT-B-32      │ (512d)
    │  └─────────────────┘
    │
    └──────────────────────────→ Speed
```

## Choosing the Right Model

### Decision Tree

```
┌─────────────────────────────────────────┐
│     Which Image Model Should I Use?      │
├─────────────────────────────────────────┤
│                                          │
│  Is speed critical?                      │
│      └── Yes → ViT-B-32 ✓               │
│                                          │
│  Need Chinese text queries?              │
│      └── Yes → Chinese-CLIP ✓           │
│                                          │
│  Starting a new project?                 │
│      └── Yes → SigLIP ✓ (best default)  │
│                                          │
│  Need highest accuracy?                  │
│      └── Yes → ViT-L-14 or SigLIP-L ✓   │
│                                          │
│  Not sure?                               │
│      └── ViT-B-32 (safe default) ✓      │
│                                          │
└─────────────────────────────────────────┘
```

### By Use Case

| Use Case | Recommended | Why |
|----------|-------------|-----|
| E-commerce product search | ViT-L-14 | Accuracy matters |
| Real-time applications | ViT-B-32 | Speed matters |
| Chinese market | Chinese-CLIP | Language support |
| New project (2024+) | SigLIP | Best overall |
| Face recognition | Specialized (FaceNet) | Domain-specific |

## Batch Processing Tips

```python
from PIL import Image
import torch

def batch_encode_images(model, image_paths, batch_size=32):
    """Efficiently encode many images."""
    all_embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(p).convert('RGB') for p in batch_paths]

        # sentence-transformers handles batching
        embeddings = model.encode(images, convert_to_tensor=True)
        all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)
```

## Memory Optimization

```python
# For GPU with limited memory
import torch

# Use float16
model = model.half()

# Use gradient checkpointing (if fine-tuning)
model.gradient_checkpointing_enable()

# Process in smaller batches
with torch.no_grad():
    for batch in batches:
        embeddings = model.encode(batch)
        # Save to disk/Milvus immediately
        save_embeddings(embeddings)
        del embeddings
        torch.cuda.empty_cache()
```
