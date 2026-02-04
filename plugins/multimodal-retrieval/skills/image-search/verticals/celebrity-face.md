# Celebrity Face Search

> Find similar-looking celebrities or search faces in a database.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Use Case

<ask_user>
What's your use case?

| Use Case | Description |
|----------|-------------|
| **Find similar celebrity** | Upload a photo, find which celebrity you look like |
| **Face retrieval** | Search for a specific person in a face database |
| **Both** | Support both modes |
</ask_user>

### 2. Face Embedding Library

<ask_user>
Choose a face embedding library:

| Library | Size | Notes |
|---------|------|-------|
| `facenet-pytorch` (recommended) | ~100MB | Lightweight, easy to install, FaceNet/InceptionResnet |
| `deepface` | ~500MB+ | Multiple backends (VGG-Face, ArcFace, Facenet), more flexible |
| `insightface` | ~300MB | High accuracy ArcFace, requires onnxruntime |
</ask_user>

### 3. Data Source

<ask_user>
Where are your face images?

| Source | Notes |
|--------|-------|
| **Local folder** | Your own image collection |
| **CelebA dataset** | 200K+ celebrity faces (Kaggle/torchvision) |
| **LFW dataset** | 13K+ faces, smaller (torchvision) |
</ask_user>

### 4. Data Scale

<ask_user>
How many face images do you have?

- Each face = 1 vector

| Face Count | Recommended Milvus |
|------------|-------------------|
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

### facenet-pytorch + uv (recommended)
```bash
uv init face-search
cd face-search
uv add pymilvus facenet-pytorch Pillow torch torchvision
```

### deepface + uv
```bash
uv init face-search
cd face-search
uv add pymilvus deepface Pillow tf-keras
```

### pip
```bash
pip install pymilvus facenet-pytorch Pillow torch torchvision
# or
pip install pymilvus deepface Pillow
```

---

## End-to-End Implementation

### Step 1: Configure Face Embedding

```python
# === Choose ONE ===

# Option A: facenet-pytorch (recommended)
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Face detector
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    device=device,
    selection_method="largest"  # Select largest face if multiple
)

# Face encoder
encoder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

DIMENSION = 512

def embed_faces(images: list[Image.Image]) -> list[list[float]]:
    """Detect faces and extract embeddings."""
    embeddings = []
    for img in images:
        # Detect and align face
        face = mtcnn(img)
        if face is None:
            embeddings.append(None)
            continue

        # Get embedding
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = encoder(face)
        embeddings.append(emb.cpu().numpy().flatten().tolist())

    return embeddings


# Option B: deepface
from deepface import DeepFace
import numpy as np

DIMENSION = 512  # Depends on model

def embed_faces(image_paths: list[str]) -> list[list[float]]:
    """Extract face embeddings using DeepFace."""
    embeddings = []
    for path in image_paths:
        try:
            result = DeepFace.represent(
                img_path=path,
                model_name="Facenet",  # or "VGG-Face", "ArcFace"
                enforce_detection=False
            )
            embeddings.append(result[0]["embedding"])
        except Exception as e:
            print(f"Error processing {path}: {e}")
            embeddings.append(None)
    return embeddings
```

### Step 2: Load Images

```python
import os
from pathlib import Path

def load_images_from_folder(folder: str) -> list[dict]:
    """Load all images from a folder."""
    images = []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for root, dirs, files in os.walk(folder):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                path = os.path.join(root, file)
                try:
                    img = Image.open(path).convert("RGB")
                    # Try to extract name from path/filename
                    name = Path(file).stem
                    images.append({
                        "path": path,
                        "name": name,
                        "image": img
                    })
                except Exception as e:
                    print(f"Error loading {path}: {e}")

    return images

images = load_images_from_folder("./celebrity_faces")
```

### Step 3: Load CelebA Dataset (optional)

```python
from torchvision.datasets import CelebA
from torchvision import transforms

def load_celeba_subset(root: str = "./data", max_samples: int = 5000):
    """Load CelebA subset."""
    # Download CelebA (first time takes a while)
    dataset = CelebA(
        root=root,
        split="all",
        download=True,
        transform=transforms.ToTensor()
    )

    images = []
    for i in range(min(len(dataset), max_samples)):
        img_tensor, _ = dataset[i]
        # Convert tensor to PIL
        img = transforms.ToPILImage()(img_tensor)
        images.append({
            "path": f"celeba_{i}",
            "name": f"celeb_{i}",
            "image": img
        })

    return images

# Load subset
images = load_celeba_subset(max_samples=5000)
```

### Step 4: Index into Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient("faces.db")  # Milvus Lite
# client = MilvusClient(uri="http://localhost:19530")  # Standalone

client.create_collection(
    collection_name="faces",
    dimension=DIMENSION,
    auto_id=True
)

def index_faces(images: list[dict], batch_size: int = 32):
    """Detect faces, embed, and index."""
    valid_data = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        pil_images = [item["image"] for item in batch]
        embeddings = embed_faces(pil_images)

        for item, emb in zip(batch, embeddings):
            if emb is not None:  # Face detected
                valid_data.append({
                    "vector": emb,
                    "path": item["path"],
                    "name": item["name"]
                })

        print(f"Processed {i + len(batch)}/{len(images)}")

    # Insert all valid faces
    if valid_data:
        client.insert(collection_name="faces", data=valid_data)
        print(f"Indexed {len(valid_data)} faces")

index_faces(images)
```

### Step 5: Search

```python
def search_similar_face(image_path: str, top_k: int = 5):
    """Find similar faces by uploading an image."""
    img = Image.open(image_path).convert("RGB")
    embeddings = embed_faces([img])

    if embeddings[0] is None:
        print("No face detected in query image")
        return []

    results = client.search(
        collection_name="faces",
        data=[embeddings[0]],
        limit=top_k,
        output_fields=["path", "name"]
    )
    return results[0]

def display_results(results):
    """Display search results."""
    print("\nSimilar faces found:")
    for i, hit in enumerate(results, 1):
        entity = hit["entity"]
        # Lower distance = more similar for L2
        similarity = 1 / (1 + hit["distance"])  # Convert to similarity score
        print(f"#{i} {entity['name']} (similarity: {similarity:.2%})")
        print(f"    Path: {entity['path']}")
```

---

## Run Example

```python
# Index celebrity faces
images = load_images_from_folder("./celebrity_faces")
index_faces(images)

# Upload a photo and find similar celebrities
results = search_similar_face("./my_photo.jpg", top_k=5)
display_results(results)
```

---

## Advanced: Face Verification

```python
def verify_faces(image1_path: str, image2_path: str, threshold: float = 0.6) -> bool:
    """Check if two images contain the same person."""
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")

    embeddings = embed_faces([img1, img2])

    if None in embeddings:
        print("Face not detected in one or both images")
        return False

    # Calculate cosine similarity
    import numpy as np
    emb1 = np.array(embeddings[0])
    emb2 = np.array(embeddings[1])

    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    print(f"Similarity: {similarity:.3f}")
    return similarity > threshold

# Example
is_same = verify_faces("./person1.jpg", "./person2.jpg")
print(f"Same person: {is_same}")
```

---

## Advanced: Group by Identity

```python
def cluster_faces(images: list[dict], threshold: float = 0.7) -> dict[int, list[str]]:
    """Group faces by identity using simple clustering."""
    # Embed all faces
    pil_images = [item["image"] for item in images]
    embeddings = embed_faces(pil_images)

    # Simple greedy clustering
    import numpy as np
    clusters = {}
    cluster_centroids = []
    cluster_id = 0

    for item, emb in zip(images, embeddings):
        if emb is None:
            continue

        emb = np.array(emb)
        assigned = False

        # Check similarity to existing clusters
        for cid, centroid in enumerate(cluster_centroids):
            similarity = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
            if similarity > threshold:
                clusters[cid].append(item["path"])
                assigned = True
                break

        if not assigned:
            clusters[cluster_id] = [item["path"]]
            cluster_centroids.append(emb)
            cluster_id += 1

    return clusters
```
