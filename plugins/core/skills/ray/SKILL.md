---
name: ray
description: "Use when user needs to process data at scale. Triggers on: batch processing, data ingestion, pipeline, parallel processing, GPU acceleration, video processing, PDF processing, large-scale."
---

# Ray - Distributed Data Processing Framework

All data processing tasks (batch import, video processing, PDF parsing, etc.) should use Ray for orchestration.

## When to Use Ray

| Scenario | Use Ray? |
|----------|----------|
| Processing < 100 items | Optional, simple loops work |
| Processing 100 - 10,000 items | Recommended, single-machine multi-core parallel |
| Processing > 10,000 items | Required, can scale to cluster |
| Need GPU acceleration | Required, Ray manages GPU resources |
| Multi-step pipeline | Recommended, clear DAG orchestration |

## Installation

```bash
pip install "ray[data]"
```

## Core Concepts

### 1. Ray Task - Parallel Functions

Turn regular functions into parallelizable tasks:

```python
import ray

ray.init()

@ray.remote
def process_file(file_path):
    # Process single file
    return result

# Process 1000 files in parallel
files = ["file1.pdf", "file2.pdf", ...]
futures = [process_file.remote(f) for f in files]
results = ray.get(futures)  # Wait for all to complete
```

### 2. Ray Actor - Stateful Services

Suitable for scenarios that need model loading (model loads only once):

```python
@ray.remote(num_gpus=1)
class EmbeddingActor:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

    def encode(self, texts):
        return self.model.encode(texts).tolist()

# Create Actor (model loads once)
actor = EmbeddingActor.remote()

# Multiple calls (reuse model)
vec1 = ray.get(actor.encode.remote(["Text 1"]))
vec2 = ray.get(actor.encode.remote(["Text 2"]))
```

### 3. Ray Data - Large-scale Data Pipelines

Suitable for batch data processing:

```python
import ray

ds = ray.data.from_items(file_list)

# Chained processing
ds = ds.map(step1)           # Step 1
ds = ds.map(step2)           # Step 2
ds = ds.map_batches(step3, batch_size=32)  # Batch processing

# Output
ds.write_parquet("output/")
```

## Common Patterns

### Pattern 1: Batch File Processing

```python
import ray

@ray.remote
def process_pdf(pdf_path):
    """Process single PDF"""
    # 1. Extract text
    # 2. Chunk
    # 3. Return results
    return chunks

def batch_process_pdfs(pdf_paths: list):
    ray.init(ignore_reinit_error=True)

    futures = [process_pdf.remote(p) for p in pdf_paths]
    results = ray.get(futures)

    return results
```

### Pattern 2: GPU Model Inference

```python
import ray

@ray.remote(num_gpus=1)
class ModelActor:
    def __init__(self, model_name):
        self.model = load_model(model_name)

    def predict(self, batch):
        return self.model(batch)

def batch_inference(items: list, batch_size: int = 32):
    ray.init(ignore_reinit_error=True)

    # Create multiple Actors for parallel inference
    num_actors = 4
    actors = [ModelActor.remote("model_name") for _ in range(num_actors)]

    results = []
    for i, batch in enumerate(chunked(items, batch_size)):
        actor = actors[i % num_actors]
        results.append(actor.predict.remote(batch))

    return ray.get(results)
```

### Pattern 3: Multi-step Pipeline (Parallel Branches)

```python
import ray

@ray.remote
def extract_audio(video_path):
    """Extract audio"""
    ...

@ray.remote
def extract_frames(video_path):
    """Extract key frames"""
    ...

@ray.remote
def transcribe(audio_path):
    """ASR transcription"""
    ...

@ray.remote
def embed_texts(texts):
    """Text vectorization"""
    ...

@ray.remote
def embed_images(images):
    """Image vectorization"""
    ...

def process_video(video_path):
    """Video processing pipeline"""
    ray.init(ignore_reinit_error=True)

    # Step 1: Parallel extract audio and frames
    audio_ref = extract_audio.remote(video_path)
    frames_ref = extract_frames.remote(video_path)

    # Step 2: ASR (depends on audio)
    transcript_ref = transcribe.remote(audio_ref)

    # Step 3: Parallel vectorization
    text_vec_ref = embed_texts.remote(transcript_ref)
    image_vec_ref = embed_images.remote(frames_ref)

    # Wait for all results
    text_vecs, image_vecs = ray.get([text_vec_ref, image_vec_ref])

    return {"text_vectors": text_vecs, "image_vectors": image_vecs}
```

### Pattern 4: Ray Data Batch Import to Milvus

```python
import ray
from pymilvus import MilvusClient

@ray.remote
class MilvusWriterActor:
    def __init__(self, uri, collection):
        self.client = MilvusClient(uri=uri)
        self.collection = collection

    def write(self, batch):
        self.client.insert(collection_name=self.collection, data=batch)
        return len(batch)

def batch_import_to_milvus(data_list: list, uri: str, collection: str):
    ray.init(ignore_reinit_error=True)

    ds = ray.data.from_items(data_list)

    # Batch write
    ds.map_batches(
        MilvusWriterActor,
        batch_size=1000,
        fn_constructor_kwargs={"uri": uri, "collection": collection},
        compute=ray.data.ActorPoolStrategy(size=2)
    )
```

## Resource Configuration

### CPU Tasks

```python
@ray.remote(num_cpus=2)  # Each task uses 2 CPUs
def cpu_task():
    ...
```

### GPU Tasks

```python
@ray.remote(num_gpus=1)  # Each task uses 1 GPU
def gpu_task():
    ...

@ray.remote(num_gpus=0.5)  # Two tasks share 1 GPU
def small_gpu_task():
    ...
```

### Actor Pool

```python
# Create pool of 4 Actors with automatic load balancing
ds.map_batches(
    MyActor,
    compute=ray.data.ActorPoolStrategy(size=4, num_gpus=1)
)
```

## Error Handling

```python
@ray.remote(max_retries=3)  # Auto-retry 3 times on failure
def unreliable_task():
    ...

# Error handling for batch tasks
futures = [process.remote(item) for item in items]

results = []
for future in futures:
    try:
        result = ray.get(future)
        results.append(result)
    except Exception as e:
        print(f"Task failed: {e}")
        results.append(None)
```

## Monitoring

```python
# Start Ray Dashboard
ray.init(dashboard_host="0.0.0.0", dashboard_port=8265)

# Visit http://localhost:8265 to view task status
```

## Scenario Examples

### Batch Video Processing

Key steps:
1. **Parallel extraction**: Audio + frame extraction simultaneously
2. **ASR transcription**: Whisper Actor, GPU accelerated
3. **Vectorization**: BGE for text, CLIP for frames
4. **Write to Milvus**: Batch insert

See `multimodal-retrieval:video-search`

### Batch PDF Import

Key steps:
1. **Parallel parsing**: PyMuPDF extracts text and images
2. **VLM captioning**: Image to text (GPU Actor)
3. **Chunking**: RecursiveCharacterTextSplitter
4. **Vectorization**: BGE Actor
5. **Write to Milvus**: Batch insert

See `multimodal-retrieval:multimodal-rag`

### Batch Image Vectorization

Key steps:
1. **Parallel loading**: PIL reads images
2. **CLIP inference**: GPU Actor Pool
3. **Write to Milvus**: Batch insert

See `multimodal-retrieval:image-search`

## Scaling to Cluster

Single-machine code needs no changes, just change startup method:

```bash
# Head node
ray start --head --port=6379

# Worker nodes
ray start --address='HEAD_NODE_IP:6379'
```

```python
# Code connects to cluster
ray.init(address="ray://HEAD_NODE_IP:10001")
```

## Related Resources

- [Ray Official Documentation](https://docs.ray.io/)
- [Ray Data Guide](https://docs.ray.io/en/latest/data/data.html)
- `core:embedding` - Vectorization model selection
- `core:indexing` - Milvus index configuration
