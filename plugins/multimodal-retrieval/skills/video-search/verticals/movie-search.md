# Movie & TV Show Search

> Search movies and TV shows by dialogue, scene description, or visual content.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Search Mode

<ask_user>
How do you want to search videos?

| Mode | Description | Requirements |
|------|-------------|--------------|
| **By dialogue/transcript** | Search spoken content | Speech-to-text (Whisper) |
| **By visual content** | Search scene content | Frame extraction (CLIP) |
| **Both** (recommended) | Search dialogue or visuals | Both components |
</ask_user>

### 2. Video Language (if dialogue search)

<ask_user>
What language are the videos in?

| Language | Notes |
|----------|-------|
| **English** | Best Whisper accuracy |
| **Chinese** | Good accuracy |
| **Other** | Whisper supports 99 languages |
| **Multiple** | Will auto-detect |
</ask_user>

### 3. Text Embedding (for transcripts)

<ask_user>
Choose transcript embedding:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality, fast | Requires API key |
| **Local Model** | Free, offline | Model download needed |
</ask_user>

### 4. Local Text Model (if local)

<ask_user>
Choose text embedding model:

| Model | Size | Notes |
|-------|------|-------|
| `all-MiniLM-L6-v2` | 80MB | Fast, good for prototyping |
| `BAAI/bge-base-en-v1.5` | 440MB | Higher quality (English) |
| `BAAI/bge-m3` | 2.2GB | Multilingual |
</ask_user>

### 5. Data Scale

<ask_user>
How many videos and how long?

- Each video = transcript chunks + keyframes
- Example: 100 × 2-hour movies ≈ 100 × (200 text chunks + 100 frames) = 30K vectors

| Vector Count | Recommended Milvus |
|--------------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 6. Project Setup

<ask_user>
Choose project management:

| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production projects |
| **pip** | Quick prototypes |
</ask_user>

---

## Dependencies

### Full setup (dialogue + visual) + uv
```bash
uv init movie-search
cd movie-search
uv add pymilvus openai-whisper transformers torch Pillow moviepy
```

### With OpenAI text embedding
```bash
uv add openai
```

### With local text embedding
```bash
uv add sentence-transformers
```

---

## End-to-End Implementation

### Step 1: Configure Embedding Models

```python
import torch
from transformers import CLIPProcessor, CLIPModel

# === CLIP for visual search ===
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device).eval()

VISUAL_DIMENSION = 512

def embed_frames(images):
    """Embed images using CLIP."""
    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()

def embed_visual_query(text):
    """Embed text query for visual search."""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()[0]


# === Text embedding for transcripts ===
# Option A: OpenAI
from openai import OpenAI
openai_client = OpenAI()

def embed_text(texts):
    resp = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

TEXT_DIMENSION = 1536

# Option B: Local
# from sentence_transformers import SentenceTransformer
# text_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
# def embed_text(texts):
#     return text_model.encode(texts, normalize_embeddings=True).tolist()
# TEXT_DIMENSION = 768
```

### Step 2: Extract Audio & Transcribe

```python
import whisper
from moviepy.editor import VideoFileClip
import os

# Load Whisper model
whisper_model = whisper.load_model("base")  # or "small", "medium", "large"

def extract_audio(video_path: str, output_path: str = None) -> str:
    """Extract audio from video."""
    if output_path is None:
        output_path = video_path.rsplit(".", 1)[0] + ".mp3"

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path, verbose=False, logger=None)
    video.close()
    return output_path

def transcribe_video(video_path: str) -> list[dict]:
    """Transcribe video and return timestamped segments."""
    audio_path = extract_audio(video_path)

    result = whisper_model.transcribe(audio_path, verbose=False)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })

    # Clean up audio file
    os.remove(audio_path)

    return segments

def chunk_transcript(segments: list[dict], target_duration: int = 30) -> list[dict]:
    """Merge segments into chunks of ~target_duration seconds."""
    chunks = []
    current_chunk = {"start": 0, "end": 0, "text": ""}

    for seg in segments:
        if current_chunk["end"] - current_chunk["start"] < target_duration:
            if current_chunk["text"]:
                current_chunk["text"] += " "
            current_chunk["text"] += seg["text"]
            current_chunk["end"] = seg["end"]
            if current_chunk["start"] == 0:
                current_chunk["start"] = seg["start"]
        else:
            if current_chunk["text"]:
                chunks.append(current_chunk)
            current_chunk = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }

    if current_chunk["text"]:
        chunks.append(current_chunk)

    return chunks
```

### Step 3: Extract Keyframes

```python
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np

def extract_keyframes(video_path: str, interval: int = 30) -> list[dict]:
    """Extract keyframes at regular intervals."""
    video = VideoFileClip(video_path)
    duration = video.duration
    frames = []

    for t in range(0, int(duration), interval):
        frame = video.get_frame(t)
        pil_image = Image.fromarray(frame)
        frames.append({
            "timestamp": t,
            "image": pil_image
        })

    video.close()
    return frames
```

### Step 4: Process Videos

```python
def process_video(video_path: str) -> dict:
    """Process a video: transcribe and extract frames."""
    print(f"Processing: {video_path}")

    # Get video info
    video = VideoFileClip(video_path)
    duration = video.duration
    video.close()

    # Transcribe
    print("  Transcribing...")
    segments = transcribe_video(video_path)
    chunks = chunk_transcript(segments, target_duration=30)

    # Extract keyframes
    print("  Extracting keyframes...")
    frames = extract_keyframes(video_path, interval=30)

    return {
        "path": video_path,
        "duration": duration,
        "transcript_chunks": chunks,
        "keyframes": frames
    }
```

### Step 5: Index into Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient("movies.db")  # Milvus Lite

# Create two collections: one for transcripts, one for frames
client.create_collection(
    collection_name="movie_transcripts",
    dimension=TEXT_DIMENSION,
    auto_id=True
)

client.create_collection(
    collection_name="movie_frames",
    dimension=VISUAL_DIMENSION,
    auto_id=True
)

def index_video(video_data: dict):
    """Index a processed video."""
    video_path = video_data["path"]
    video_name = os.path.basename(video_path)

    # Index transcript chunks
    if video_data["transcript_chunks"]:
        texts = [c["text"] for c in video_data["transcript_chunks"]]
        vectors = embed_text(texts)

        data = [
            {
                "vector": vec,
                "video": video_name,
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"]
            }
            for vec, chunk in zip(vectors, video_data["transcript_chunks"])
        ]
        client.insert(collection_name="movie_transcripts", data=data)

    # Index keyframes
    if video_data["keyframes"]:
        images = [f["image"] for f in video_data["keyframes"]]
        vectors = embed_frames(images)

        data = [
            {
                "vector": vec,
                "video": video_name,
                "timestamp": frame["timestamp"]
            }
            for vec, frame in zip(vectors, video_data["keyframes"])
        ]
        client.insert(collection_name="movie_frames", data=data)

    print(f"Indexed {len(video_data['transcript_chunks'])} transcript chunks, "
          f"{len(video_data['keyframes'])} frames")
```

### Step 6: Search

```python
def search_by_dialogue(query: str, top_k: int = 5):
    """Search by dialogue/transcript content."""
    query_vector = embed_text([query])[0]

    results = client.search(
        collection_name="movie_transcripts",
        data=[query_vector],
        limit=top_k,
        output_fields=["video", "start", "end", "text"]
    )
    return results[0]

def search_by_scene(query: str, top_k: int = 5):
    """Search by visual scene description."""
    query_vector = embed_visual_query(query)

    results = client.search(
        collection_name="movie_frames",
        data=[query_vector],
        limit=top_k,
        output_fields=["video", "timestamp"]
    )
    return results[0]

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def print_dialogue_results(results):
    """Print dialogue search results."""
    for i, hit in enumerate(results, 1):
        e = hit["entity"]
        print(f"\n#{i} [{e['video']}] {format_timestamp(e['start'])} - {format_timestamp(e['end'])}")
        print(f"    Score: {hit['distance']:.3f}")
        print(f"    \"{e['text'][:200]}...\"")

def print_scene_results(results):
    """Print visual search results."""
    for i, hit in enumerate(results, 1):
        e = hit["entity"]
        print(f"\n#{i} [{e['video']}] at {format_timestamp(e['timestamp'])}")
        print(f"    Score: {hit['distance']:.3f}")
```

---

## Run Example

```python
import os

# Process and index videos
video_folder = "./movies"
for file in os.listdir(video_folder):
    if file.endswith((".mp4", ".mkv", ".avi")):
        video_path = os.path.join(video_folder, file)
        video_data = process_video(video_path)
        index_video(video_data)

# Search by dialogue
print("\n=== Searching: 'I'll be back' ===")
print_dialogue_results(search_by_dialogue("I'll be back"))

# Search by scene description
print("\n=== Searching: 'car chase at night' ===")
print_scene_results(search_by_scene("car chase at night in city"))

print("\n=== Searching: 'explosion' ===")
print_scene_results(search_by_scene("big explosion with fire"))
```

---

## Advanced: Jump to Timestamp

```python
from moviepy.editor import VideoFileClip

def play_from_timestamp(video_path: str, timestamp: float, duration: float = 10):
    """Extract and save a clip starting from timestamp."""
    video = VideoFileClip(video_path)
    end_time = min(timestamp + duration, video.duration)
    clip = video.subclip(timestamp, end_time)

    output_path = f"clip_{int(timestamp)}.mp4"
    clip.write_videofile(output_path, verbose=False, logger=None)
    clip.close()
    video.close()

    return output_path
```
