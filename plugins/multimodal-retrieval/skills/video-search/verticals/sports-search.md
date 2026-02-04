# Sports Video Search

> Search sports videos by action, event, or scene description.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Search Mode

<ask_user>
How do you want to search sports videos?

| Mode | Description |
|------|-------------|
| **By commentary** | Search spoken commentary/analysis |
| **By visual action** | Search by scene description (goal, foul, etc.) |
| **Both** (recommended) | Search commentary or visuals |
</ask_user>

### 2. Sports Type

<ask_user>
What type of sports content?

| Type | Notes |
|------|-------|
| **Football/Soccer** | Goals, fouls, passes |
| **Basketball** | Dunks, 3-pointers, blocks |
| **General sports** | Any sports content |
| **eSports** | Gaming content |
</ask_user>

### 3. Video Language (if commentary search)

<ask_user>
What language is the commentary in?

| Language | Notes |
|----------|-------|
| **English** | Best Whisper accuracy |
| **Chinese** | Good accuracy |
| **Other** | Whisper supports 99 languages |
</ask_user>

### 4. Text Embedding (for transcripts)

<ask_user>
Choose transcript embedding:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download |
</ask_user>

### 5. Data Scale

<ask_user>
How many sports videos do you have?

- Each video = commentary chunks + keyframes
- Example: 100 matches × 90 min = 100 × (300 text + 150 frames) = 45K vectors

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

### uv
```bash
uv init sports-search
cd sports-search
uv add pymilvus openai-whisper transformers torch Pillow moviepy
uv add sentence-transformers  # or openai for text embedding
```

---

## End-to-End Implementation

### Step 1: Configure Models

```python
import torch
from transformers import CLIPProcessor, CLIPModel
import whisper

# CLIP for visual search
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device).eval()

VISUAL_DIMENSION = 512

def embed_frames(images):
    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()

def embed_visual_query(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()[0]

# Whisper for speech-to-text
whisper_model = whisper.load_model("base")

# Text embedding
from sentence_transformers import SentenceTransformer
text_model = SentenceTransformer("all-MiniLM-L6-v2")
TEXT_DIMENSION = 384

def embed_text(texts):
    return text_model.encode(texts, normalize_embeddings=True).tolist()
```

### Step 2: Extract Commentary

```python
from moviepy.editor import VideoFileClip
import os

def extract_and_transcribe(video_path: str) -> list[dict]:
    """Extract audio and transcribe commentary."""
    # Extract audio
    video = VideoFileClip(video_path)
    audio_path = video_path.rsplit(".", 1)[0] + ".mp3"
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()

    # Transcribe
    result = whisper_model.transcribe(audio_path, verbose=False)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })

    os.remove(audio_path)
    return segments

def chunk_commentary(segments: list[dict], target_duration: int = 20) -> list[dict]:
    """Merge segments into chunks."""
    chunks = []
    current = {"start": 0, "end": 0, "text": ""}

    for seg in segments:
        if current["end"] - current["start"] < target_duration:
            current["text"] += " " + seg["text"]
            current["end"] = seg["end"]
            if not current["start"]:
                current["start"] = seg["start"]
        else:
            if current["text"].strip():
                chunks.append(current)
            current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}

    if current["text"].strip():
        chunks.append(current)

    return chunks
```

### Step 3: Extract Action Frames

```python
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np

def extract_action_frames(video_path: str, interval: int = 20) -> list[dict]:
    """Extract frames at regular intervals."""
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

def detect_highlight_moments(video_path: str) -> list[float]:
    """Detect potential highlight moments based on audio energy."""
    import librosa

    video = VideoFileClip(video_path)
    audio_path = video_path.rsplit(".", 1)[0] + "_temp.mp3"
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()

    y, sr = librosa.load(audio_path, sr=22050)
    os.remove(audio_path)

    # Calculate RMS energy
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)

    # Find peaks (potential exciting moments)
    threshold = np.percentile(rms, 90)
    highlight_times = times[rms > threshold]

    # Merge nearby timestamps
    merged = []
    for t in highlight_times:
        if not merged or t - merged[-1] > 30:  # 30s apart
            merged.append(t)

    return merged[:20]  # Top 20 highlights
```

### Step 4: Process Sports Video

```python
def process_sports_video(video_path: str) -> dict:
    """Process a sports video for searching."""
    print(f"Processing: {video_path}")
    video_name = os.path.basename(video_path)

    # Transcribe commentary
    print("  Transcribing commentary...")
    segments = extract_and_transcribe(video_path)
    chunks = chunk_commentary(segments, target_duration=20)

    # Extract frames
    print("  Extracting frames...")
    frames = extract_action_frames(video_path, interval=20)

    # Detect highlights (optional)
    print("  Detecting highlights...")
    try:
        highlights = detect_highlight_moments(video_path)
    except:
        highlights = []

    return {
        "path": video_path,
        "name": video_name,
        "commentary_chunks": chunks,
        "frames": frames,
        "highlight_times": highlights
    }
```

### Step 5: Index into Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient("sports.db")

# Create collections
client.create_collection(
    collection_name="sports_commentary",
    dimension=TEXT_DIMENSION,
    auto_id=True
)

client.create_collection(
    collection_name="sports_frames",
    dimension=VISUAL_DIMENSION,
    auto_id=True
)

def index_sports_video(video_data: dict):
    """Index a processed sports video."""
    video_name = video_data["name"]

    # Index commentary
    if video_data["commentary_chunks"]:
        texts = [c["text"] for c in video_data["commentary_chunks"]]
        vectors = embed_text(texts)

        data = [
            {
                "vector": vec,
                "video": video_name,
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"]
            }
            for vec, chunk in zip(vectors, video_data["commentary_chunks"])
        ]
        client.insert(collection_name="sports_commentary", data=data)

    # Index frames
    if video_data["frames"]:
        images = [f["image"] for f in video_data["frames"]]
        vectors = embed_frames(images)

        # Mark highlight frames
        highlight_set = set(int(t) for t in video_data.get("highlight_times", []))

        data = [
            {
                "vector": vec,
                "video": video_name,
                "timestamp": frame["timestamp"],
                "is_highlight": int(frame["timestamp"]) in highlight_set
            }
            for vec, frame in zip(vectors, video_data["frames"])
        ]
        client.insert(collection_name="sports_frames", data=data)

    print(f"Indexed {len(video_data['commentary_chunks'])} commentary chunks, "
          f"{len(video_data['frames'])} frames")
```

### Step 6: Search

```python
def search_commentary(query: str, top_k: int = 5):
    """Search by commentary content."""
    query_vector = embed_text([query])[0]

    results = client.search(
        collection_name="sports_commentary",
        data=[query_vector],
        limit=top_k,
        output_fields=["video", "start", "end", "text"]
    )
    return results[0]

def search_visual(query: str, top_k: int = 5, highlights_only: bool = False):
    """Search by visual scene description."""
    query_vector = embed_visual_query(query)

    filter_expr = "is_highlight == true" if highlights_only else None

    results = client.search(
        collection_name="sports_frames",
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=["video", "timestamp", "is_highlight"]
    )
    return results[0]

def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def print_commentary_results(results):
    for i, hit in enumerate(results, 1):
        e = hit["entity"]
        print(f"\n#{i} [{e['video']}] {format_time(e['start'])} - {format_time(e['end'])}")
        print(f"    Score: {hit['distance']:.3f}")
        print(f"    \"{e['text'][:150]}...\"")

def print_visual_results(results):
    for i, hit in enumerate(results, 1):
        e = hit["entity"]
        hl = "⭐ HIGHLIGHT" if e.get("is_highlight") else ""
        print(f"\n#{i} [{e['video']}] at {format_time(e['timestamp'])} {hl}")
        print(f"    Score: {hit['distance']:.3f}")
```

---

## Run Example

```python
# Process and index sports videos
video_folder = "./sports_videos"
for file in os.listdir(video_folder):
    if file.endswith((".mp4", ".mkv")):
        video_path = os.path.join(video_folder, file)
        video_data = process_sports_video(video_path)
        index_sports_video(video_data)

# Search by commentary
print("\n=== 'goal scored' ===")
print_commentary_results(search_commentary("goal scored"))

print("\n=== 'penalty kick' ===")
print_commentary_results(search_commentary("penalty kick"))

# Search by visual content
print("\n=== 'player celebrating' ===")
print_visual_results(search_visual("player celebrating with arms up"))

print("\n=== 'goalkeeper diving' ===")
print_visual_results(search_visual("goalkeeper diving to save"))

# Search highlights only
print("\n=== Highlights: 'crowd cheering' ===")
print_visual_results(search_visual("excited crowd cheering", highlights_only=True))
```

---

## Sports-Specific Queries

```python
# Common sports search queries
SPORTS_QUERIES = {
    "football": [
        "goal scored",
        "penalty kick",
        "red card",
        "offside",
        "free kick",
        "corner kick",
        "header goal",
        "goalkeeper save"
    ],
    "basketball": [
        "slam dunk",
        "three pointer",
        "blocked shot",
        "fast break",
        "alley oop",
        "free throw"
    ]
}
```
