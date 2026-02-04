# Educational Video Search

> Search lecture videos by knowledge point, topic, or spoken content.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Search Mode

<ask_user>
How do you want to search educational videos?

| Mode | Description |
|------|-------------|
| **By transcript** | Search spoken narration/lecture content |
| **By slides/frames** | Search PPT/whiteboard content (OCR) |
| **Both** (recommended) | Full content search |
</ask_user>

### 2. Video Language

<ask_user>
What language are the videos in?

| Language | ASR Model |
|----------|-----------|
| **English** | Whisper (best accuracy) |
| **Chinese** | Whisper or FunASR |
| **Multilingual** | Whisper (auto-detect) |
</ask_user>

### 3. Text Embedding

<ask_user>
Choose transcript embedding:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download |
</ask_user>

### 4. Local Model (if local)

<ask_user>
| Model | Size | Notes |
|-------|------|-------|
| `all-MiniLM-L6-v2` | 80MB | Fast |
| `BAAI/bge-base-en-v1.5` | 440MB | English |
| `BAAI/bge-base-zh-v1.5` | 400MB | Chinese |
</ask_user>

### 5. Data Scale

<ask_user>
How many videos do you have?

- Each video = transcript chunks + keyframes
- Example: 100 lectures × 1 hour ≈ 100 × (60 chunks + 30 frames) = 9K vectors

| Vector Count | Recommended Milvus |
|--------------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 6. Project Setup

<ask_user>
| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production projects |
| **pip** | Quick prototypes |
</ask_user>

---

## Dependencies

```bash
uv init edu-video-search
cd edu-video-search
uv add pymilvus openai-whisper sentence-transformers opencv-python pytesseract
```

---

## End-to-End Implementation

### Step 1: Configure Models

```python
import whisper
from sentence_transformers import SentenceTransformer

# ASR model
asr_model = whisper.load_model("base")  # or "small", "medium", "large"

# Text embedding
text_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

DIMENSION = 768

def embed(texts: list[str]) -> list[list[float]]:
    return text_model.encode(texts, normalize_embeddings=True).tolist()
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("education.db")

schema = client.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("content_type", DataType.VARCHAR, max_length=16)  # transcript/frame
schema.add_field("start_time", DataType.FLOAT)
schema.add_field("end_time", DataType.FLOAT)
schema.add_field("video_id", DataType.VARCHAR, max_length=64)
schema.add_field("video_title", DataType.VARCHAR, max_length=256)
schema.add_field("course", DataType.VARCHAR, max_length=128)
schema.add_field("chapter", DataType.VARCHAR, max_length=128)
schema.add_field("keywords", DataType.VARCHAR, max_length=256)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("edu_videos", schema=schema, index_params=index_params)
```

### Step 3: Extract & Transcribe

```python
import os
import cv2

def transcribe_video(video_path: str) -> list[dict]:
    """Extract audio and transcribe."""
    audio_path = video_path.rsplit(".", 1)[0] + ".wav"
    os.system(f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y 2>/dev/null')

    result = asr_model.transcribe(audio_path, language="en")
    os.remove(audio_path)

    return [{"text": s["text"], "start": s["start"], "end": s["end"]}
            for s in result["segments"]]

def merge_segments(segments: list[dict], target_duration: int = 30) -> list[dict]:
    """Merge into ~30s chunks."""
    merged = []
    current = {"text": "", "start": 0, "end": 0}

    for seg in segments:
        if not current["text"]:
            current["start"] = seg["start"]

        current["text"] += " " + seg["text"]
        current["end"] = seg["end"]

        if current["end"] - current["start"] >= target_duration:
            merged.append(current)
            current = {"text": "", "start": 0, "end": 0}

    if current["text"]:
        merged.append(current)

    return merged

def extract_keywords(text: str) -> str:
    """Extract keywords using simple frequency."""
    from collections import Counter
    words = [w.lower() for w in text.split() if len(w) > 4]
    top = Counter(words).most_common(5)
    return ",".join(w for w, _ in top)
```

### Step 4: Extract Keyframes with OCR

```python
import pytesseract
from PIL import Image

def extract_frames_with_ocr(video_path: str, interval: int = 60) -> list[dict]:
    """Extract frames and OCR text (for PPT/whiteboard)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            timestamp = count / fps
            # OCR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            try:
                text = pytesseract.image_to_string(pil_img, lang='eng')
                if text.strip():
                    frames.append({"text": text.strip(), "timestamp": timestamp})
            except:
                pass

        count += 1

    cap.release()
    return frames
```

### Step 5: Process & Index Video

```python
def process_video(video_path: str, video_id: str, course: str, chapter: str = ""):
    """Process and index an educational video."""
    video_title = os.path.basename(video_path)
    data = []

    # Transcribe
    print("Transcribing...")
    segments = transcribe_video(video_path)
    chunks = merge_segments(segments, target_duration=30)

    for i, chunk in enumerate(chunks):
        keywords = extract_keywords(chunk["text"])
        embedding = embed([chunk["text"]])[0]

        data.append({
            "embedding": embedding,
            "content": chunk["text"][:5000],
            "content_type": "transcript",
            "start_time": chunk["start"],
            "end_time": chunk["end"],
            "video_id": video_id,
            "video_title": video_title,
            "course": course,
            "chapter": chapter,
            "keywords": keywords
        })

    # Extract frames with OCR
    print("Extracting frames...")
    frames = extract_frames_with_ocr(video_path, interval=60)

    for frame in frames:
        if len(frame["text"]) > 50:  # Skip short OCR results
            embedding = embed([frame["text"]])[0]
            data.append({
                "embedding": embedding,
                "content": frame["text"][:5000],
                "content_type": "frame",
                "start_time": frame["timestamp"],
                "end_time": frame["timestamp"] + 60,
                "video_id": video_id,
                "video_title": video_title,
                "course": course,
                "chapter": chapter,
                "keywords": ""
            })

    client.insert(collection_name="edu_videos", data=data)
    print(f"Indexed {len(data)} segments")
```

### Step 6: Search

```python
def search_videos(query: str, top_k: int = 10, course: str = None):
    """Search educational videos."""
    query_embedding = embed([query])[0]

    filters = ['content_type == "transcript"']
    if course:
        filters.append(f'course == "{course}"')

    results = client.search(
        collection_name="edu_videos",
        data=[query_embedding],
        filter=" and ".join(filters),
        limit=top_k,
        output_fields=["content", "start_time", "end_time", "video_id",
                      "video_title", "course", "chapter", "keywords"]
    )
    return results[0]

def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def print_results(results):
    for i, r in enumerate(results, 1):
        e = r["entity"]
        print(f"\n#{i} [{e['course']}] {e['video_title']}")
        print(f"    Chapter: {e['chapter']}")
        print(f"    Time: {format_time(e['start_time'])} - {format_time(e['end_time'])}")
        print(f"    Keywords: {e['keywords']}")
        print(f"    Score: {r['distance']:.3f}")
        print(f"    Content: {e['content'][:150]}...")
```

---

## Run Example

```python
# Index lecture videos
process_video(
    video_path="machine_learning_01.mp4",
    video_id="ml_001",
    course="Machine Learning",
    chapter="Chapter 1: Introduction"
)

# Search
results = search_videos("gradient descent algorithm")
print_results(results)

results = search_videos("backpropagation neural network", course="Machine Learning")
print_results(results)
```
