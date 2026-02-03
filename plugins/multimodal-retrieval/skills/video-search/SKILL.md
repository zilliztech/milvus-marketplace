---
name: video-search
description: "Use when user needs to search video content by text or image. Triggers on: video search, video retrieval, video clips, meeting recordings, tutorial videos, surveillance playback, find moment in video."
---

# Video Search

Semantic search on video content — find specific moments by describing what you're looking for.

## When to Activate

Activate this skill when:
- User wants to **search within video content** by description
- User mentions "find in video", "video search", "meeting recordings"
- User has **tutorial videos, meetings, or surveillance footage** to search
- User needs to find **specific scenes or discussions** in long videos

**Do NOT activate** when:
- User only needs image search → use `image-search`
- User wants to search images by text → use `text-to-image-search`
- User has static documents with images → use `multimodal-rag`

## Interactive Flow

### Step 1: Understand Video Type

"What type of videos are you searching?"

A) **Speech-heavy** (tutorials, meetings, lectures)
   - Primary content is spoken words
   - ASR (speech-to-text) is key

B) **Visual-heavy** (surveillance, sports, vlogs)
   - Actions and scenes matter more than speech
   - Keyframe extraction is key

C) **Mixed** (documentaries, how-to videos)
   - Both speech and visuals important
   - Need both approaches

Which describes your videos? (A/B/C)

### Step 2: Determine Search Granularity

"How precise should search results be?"

| Granularity | Segment Length | Use Case |
|-------------|----------------|----------|
| **Coarse** | 5-10 minutes | "Find the meeting about budget" |
| **Medium** | 30-60 seconds | "Find where they discuss pricing" |
| **Fine** | 5-15 seconds | "Find exactly when John mentioned the deadline" |

### Step 3: Confirm Configuration

"Based on your requirements:

- **Processing**: ASR (Whisper) + Keyframes every 30s
- **Segment size**: 30 seconds
- **Embedding**: BGE for text, CLIP for frames

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Video as Searchable Book

Think of video processing as **converting a video into a searchable book**:
- Each chapter = video segment (30-60 seconds)
- Each chapter has text (transcript) + illustrations (keyframes)
- Search finds the right "chapter"

```
┌─────────────────────────────────────────────────────────┐
│                    Video Search Pipeline                 │
│                                                          │
│  Original Video (2 hours)                                │
│       │                                                  │
│       ├────────────────────────────────────┐            │
│       │                                    │            │
│       ▼                                    ▼            │
│  ┌──────────────┐                  ┌──────────────┐    │
│  │ Audio Track  │                  │ Video Track  │    │
│  └──────┬───────┘                  └──────┬───────┘    │
│         │                                  │            │
│         ▼                                  ▼            │
│  ┌──────────────┐                  ┌──────────────┐    │
│  │    Whisper   │                  │   Keyframe   │    │
│  │     ASR      │                  │  Extraction  │    │
│  └──────┬───────┘                  └──────┬───────┘    │
│         │                                  │            │
│         ▼                                  ▼            │
│  [Transcript segments]             [Keyframe images]    │
│  "At 0:30, John said..."           [img1] [img2] ...   │
│         │                                  │            │
│         ▼                                  ▼            │
│  ┌──────────────┐                  ┌──────────────┐    │
│  │     BGE      │                  │    CLIP      │    │
│  │   Encoder    │                  │   Encoder    │    │
│  └──────┬───────┘                  └──────┬───────┘    │
│         │                                  │            │
│         └────────────┬─────────────────────┘            │
│                      ▼                                   │
│              ┌──────────────┐                           │
│              │    Milvus    │                           │
│              │   Storage    │                           │
│              └──────────────┘                           │
└─────────────────────────────────────────────────────────┘
```

### Two Search Approaches

| Approach | What it Searches | Best For |
|----------|------------------|----------|
| **Transcript** | Spoken words | "What did they say about X?" |
| **Keyframes** | Visual content | "Find the scene with Y" |

Both can be combined for comprehensive search.

## Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

class VideoSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.text_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.collection_name = "video_search"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("video_path", DataType.VARCHAR, max_length=512)
        schema.add_field("content_type", DataType.VARCHAR, max_length=16)  # transcript/frame
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("start_time", DataType.FLOAT)
        schema.add_field("end_time", DataType.FLOAT)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

        index_params = self.client.prepare_index_params()
        index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def search(self, query: str, limit: int = 10, search_type: str = "all") -> list:
        """Search video clips
        search_type: "all" | "transcript" | "frame"
        """
        embedding = self.text_model.encode(query).tolist()

        filter_expr = ""
        if search_type == "transcript":
            filter_expr = 'content_type == "transcript"'
        elif search_type == "frame":
            filter_expr = 'content_type == "frame"'

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            filter=filter_expr if filter_expr else None,
            limit=limit,
            output_fields=["video_path", "content", "start_time", "end_time", "content_type"]
        )

        return [{
            "video": hit["entity"]["video_path"],
            "type": hit["entity"]["content_type"],
            "content": hit["entity"]["content"][:200] + "..." if len(hit["entity"]["content"]) > 200 else hit["entity"]["content"],
            "start": hit["entity"]["start_time"],
            "end": hit["entity"]["end_time"],
            "score": hit["distance"]
        } for hit in results[0]]

    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

# Usage
search = VideoSearch()
results = search.search("how to configure the database connection")

for r in results:
    start = search.format_timestamp(r['start'])
    end = search.format_timestamp(r['end'])
    print(f"[{start} - {end}] ({r['type']})")
    print(f"  {r['content']}")
    print(f"  Score: {r['score']:.3f}")
    print()
```

## Video Processing Pipeline

### Audio Processing (Transcription)

```python
import whisper
import subprocess

def extract_audio(video_path: str, audio_path: str):
    """Extract audio track from video."""
    subprocess.run([
        'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1', audio_path, '-y'
    ], check=True)

def transcribe_audio(audio_path: str, segment_length: int = 30):
    """Transcribe and segment audio."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)

    segments = []
    current_segment = {"text": "", "start": 0, "end": 0}

    for segment in result["segments"]:
        if segment["end"] - current_segment["start"] > segment_length:
            if current_segment["text"]:
                segments.append(current_segment)
            current_segment = {
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"]
            }
        else:
            current_segment["text"] += " " + segment["text"]
            current_segment["end"] = segment["end"]

    if current_segment["text"]:
        segments.append(current_segment)

    return segments
```

### Frame Extraction

```python
import cv2

def extract_keyframes(video_path: str, interval_seconds: int = 30):
    """Extract keyframes at regular intervals."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frames.append({
                "frame": frame,
                "timestamp": timestamp
            })

        frame_count += 1

    cap.release()
    return frames
```

## Processing Strategy by Video Type

| Video Type | ASR | Keyframes | Segment Length |
|------------|-----|-----------|----------------|
| **Tutorials** | ✅ Primary | Every 30s | 30s |
| **Meetings** | ✅ Primary | Every 60s | 60s |
| **Surveillance** | ❌ Skip | Every 5s | 10s |
| **Movies/Shows** | ✅ Subtitles | Every 10s | 30s |
| **Sports** | ⚠️ Commentary | Every 3s | 15s |

## Common Pitfalls

### ❌ Pitfall 1: Processing Full Resolution

**Problem**: Processing takes forever on 4K videos

**Why**: Video processing is compute-intensive

**Fix**: Downscale for processing
```python
# Extract at 720p for processing
subprocess.run([
    'ffmpeg', '-i', input_path, '-vf', 'scale=-1:720',
    output_path, '-y'
])
```

### ❌ Pitfall 2: Too Many Keyframes

**Problem**: Storage explodes with frequent keyframes

**Why**: Every 5 seconds on a 2-hour video = 1440 frames

**Fix**: Use scene detection or longer intervals
```python
# Scene-change detection
from scenedetect import detect, ContentDetector
scenes = detect(video_path, ContentDetector())
```

### ❌ Pitfall 3: Ignoring ASR Errors

**Problem**: Transcription errors make search miss results

**Why**: Speech recognition isn't perfect

**Fix**: Store both raw and corrected transcripts, or use phonetic search

### ❌ Pitfall 4: No Timestamp Indexing

**Problem**: Can't quickly seek to result in video player

**Why**: Only stored content, not timestamps

**Fix**: Always store start_time and end_time for each segment

## When to Level Up

| Need | Upgrade To |
|------|------------|
| Search by uploading image | Combine with `image-search` |
| Q&A on video content | Add RAG layer |
| Real-time streaming | Consider specialized tools |
| Speaker identification | Add speaker diarization |

## References

- Frame extraction tools: `references/frame-sampling.md`
- ASR models: Whisper, FunASR, Azure Speech
- Batch processing: `core:ray`
