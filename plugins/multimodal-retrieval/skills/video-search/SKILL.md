---
name: video-search
description: "Use when user needs to search video content by text or image. Triggers on: video search, video retrieval, video clips, meeting recordings, tutorial videos, surveillance playback."
---

# Video Search

Semantic search on video content, supporting retrieval by description, subtitles, or visual content.

## Use Cases

- Tutorial video retrieval (find specific knowledge points)
- Meeting recording search (find specific discussions)
- Surveillance playback (find specific scenes)
- Film clip search

## Architecture

```
Video → Frame extraction + ASR → Vectorize → Store
Query → Vector search → Return video clips (timestamps)
```

## Data Processing

Batch video processing is recommended using Ray orchestration (see `core:ray`).

**Key Steps**:

1. **Parallel extraction**: Audio + frame extraction simultaneously (ffmpeg + OpenCV)
2. **ASR transcription**: Whisper, merge in 30-second segments
3. **Vectorization**: Subtitles use BGE, frames use CLIP
4. **Write to Milvus**: Batch insert

**Tool Selection**:

| Step | Tool |
|------|------|
| Audio extraction | ffmpeg |
| Video frame extraction | OpenCV (cv2) |
| Speech recognition | Whisper / FunASR |
| Text vectorization | BGE-large-zh |
| Image vectorization | CLIP |

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("video_path", DataType.VARCHAR, max_length=512)
schema.add_field("content_type", DataType.VARCHAR, max_length=16)  # transcript/frame
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("start_time", DataType.FLOAT)
schema.add_field("end_time", DataType.FLOAT)
schema.add_field("frame_path", DataType.VARCHAR, max_length=512)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

index_params.add_index("embedding", index_type="HNSW", metric_type="COSINE",
                       params={"M": 16, "efConstruction": 256})
```

## Search Implementation

```python
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

class VideoSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.text_model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

    def search(self, query: str, limit: int = 10, search_type: str = "transcript") -> list:
        """Search video clips"""
        embedding = self.text_model.encode(query).tolist()

        filter_expr = ""
        if search_type == "transcript":
            filter_expr = 'content_type == "transcript"'
        elif search_type == "frame":
            filter_expr = 'content_type == "frame"'

        results = self.client.search(
            collection_name="video_search",
            data=[embedding],
            filter=filter_expr,
            limit=limit,
            output_fields=["video_path", "content", "start_time", "end_time"]
        )

        return [{
            "video": hit["entity"]["video_path"],
            "content": hit["entity"]["content"][:100] + "...",
            "start": hit["entity"]["start_time"],
            "end": hit["entity"]["end_time"],
            "score": hit["distance"]
        } for hit in results[0]]

# Usage
search = VideoSearch()
results = search.search("basic concepts of machine learning")
for r in results:
    print(f"[{r['start']:.0f}s - {r['end']:.0f}s] {r['content']}")
```

## Processing Strategies

| Scenario | Strategy |
|----------|----------|
| Tutorial videos | Subtitle-focused + sparse keyframes (30s) |
| Surveillance videos | Keyframe-focused (5s), no audio |
| Meeting recordings | Subtitle-focused + speaker identification |
| Film content | Subtitles + dense keyframes (10s) |

## ASR Model Selection

| Model | Features | Use Case |
|-------|----------|----------|
| Whisper | Multilingual, open-source | General |
| FunASR | Chinese optimized | Chinese |
| Azure Speech | API, high quality | Production |

## Vertical Applications

See `verticals/` directory for detailed guides:
- `education.md` - Tutorial videos
- `meeting.md` - Meeting recordings
- `surveillance.md` - Surveillance videos

## Related Tools

- Data processing orchestration: `core:ray`
- Vectorization: `core:embedding`
- Text-to-image search: `multimodal-retrieval:text-to-image-search`
