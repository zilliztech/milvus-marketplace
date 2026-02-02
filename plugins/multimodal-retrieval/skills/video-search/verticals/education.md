# Educational Video Search

## Use Cases

- Online course knowledge point search
- Training video retrieval
- Lecture/public course search
- MOOC platform video retrieval

## Data Characteristics

- Has subtitles/narration (audio-driven)
- Dense knowledge points
- PPT/whiteboard frames
- Requires positioning by knowledge points

## Recommended Configuration

| Config | Recommended Value | Description |
|--------|------------------|-------------|
| ASR Model | Whisper large | High accuracy |
| | FunASR | Chinese optimized |
| Text Embedding | `BAAI/bge-large-en-v1.5` | Subtitles/narration |
| Frame Interval | 30-60 seconds | Educational content changes slowly |
| Segment Length | 30-60 seconds | One knowledge point |

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

# Content
schema.add_field("content_type", DataType.VARCHAR, max_length=16)   # transcript/frame
schema.add_field("content", DataType.VARCHAR, max_length=65535)     # Subtitle text or frame description
schema.add_field("frame_path", DataType.VARCHAR, max_length=512)

# Time positioning
schema.add_field("start_time", DataType.FLOAT)                      # Start time (seconds)
schema.add_field("end_time", DataType.FLOAT)                        # End time

# Course information
schema.add_field("video_id", DataType.VARCHAR, max_length=64)
schema.add_field("video_title", DataType.VARCHAR, max_length=256)
schema.add_field("course", DataType.VARCHAR, max_length=128)        # Course name
schema.add_field("teacher", DataType.VARCHAR, max_length=64)        # Instructor
schema.add_field("chapter", DataType.VARCHAR, max_length=128)       # Chapter
schema.add_field("keywords", DataType.VARCHAR, max_length=256)      # Knowledge point keywords
```

## Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
import whisper
import cv2
import os

class EducationVideoSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.asr = whisper.load_model("large")
        self._init_collection()

    def _extract_transcript(self, video_path: str) -> list:
        """Extract subtitles"""
        # Extract audio
        audio_path = video_path.replace(video_path.split('.')[-1], 'wav')
        os.system(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path} -y")

        # ASR transcription
        result = self.asr.transcribe(audio_path, language="en")

        segments = []
        for seg in result["segments"]:
            segments.append({
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"]
            })

        return segments

    def _merge_segments(self, segments: list, target_duration: int = 30) -> list:
        """Merge subtitle segments (by time window)"""
        merged = []
        current_text = ""
        current_start = 0

        for seg in segments:
            if not current_text:
                current_start = seg["start"]

            current_text += seg["text"] + " "

            if seg["end"] - current_start >= target_duration:
                merged.append({
                    "text": current_text.strip(),
                    "start": current_start,
                    "end": seg["end"]
                })
                current_text = ""

        if current_text:
            merged.append({
                "text": current_text.strip(),
                "start": current_start,
                "end": segments[-1]["end"]
            })

        return merged

    def _extract_keywords(self, text: str) -> str:
        """Extract knowledge point keywords (can use LLM or TF-IDF)"""
        # Simplified version: use keyword extraction
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Simple keyword extraction - in production use better NLP tools
        words = text.lower().split()
        # Return top frequent meaningful words
        from collections import Counter
        word_counts = Counter(w for w in words if len(w) > 4)
        keywords = [w for w, _ in word_counts.most_common(5)]
        return ",".join(keywords)

    def add_video(self, video_path: str, video_id: str, course: str,
                  teacher: str, chapter: str = "", output_dir: str = "./edu_video_data"):
        """Process educational video"""
        os.makedirs(output_dir, exist_ok=True)
        video_title = os.path.basename(video_path)
        data = []

        # 1. Extract and process subtitles
        print("Extracting subtitles...")
        raw_segments = self._extract_transcript(video_path)
        merged_segments = self._merge_segments(raw_segments, target_duration=30)

        for i, seg in enumerate(merged_segments):
            keywords = self._extract_keywords(seg["text"])
            embedding = self.model.encode(seg["text"]).tolist()

            data.append({
                "id": f"{video_id}_transcript_{i}",
                "embedding": embedding,
                "content_type": "transcript",
                "content": seg["text"],
                "frame_path": "",
                "start_time": seg["start"],
                "end_time": seg["end"],
                "video_id": video_id,
                "video_title": video_title,
                "course": course,
                "teacher": teacher,
                "chapter": chapter,
                "keywords": keywords
            })

        # 2. Extract key frames (longer intervals for educational videos)
        print("Extracting key frames...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 60)  # One frame every 60 seconds

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frame_path = os.path.join(output_dir, f"{video_id}_frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)

                # For educational videos, frames mainly locate PPT/whiteboard
                # Can use OCR to extract text
                frame_text = self._ocr_frame(frame_path)

                if frame_text:
                    embedding = self.model.encode(frame_text).tolist()
                    data.append({
                        "id": f"{video_id}_frame_{saved_count}",
                        "embedding": embedding,
                        "content_type": "frame",
                        "content": frame_text,
                        "frame_path": frame_path,
                        "start_time": timestamp,
                        "end_time": timestamp + 60,
                        "video_id": video_id,
                        "video_title": video_title,
                        "course": course,
                        "teacher": teacher,
                        "chapter": chapter,
                        "keywords": ""
                    })

                saved_count += 1

            frame_count += 1

        cap.release()

        # Batch insert
        if data:
            self.client.insert(collection_name="education_videos", data=data)

        return {
            "video_id": video_id,
            "transcript_segments": len([d for d in data if d["content_type"] == "transcript"]),
            "frames": len([d for d in data if d["content_type"] == "frame"])
        }

    def _ocr_frame(self, frame_path: str) -> str:
        """OCR extract text from frame (PPT/whiteboard)"""
        try:
            import pytesseract
            from PIL import Image
            image = Image.open(frame_path)
            text = pytesseract.image_to_string(image, lang='eng')
            return text.strip()
        except:
            return ""

    def search(self, query: str, course: str = None, teacher: str = None,
               limit: int = 10) -> list:
        """Search knowledge points"""
        embedding = self.model.encode(query).tolist()

        # Build filter conditions
        filters = ['content_type == "transcript"']  # Mainly search subtitles
        if course:
            filters.append(f'course == "{course}"')
        if teacher:
            filters.append(f'teacher == "{teacher}"')

        filter_expr = ' and '.join(filters)

        results = self.client.search(
            collection_name="education_videos",
            data=[embedding],
            filter=filter_expr,
            limit=limit,
            output_fields=["content", "start_time", "end_time", "video_id",
                          "video_title", "course", "teacher", "chapter", "keywords"]
        )

        return [{
            "video_id": r["entity"]["video_id"],
            "video_title": r["entity"]["video_title"],
            "course": r["entity"]["course"],
            "teacher": r["entity"]["teacher"],
            "chapter": r["entity"]["chapter"],
            "content": r["entity"]["content"][:200] + "...",
            "keywords": r["entity"]["keywords"],
            "start_time": r["entity"]["start_time"],
            "end_time": r["entity"]["end_time"],
            "url": self._generate_url(r["entity"]["video_id"], r["entity"]["start_time"]),
            "score": r["distance"]
        } for r in results[0]]

    def _generate_url(self, video_id: str, start_time: float) -> str:
        """Generate playback link with timestamp"""
        # Adjust based on actual platform
        return f"/play/{video_id}?t={int(start_time)}"

    def search_by_chapter(self, course: str, chapter: str) -> list:
        """Browse by chapter"""
        results = self.client.query(
            collection_name="education_videos",
            filter=f'course == "{course}" and chapter == "{chapter}" and content_type == "transcript"',
            output_fields=["content", "start_time", "end_time", "keywords"],
            limit=100
        )

        # Sort by time
        results.sort(key=lambda x: x["start_time"])
        return results
```

## Examples

```python
search = EducationVideoSearch()

# Add course video
stats = search.add_video(
    video_path="machine_learning_01.mp4",
    video_id="ml_001",
    course="Machine Learning",
    teacher="Prof. Smith",
    chapter="Chapter 1 Introduction"
)

# Search knowledge points
results = search.search(
    "What is gradient descent algorithm",
    course="Machine Learning"
)

print("Search results:")
for r in results:
    print(f"\n[{r['course']} - {r['chapter']}]")
    print(f"Instructor: {r['teacher']}")
    print(f"Time: {r['start_time']:.0f}s - {r['end_time']:.0f}s")
    print(f"Keywords: {r['keywords']}")
    print(f"Content: {r['content']}")
    print(f"Play link: {r['url']}")
```

## Advanced Features

### Knowledge Graph

```python
def build_knowledge_graph(self, course: str):
    """Build course knowledge graph"""
    # Get all segment keywords
    results = self.client.query(
        collection_name="education_videos",
        filter=f'course == "{course}" and content_type == "transcript"',
        output_fields=["keywords", "chapter"],
        limit=1000
    )

    # Statistics: keyword frequency and chapter relationships
    keyword_stats = {}
    chapter_keywords = {}

    for r in results:
        chapter = r["chapter"]
        keywords = r["keywords"].split(",")

        for kw in keywords:
            if kw not in keyword_stats:
                keyword_stats[kw] = {"count": 0, "chapters": set()}
            keyword_stats[kw]["count"] += 1
            keyword_stats[kw]["chapters"].add(chapter)

    return keyword_stats
```
