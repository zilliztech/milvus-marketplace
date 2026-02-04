# Similar Music Search

> Find similar songs by audio features, melody, or style.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Search Mode

<ask_user>
How do you want to search for similar music?

| Mode | Description |
|------|-------------|
| **Audio features** | Search by melody, rhythm, timbre |
| **Lyrics** | Search by lyric content (requires lyrics text) |
| **Both** | Combine audio + lyrics similarity |
</ask_user>

### 2. Audio Embedding Model

<ask_user>
Choose an audio embedding approach:

| Model | Notes |
|-------|-------|
| `laion/clap-htsat-unfused` (recommended) | CLAP model, supports text-to-audio search |
| `openai/whisper-base` + text embedding | Transcribe vocals, then text search |
| `librosa` features + custom model | Traditional audio features (MFCC, etc.) |
</ask_user>

### 3. Data Source

<ask_user>
Where are your music files?

| Source | Notes |
|--------|-------|
| **Local folder** | Your own music collection (MP3, WAV, FLAC) |
| **Free Music Archive (FMA)** | Public domain music dataset |
| **Spotify previews** | 30s clips via API |
</ask_user>

### 4. Data Scale

<ask_user>
How many songs do you have?

- Each song = 1-2 vectors (audio + optional lyrics)

| Song Count | Recommended Milvus |
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

### CLAP + uv (recommended)
```bash
uv init music-search
cd music-search
uv add pymilvus transformers torch librosa soundfile
```

### pip
```bash
pip install pymilvus transformers torch librosa soundfile
```

---

## End-to-End Implementation

### Step 1: Configure CLAP Model

```python
import torch
from transformers import ClapProcessor, ClapModel
import librosa
import numpy as np

# Load CLAP model
model_name = "laion/clap-htsat-unfused"
processor = ClapProcessor.from_pretrained(model_name)
model = ClapModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

DIMENSION = 512

def embed_audio(audio_paths: list[str]) -> list[list[float]]:
    """Embed audio files using CLAP."""
    embeddings = []

    for path in audio_paths:
        try:
            # Load audio (resample to 48kHz for CLAP)
            audio, sr = librosa.load(path, sr=48000, mono=True)

            # Take first 10 seconds if too long
            max_len = 10 * sr
            if len(audio) > max_len:
                audio = audio[:max_len]

            inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                features = model.get_audio_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            embeddings.append(features.cpu().numpy().flatten().tolist())
        except Exception as e:
            print(f"Error processing {path}: {e}")
            embeddings.append(None)

    return embeddings

def embed_text(texts: list[str]) -> list[list[float]]:
    """Embed text descriptions using CLAP."""
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().tolist()
```

### Step 2: Load Music Files

```python
import os
from pathlib import Path
from mutagen import File as MutagenFile

def extract_metadata(file_path: str) -> dict:
    """Extract metadata from audio file."""
    metadata = {
        "title": Path(file_path).stem,
        "artist": "",
        "album": "",
        "genre": ""
    }

    try:
        audio = MutagenFile(file_path, easy=True)
        if audio:
            metadata["title"] = audio.get("title", [metadata["title"]])[0]
            metadata["artist"] = audio.get("artist", [""])[0]
            metadata["album"] = audio.get("album", [""])[0]
            metadata["genre"] = audio.get("genre", [""])[0]
    except:
        pass

    return metadata

def load_music_folder(folder: str) -> list[dict]:
    """Load all music files from a folder."""
    songs = []
    extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

    for root, dirs, files in os.walk(folder):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                path = os.path.join(root, file)
                metadata = extract_metadata(path)
                songs.append({
                    "path": path,
                    **metadata
                })

    return songs

songs = load_music_folder("./music")
print(f"Found {len(songs)} songs")
```

### Step 3: Load FMA Dataset (optional)

```python
import requests
import zipfile

def download_fma_small(output_dir: str = "fma_small"):
    """Download FMA small subset (~8GB, 8000 tracks)."""
    # Note: This is a large download
    url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"

    if os.path.exists(output_dir):
        print("FMA already downloaded")
        return output_dir

    print("Downloading FMA small (this will take a while)...")
    response = requests.get(url, stream=True)
    zip_path = "fma_small.zip"

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")

    os.remove(zip_path)
    return output_dir

# For smaller demo, use a subset
def load_fma_subset(fma_dir: str, max_songs: int = 1000) -> list[dict]:
    """Load a subset of FMA dataset."""
    songs = []

    for root, dirs, files in os.walk(fma_dir):
        for file in files:
            if file.endswith(".mp3"):
                if len(songs) >= max_songs:
                    return songs
                path = os.path.join(root, file)
                songs.append({
                    "path": path,
                    "title": Path(file).stem,
                    "artist": "",
                    "album": "",
                    "genre": ""
                })

    return songs
```

### Step 4: Index into Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient("music.db")  # Milvus Lite

client.create_collection(
    collection_name="songs",
    dimension=DIMENSION,
    auto_id=True
)

def index_songs(songs: list[dict], batch_size: int = 10):
    """Embed and index songs."""
    for i in range(0, len(songs), batch_size):
        batch = songs[i:i+batch_size]
        paths = [s["path"] for s in batch]
        embeddings = embed_audio(paths)

        data = []
        for song, emb in zip(batch, embeddings):
            if emb is not None:
                data.append({
                    "vector": emb,
                    "path": song["path"],
                    "title": song["title"],
                    "artist": song["artist"],
                    "album": song["album"],
                    "genre": song["genre"]
                })

        if data:
            client.insert(collection_name="songs", data=data)

        print(f"Indexed {i + len(batch)}/{len(songs)}")

index_songs(songs)
```

### Step 5: Search

```python
def search_by_audio(audio_path: str, top_k: int = 5):
    """Find similar songs by uploading audio."""
    embeddings = embed_audio([audio_path])

    if embeddings[0] is None:
        print("Could not process audio file")
        return []

    results = client.search(
        collection_name="songs",
        data=[embeddings[0]],
        limit=top_k,
        output_fields=["path", "title", "artist", "genre"]
    )
    return results[0]

def search_by_description(query: str, top_k: int = 5):
    """Find songs by text description."""
    query_vector = embed_text([query])[0]

    results = client.search(
        collection_name="songs",
        data=[query_vector],
        limit=top_k,
        output_fields=["path", "title", "artist", "genre"]
    )
    return results[0]

def display_results(results):
    for i, hit in enumerate(results, 1):
        e = hit["entity"]
        print(f"\n#{i} {e['title']}")
        if e["artist"]:
            print(f"    Artist: {e['artist']}")
        if e["genre"]:
            print(f"    Genre: {e['genre']}")
        print(f"    Score: {hit['distance']:.3f}")
        print(f"    Path: {e['path']}")
```

---

## Run Example

```python
# Index music collection
songs = load_music_folder("./music")
index_songs(songs)

# Search by audio
results = search_by_audio("./query_song.mp3")
display_results(results)

# Search by text description
results = search_by_description("upbeat electronic dance music")
display_results(results)

results = search_by_description("calm piano melody")
display_results(results)

results = search_by_description("heavy metal with guitar solo")
display_results(results)
```

---

## Advanced: Audio Feature Extraction with Librosa

```python
def extract_audio_features(audio_path: str) -> dict:
    """Extract traditional audio features using librosa."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    features = {
        "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0]),
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y))),
        "mfcc_mean": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist()
    }

    return features

# Can be used for filtering or hybrid search
features = extract_audio_features("./song.mp3")
print(f"Tempo: {features['tempo']:.1f} BPM")
```

---

## Advanced: Lyrics-based Search

```python
# If you have lyrics, combine with audio search
from sentence_transformers import SentenceTransformer

lyrics_model = SentenceTransformer("all-MiniLM-L6-v2")
LYRICS_DIMENSION = 384

def embed_lyrics(lyrics: list[str]) -> list[list[float]]:
    return lyrics_model.encode(lyrics, normalize_embeddings=True).tolist()

# Create separate collection for lyrics
client.create_collection(
    collection_name="song_lyrics",
    dimension=LYRICS_DIMENSION,
    auto_id=True
)

def search_by_lyrics(query: str, top_k: int = 5):
    """Search songs by lyric content."""
    query_vector = embed_lyrics([query])[0]

    results = client.search(
        collection_name="song_lyrics",
        data=[query_vector],
        limit=top_k,
        output_fields=["title", "artist", "lyrics_snippet"]
    )
    return results[0]
```
