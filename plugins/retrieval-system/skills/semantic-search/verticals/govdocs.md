# GOVDOCS1 Government Document Search

> Search government documents from the GOVDOCS1 corpus.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Document Types

<ask_user>
Which document types do you want to index?

| Type | Notes |
|------|-------|
| **PDF only** | Most common, easiest to process |
| **Multiple formats** | PDF, DOC, TXT, HTML, etc. |
| **All formats** | Includes images, emails, etc. |
</ask_user>

### 2. Data Subset

<ask_user>
How much of GOVDOCS1 do you want to use?

| Subset | Size | Documents |
|--------|------|-----------|
| **Tiny** (thread 0 only) | ~5GB | ~1K docs |
| **Small** (threads 0-9) | ~50GB | ~10K docs |
| **Medium** (threads 0-49) | ~250GB | ~50K docs |
| **Full** | ~500GB | ~1M docs |

Note: Full dataset is very large. Start with a subset.
</ask_user>

### 3. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download needed |
</ask_user>

### 4. Local Model (if local)

<ask_user>
Choose embedding model:

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast |
| `BAAI/bge-base-en-v1.5` (recommended) | 768 | 440MB | Higher quality |
</ask_user>

### 5. Data Scale

<ask_user>
Based on your subset choice:

- Tiny: ~1K docs → ~10K vectors
- Small: ~10K docs → ~100K vectors
- Medium: ~50K docs → ~500K vectors
- Full: ~1M docs → ~10M vectors

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

### OpenAI + uv
```bash
uv init govdocs-search
cd govdocs-search
uv add pymilvus openai pymupdf python-docx beautifulsoup4 requests tqdm
```

### Local Model + uv
```bash
uv init govdocs-search
cd govdocs-search
uv add pymilvus sentence-transformers pymupdf python-docx beautifulsoup4 requests tqdm
```

---

## End-to-End Implementation

### Step 1: Configure Embedding

```python
# === Choose ONE ===

# Option A: OpenAI API
from openai import OpenAI
client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    # Handle empty texts
    texts = [t if t.strip() else "empty document" for t in texts]
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

# Option B: Local Model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed(texts: list[str]) -> list[list[float]]:
    texts = [t if t.strip() else "empty document" for t in texts]
    return model.encode(texts, normalize_embeddings=True).tolist()

DIMENSION = 768
```

### Step 2: Download GOVDOCS1 Subset

```python
import os
import requests
import zipfile
from tqdm import tqdm

def download_govdocs_thread(thread_num: int, output_dir: str = "govdocs"):
    """Download a single GOVDOCS1 thread (zip file)."""
    os.makedirs(output_dir, exist_ok=True)

    # GOVDOCS1 is hosted at digitalcorpora.org
    url = f"https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/{thread_num:03d}.zip"
    zip_path = os.path.join(output_dir, f"{thread_num:03d}.zip")
    extract_path = os.path.join(output_dir, f"{thread_num:03d}")

    if os.path.exists(extract_path):
        print(f"Thread {thread_num} already downloaded")
        return extract_path

    print(f"Downloading thread {thread_num}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Extracting thread {thread_num}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_dir)

    os.remove(zip_path)
    return extract_path

def download_govdocs_subset(threads: list[int], output_dir: str = "govdocs"):
    """Download multiple GOVDOCS1 threads."""
    paths = []
    for t in threads:
        path = download_govdocs_thread(t, output_dir)
        paths.append(path)
    return paths

# Download tiny subset (thread 0 only)
download_govdocs_subset([0])
```

### Step 3: Parse Multiple Formats

```python
import fitz  # pymupdf
from docx import Document
from bs4 import BeautifulSoup
from pathlib import Path

def extract_pdf(file_path: str) -> str:
    """Extract text from PDF."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except:
        return ""

def extract_docx(file_path: str) -> str:
    """Extract text from DOCX."""
    try:
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    except:
        return ""

def extract_html(file_path: str) -> str:
    """Extract text from HTML."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            return soup.get_text()
    except:
        return ""

def extract_txt(file_path: str) -> str:
    """Extract text from TXT."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return ""

def extract_text(file_path: str) -> str:
    """Extract text from any supported file type."""
    ext = Path(file_path).suffix.lower()

    extractors = {
        ".pdf": extract_pdf,
        ".docx": extract_docx,
        ".doc": extract_docx,  # May not work for old .doc
        ".html": extract_html,
        ".htm": extract_html,
        ".txt": extract_txt,
        ".text": extract_txt
    }

    extractor = extractors.get(ext)
    if extractor:
        return extractor(file_path)
    return ""
```

### Step 4: Load Documents

```python
import os
from pathlib import Path

def load_govdocs(base_dir: str, file_types: set = None) -> list[dict]:
    """Load documents from GOVDOCS directory."""
    if file_types is None:
        file_types = {".pdf", ".docx", ".doc", ".html", ".htm", ".txt"}

    documents = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if Path(file).suffix.lower() in file_types:
                file_path = os.path.join(root, file)
                text = extract_text(file_path)

                if text and len(text) > 100:  # Skip very short docs
                    documents.append({
                        "path": file_path,
                        "filename": file,
                        "file_type": Path(file).suffix.lower(),
                        "text": text
                    })

    return documents

# Load PDF files only (fastest)
docs = load_govdocs("govdocs/000", file_types={".pdf"})
print(f"Loaded {len(docs)} documents")
```

### Step 5: Chunk Documents

```python
def chunk_document(doc: dict, chunk_size: int = 1000, overlap: int = 150) -> list[dict]:
    """Split document into chunks."""
    text = doc["text"]
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "filename": doc["filename"],
                "file_type": doc["file_type"],
                "chunk_index": len(chunks),
                "text": chunk_text
            })

        start = end - overlap

    return chunks

def process_all_docs(docs: list[dict]) -> list[dict]:
    """Process all documents into chunks."""
    all_chunks = []
    for doc in tqdm(docs, desc="Chunking"):
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    return all_chunks

chunks = process_all_docs(docs)
print(f"Created {len(chunks)} chunks")
```

### Step 6: Index into Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient("govdocs.db")  # Milvus Lite

client.create_collection(
    collection_name="govdocs",
    dimension=DIMENSION,
    auto_id=True
)

def index_chunks(chunks: list[dict], batch_size: int = 50):
    """Embed and index document chunks."""
    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]

        try:
            vectors = embed(texts)

            data = [
                {
                    "vector": vec,
                    "filename": c["filename"],
                    "file_type": c["file_type"],
                    "chunk_index": c["chunk_index"],
                    "text": c["text"][:1000]
                }
                for vec, c in zip(vectors, batch)
            ]
            client.insert(collection_name="govdocs", data=data)
        except Exception as e:
            print(f"Error indexing batch {i}: {e}")

index_chunks(chunks)
```

### Step 7: Search

```python
def search_govdocs(query: str, top_k: int = 5, file_type: str = None):
    """Search government documents."""
    query_vector = embed([query])[0]

    filter_expr = f'file_type == "{file_type}"' if file_type else None

    results = client.search(
        collection_name="govdocs",
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=["filename", "file_type", "text"]
    )
    return results[0]

def print_results(results):
    for i, hit in enumerate(results, 1):
        e = hit["entity"]
        print(f"\n{'='*60}")
        print(f"#{i} [{e['filename']}] ({e['file_type']})")
        print(f"Score: {hit['distance']:.3f}")
        print(f"\n{e['text'][:400]}...")
```

---

## Run Example

```python
# Download and process
download_govdocs_subset([0])  # Thread 0 only
docs = load_govdocs("govdocs/000", file_types={".pdf"})
chunks = process_all_docs(docs)
index_chunks(chunks)

# Search examples
print_results(search_govdocs("environmental protection regulations"))
print_results(search_govdocs("budget allocation"))
print_results(search_govdocs("public health policy"))
print_results(search_govdocs("tax form instructions", file_type=".pdf"))
```

---

## Advanced: File Type Statistics

```python
from collections import Counter

def analyze_corpus(base_dir: str) -> dict:
    """Analyze file types in GOVDOCS corpus."""
    extensions = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            extensions.append(ext)

    return dict(Counter(extensions).most_common(20))

# Example output:
# {'.pdf': 5000, '.html': 2000, '.txt': 1000, '.doc': 500, ...}
stats = analyze_corpus("govdocs/000")
print("File type distribution:")
for ext, count in stats.items():
    print(f"  {ext}: {count}")
```

---

## Notes

- GOVDOCS1 is a research corpus, may contain outdated/broken files
- Start with a small subset to test your pipeline
- PDF extraction works best; other formats may have issues
- Full dataset requires significant storage and processing time
