# Milvus Documentation Search

> Search Milvus official documentation by concept, API, or use case.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Documentation Language

<ask_user>
Which documentation language do you need?

| Option | Source |
|--------|--------|
| **English** | milvus.io/docs |
| **Chinese** | milvus.io/docs (中文版) |
| **Both** | Clone both language docs |
</ask_user>

### 2. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality, fast | Requires API key, costs money |
| **Local Model** | Free, offline | Model download required |

Do you have an OpenAI API Key?
</ask_user>

### 3. Local Model Selection (if local)

<ask_user>
**For English docs:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast, CPU-friendly |
| `BAAI/bge-base-en-v1.5` (recommended) | 768 | 440MB | Higher quality |

**For Chinese docs:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-base-zh-v1.5` (recommended) | 768 | 400MB | Balanced |

**For both languages:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-m3` (recommended) | 1024 | 2.2GB | Best multilingual |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470MB | Lighter option |
</ask_user>

### 4. Project Setup

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
uv init milvus-docs-search
cd milvus-docs-search
uv add pymilvus openai gitpython pyyaml
```

### Local Model + uv
```bash
uv init milvus-docs-search
cd milvus-docs-search
uv add pymilvus sentence-transformers gitpython pyyaml
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
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

# Option B: Local Model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()

DIMENSION = 768
```

### Step 2: Clone Documentation

```python
import os
from git import Repo

def clone_milvus_docs(target_dir: str = "milvus-docs"):
    """Clone Milvus documentation repository."""
    repo_url = "https://github.com/milvus-io/milvus-docs"

    if os.path.exists(target_dir):
        print("Pulling latest docs...")
        repo = Repo(target_dir)
        repo.remotes.origin.pull()
    else:
        print("Cloning Milvus docs...")
        Repo.clone_from(repo_url, target_dir, depth=1)  # Shallow clone

    return target_dir

docs_path = clone_milvus_docs()
```

### Step 3: Parse Markdown Files

```python
import re
import yaml
from pathlib import Path

def parse_markdown(file_path: str) -> dict:
    """Parse a markdown file, extracting frontmatter and content."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract YAML frontmatter
    frontmatter = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1])
            except:
                pass
            content = parts[2]

    # Clean content
    content = re.sub(r'```[\s\S]*?```', '[CODE_BLOCK]', content)  # Replace code blocks
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # Remove links, keep text
    content = re.sub(r'#+\s*', '', content)  # Remove headers
    content = re.sub(r'\n{3,}', '\n\n', content)  # Normalize whitespace

    return {
        "title": frontmatter.get("title", Path(file_path).stem),
        "summary": frontmatter.get("summary", ""),
        "content": content.strip(),
        "path": file_path
    }

def load_docs(docs_dir: str, lang: str = "en") -> list[dict]:
    """Load all documentation files."""
    docs = []
    site_dir = os.path.join(docs_dir, "site", lang)

    if not os.path.exists(site_dir):
        # Fallback to different structure
        site_dir = docs_dir

    for root, dirs, files in os.walk(site_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    doc = parse_markdown(file_path)
                    if doc["content"]:  # Skip empty files
                        docs.append(doc)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

    return docs

docs = load_docs(docs_path, lang="en")
print(f"Loaded {len(docs)} documentation pages")
```

### Step 4: Chunk Documents

```python
def chunk_doc(doc: dict, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Split document into chunks."""
    content = doc["content"]
    chunks = []

    # Split by paragraphs first
    paragraphs = content.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Create chunk records
    result = []
    for i, chunk_text in enumerate(chunks):
        result.append({
            "title": doc["title"],
            "path": doc["path"],
            "chunk_index": i,
            "text": chunk_text
        })

    return result

def process_all_docs(docs: list[dict]) -> list[dict]:
    """Process all documents into chunks."""
    all_chunks = []
    for doc in docs:
        chunks = chunk_doc(doc)
        all_chunks.extend(chunks)
    return all_chunks

chunks = process_all_docs(docs)
print(f"Created {len(chunks)} chunks")
```

### Step 5: Index into Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient("milvus_docs.db")  # Milvus Lite
# client = MilvusClient(uri="http://localhost:19530")  # Standalone

client.create_collection(
    collection_name="milvus_docs",
    dimension=DIMENSION,
    auto_id=True
)

def index_chunks(chunks: list[dict], batch_size: int = 50):
    """Embed and index documentation chunks."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        vectors = embed(texts)

        data = [
            {
                "vector": vec,
                "title": c["title"],
                "path": c["path"],
                "chunk_index": c["chunk_index"],
                "text": c["text"][:1000]  # Truncate for storage
            }
            for vec, c in zip(vectors, batch)
        ]
        client.insert(collection_name="milvus_docs", data=data)
        print(f"Indexed {i + len(batch)}/{len(chunks)}")

index_chunks(chunks)
```

### Step 6: Search

```python
def search_docs(query: str, top_k: int = 5):
    """Search Milvus documentation."""
    query_vector = embed([query])[0]

    results = client.search(
        collection_name="milvus_docs",
        data=[query_vector],
        limit=top_k,
        output_fields=["title", "path", "text"]
    )
    return results[0]

def print_results(results):
    """Pretty print search results."""
    for i, hit in enumerate(results, 1):
        entity = hit["entity"]
        print(f"\n{'='*60}")
        print(f"#{i} [{entity['title']}] (score: {hit['distance']:.3f})")
        print(f"Path: {entity['path']}")
        print(f"\n{entity['text'][:300]}...")
```

---

## Run Example

```python
# Full pipeline
docs_path = clone_milvus_docs()
docs = load_docs(docs_path, lang="en")
chunks = process_all_docs(docs)
index_chunks(chunks)

# Search examples
print_results(search_docs("how to create a collection"))
print_results(search_docs("what is HNSW index"))
print_results(search_docs("hybrid search with BM25"))
print_results(search_docs("connect to Zilliz Cloud"))
```

---

## Advanced: Multi-language Search

```python
# For multilingual, use bge-m3
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")

# Load both English and Chinese docs
en_docs = load_docs(docs_path, lang="en")
zh_docs = load_docs(docs_path, lang="zh")
all_docs = en_docs + zh_docs

# Index with language tag
for doc in all_docs:
    doc["lang"] = "en" if "/en/" in doc["path"] else "zh"

# Search with language filter
def search_docs_by_lang(query: str, lang: str = None, top_k: int = 5):
    query_vector = embed([query])[0]

    filter_expr = f'lang == "{lang}"' if lang else None

    results = client.search(
        collection_name="milvus_docs",
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=["title", "path", "text", "lang"]
    )
    return results[0]
```
