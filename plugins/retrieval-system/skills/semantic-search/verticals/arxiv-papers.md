# arXiv Paper Search

> Search academic papers from arXiv by research topic, method, or concept.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Paper Language

<ask_user>
What language are your papers primarily in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** (arXiv, ACL, NeurIPS, etc.) | English-optimized models |
| **Chinese** (知网, 万方, arXiv Chinese) | Chinese-optimized models |
| **Multilingual** | Multilingual models |
</ask_user>

### 2. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality, fast, no GPU needed | Requires API key, costs money, data sent to cloud |
| **Local Model** | Free, offline, data stays local | Model download required, large models need GPU |

Do you have an OpenAI API Key and willing to use it?
</ask_user>

### 3. Local Model Selection (if local)

<ask_user>
**For English papers:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast, CPU-friendly, good for prototyping |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Higher quality |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3GB | Best quality, GPU recommended |
| `allenai/specter2` | 768 | 440MB | Trained on scientific papers |

**For Chinese papers:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-small-zh-v1.5` | 512 | 95MB | Fast, CPU-friendly |
| `BAAI/bge-base-zh-v1.5` | 768 | 400MB | Balanced |
| `BAAI/bge-large-zh-v1.5` | 1024 | 1.3GB | Best quality, GPU recommended |

**For multilingual:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470MB | 50+ languages, CPU-friendly |
| `BAAI/bge-m3` | 1024 | 2.2GB | Best multilingual, GPU recommended |
</ask_user>

### 4. Data Scale

<ask_user>
How many papers do you plan to index?

- Each paper ≈ 10-50 chunks (split by section/paragraph)
- Example: 200 papers ≈ 5K-10K vectors

| Vector Count | Recommended Milvus |
|--------------|-------------------|
| < 100K | **Milvus Lite** (embedded, zero deployment) |
| 100K - 10M | **Milvus Standalone** (Docker) |
| > 10M | **Zilliz Cloud** (managed service) |
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

Based on your choices:

### OpenAI + uv
```bash
uv init arxiv-search
cd arxiv-search
uv add pymilvus openai arxiv pymupdf
```

### Local Model + uv
```bash
uv init arxiv-search
cd arxiv-search
uv add pymilvus sentence-transformers arxiv pymupdf
```

### pip
```bash
pip install pymilvus arxiv pymupdf
pip install openai  # or sentence-transformers
```

---

## End-to-End Implementation

### Step 1: Configure Embedding

```python
# === Choose ONE of the following ===

# Option A: OpenAI API
from openai import OpenAI
client = OpenAI()  # Uses OPENAI_API_KEY env var

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

# Option B: Local Model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-en-v1.5")  # Change based on your choice

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()

DIMENSION = 768  # Adjust based on model
```

### Step 2: Download Papers from arXiv

```python
import arxiv

def download_papers(query: str, max_results: int = 100, output_dir: str = "papers"):
    """Download papers from arXiv by search query."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in arxiv.Client().results(search):
        # Download PDF
        pdf_path = result.download_pdf(dirpath=output_dir)
        papers.append({
            "id": result.entry_id,
            "title": result.title,
            "abstract": result.summary,
            "authors": [a.name for a in result.authors],
            "published": result.published.isoformat(),
            "pdf_path": pdf_path
        })
    return papers

# Example: Download 100 papers about "vector database"
papers = download_papers("vector database", max_results=100)
```

### Step 3: Extract Text & Chunk

```python
import fitz  # pymupdf

def extract_text(pdf_path: str) -> str:
    """Extract text from PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def process_papers(papers: list[dict]) -> list[dict]:
    """Process all papers into chunks."""
    all_chunks = []
    for paper in papers:
        text = extract_text(paper["pdf_path"])
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "paper_id": paper["id"],
                "title": paper["title"],
                "chunk_index": i,
                "text": chunk
            })
    return all_chunks
```

### Step 4: Index into Milvus

```python
from pymilvus import MilvusClient

# Choose based on your scale:
client = MilvusClient("papers.db")  # Milvus Lite
# client = MilvusClient(uri="http://localhost:19530")  # Standalone
# client = MilvusClient(uri="https://xxx.zillizcloud.com", token="xxx")  # Zilliz Cloud

# Create collection
client.create_collection(
    collection_name="papers",
    dimension=DIMENSION,
    auto_id=True
)

def index_chunks(chunks: list[dict], batch_size: int = 100):
    """Embed and insert chunks in batches."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        vectors = embed(texts)

        data = [
            {
                "vector": vec,
                "paper_id": c["paper_id"],
                "title": c["title"],
                "chunk_index": c["chunk_index"],
                "text": c["text"][:500]  # Store truncated text
            }
            for vec, c in zip(vectors, batch)
        ]
        client.insert(collection_name="papers", data=data)

# Process and index
chunks = process_papers(papers)
index_chunks(chunks)
```

### Step 5: Search

```python
def search_papers(query: str, top_k: int = 5):
    """Search papers by semantic query."""
    query_vector = embed([query])[0]
    results = client.search(
        collection_name="papers",
        data=[query_vector],
        limit=top_k,
        output_fields=["paper_id", "title", "text"]
    )
    return results[0]

# Example queries
results = search_papers("attention mechanism in transformers")
for hit in results:
    print(f"[{hit['entity']['title']}] (score: {hit['distance']:.3f})")
    print(f"  {hit['entity']['text'][:200]}...")
    print()
```

---

## Run Example

```python
# Full pipeline
papers = download_papers("large language model", max_results=50)
chunks = process_papers(papers)
index_chunks(chunks)

# Search
results = search_papers("how does RLHF work")
```
