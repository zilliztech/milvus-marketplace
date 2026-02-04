# Academic Paper Search

> Search academic papers by research topic, method, or concept with citation-aware ranking.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Paper Language

<ask_user>
What language are your papers primarily in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** (arXiv, ACL, NeurIPS) | English-optimized models |
| **Chinese** (知网, 万方) | Chinese-optimized models |
| **Multilingual** | Multilingual models |
</ask_user>

### 2. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality, fast | Requires API key, costs money |
| **Local Model** | Free, offline | Model download required |
</ask_user>

### 3. Local Model Selection (if local)

<ask_user>
**For English papers:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast prototyping |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Higher quality |
| `allenai/specter2` | 768 | 440MB | Scientific paper specialized |

**For Chinese papers:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-base-zh-v1.5` | 768 | 400MB | Balanced |
| `BAAI/bge-large-zh-v1.5` | 1024 | 1.3GB | Best quality |

**For multilingual:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-m3` | 1024 | 2.2GB | Best multilingual |
</ask_user>

### 4. Data Scale

<ask_user>
How many papers do you plan to index?

- Each paper with multi-field indexing ≈ 3-5 vectors (title, abstract, content)
- Example: 1000 papers ≈ 3K-5K vectors

| Vector Count | Recommended Milvus |
|--------------|-------------------|
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

### OpenAI + uv
```bash
uv init academic-search
cd academic-search
uv add pymilvus openai
```

### Local Model + uv
```bash
uv init academic-search
cd academic-search
uv add pymilvus sentence-transformers
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
model = SentenceTransformer("BAAI/bge-m3")  # or allenai/specter2

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()

DIMENSION = 1024  # Adjust based on model
```

### Step 2: Create Collection with Multi-Field Schema

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("papers.db")  # Milvus Lite

# Create schema for multi-field indexing
schema = client.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("abstract", DataType.VARCHAR, max_length=65535)
schema.add_field("authors", DataType.VARCHAR, max_length=1024)
schema.add_field("venue", DataType.VARCHAR, max_length=256)
schema.add_field("year", DataType.INT32)
schema.add_field("citations", DataType.INT32)
schema.add_field("field", DataType.VARCHAR, max_length=128)
schema.add_field("title_embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("abstract_embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)

index_params = client.prepare_index_params()
index_params.add_index("title_embedding", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index("abstract_embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("papers", schema=schema, index_params=index_params)
```

### Step 3: Index Papers

```python
def index_papers(papers: list[dict]):
    """Index papers with multi-field embeddings.

    papers: [{"title": "...", "abstract": "...", "authors": [...], ...}]
    """
    titles = [p["title"] for p in papers]
    abstracts = [p["abstract"] for p in papers]

    title_embeddings = embed(titles)
    abstract_embeddings = embed(abstracts)

    data = [
        {
            "title": p["title"],
            "abstract": p["abstract"][:5000],
            "authors": ", ".join(p.get("authors", [])),
            "venue": p.get("venue", ""),
            "year": p.get("year", 0),
            "citations": p.get("citations", 0),
            "field": p.get("field", ""),
            "title_embedding": te,
            "abstract_embedding": ae
        }
        for p, te, ae in zip(papers, title_embeddings, abstract_embeddings)
    ]

    client.insert(collection_name="papers", data=data)
    print(f"Indexed {len(papers)} papers")
```

### Step 4: Search with Citation Weighting

```python
import math

def search_papers(query: str, top_k: int = 10, field: str = None):
    """Search papers with title-priority and citation boosting."""
    query_embedding = embed([query])[0]

    filter_expr = f'field == "{field}"' if field else None

    # Search titles first
    title_results = client.search(
        collection_name="papers",
        data=[query_embedding],
        anns_field="title_embedding",
        filter=filter_expr,
        limit=top_k,
        output_fields=["title", "authors", "year", "citations", "abstract"]
    )

    # Search abstracts
    abstract_results = client.search(
        collection_name="papers",
        data=[query_embedding],
        anns_field="abstract_embedding",
        filter=filter_expr,
        limit=top_k,
        output_fields=["title", "authors", "year", "citations", "abstract"]
    )

    # Merge and rank with citation boost
    seen = set()
    merged = []

    for r in title_results[0] + abstract_results[0]:
        title = r["entity"]["title"]
        if title in seen:
            continue
        seen.add(title)

        # Citation boost (logarithmic)
        citations = r["entity"]["citations"]
        citation_boost = math.log10(citations + 1) / 5

        # Recency boost
        year = r["entity"]["year"]
        recency_boost = (year - 2000) / 25 * 0.1 if year > 2000 else 0

        r["final_score"] = r["distance"] + citation_boost + recency_boost
        merged.append(r)

    merged.sort(key=lambda x: x["final_score"], reverse=True)
    return merged[:top_k]

def print_results(results):
    for i, r in enumerate(results, 1):
        e = r["entity"]
        print(f"\n#{i} {e['title']}")
        print(f"    Authors: {e['authors'][:50]}...")
        print(f"    Year: {e['year']} | Citations: {e['citations']}")
        print(f"    Score: {r['final_score']:.3f}")
```

### Step 5: Find Related Papers

```python
def find_related_papers(paper_title: str, top_k: int = 10):
    """Find papers related to a given paper."""
    # Get the paper's abstract embedding
    results = client.query(
        collection_name="papers",
        filter=f'title == "{paper_title}"',
        output_fields=["abstract_embedding"],
        limit=1
    )

    if not results:
        return []

    paper_embedding = results[0]["abstract_embedding"]

    # Search for similar papers (exclude self)
    related = client.search(
        collection_name="papers",
        data=[paper_embedding],
        anns_field="abstract_embedding",
        filter=f'title != "{paper_title}"',
        limit=top_k,
        output_fields=["title", "authors", "year", "citations"]
    )

    return related[0]
```

---

## Run Example

```python
# Sample papers data
papers = [
    {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        "authors": ["Vaswani", "Shazeer", "Parmar"],
        "venue": "NeurIPS",
        "year": 2017,
        "citations": 50000,
        "field": "NLP"
    },
    # ... more papers
]

# Index
index_papers(papers)

# Search
results = search_papers("transformer attention mechanism")
print_results(results)

# Find related
related = find_related_papers("Attention Is All You Need")
```
