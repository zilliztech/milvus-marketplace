# Plagiarism & Content Detection

> Detect plagiarism and content spinning using semantic similarity.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Document Language

<ask_user>
What language are your documents in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** | English models |
| **Chinese** | Chinese models |
| **Multilingual** | Multilingual models |
</ask_user>

### 2. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download |

Local options:
- `BAAI/bge-large-en-v1.5` (1024d, English)
- `BAAI/bge-large-zh-v1.5` (1024d, Chinese)
</ask_user>

### 3. Data Scale

<ask_user>
How many documents in your reference library?

- Each document ≈ 10-50 chunks

| Document Count | Recommended Milvus |
|----------------|-------------------|
| < 10K | **Milvus Lite** |
| 10K - 1M | **Milvus Standalone** |
| > 1M | **Zilliz Cloud** |
</ask_user>

### 4. Project Setup

<ask_user>
| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production |
| **pip** | Quick prototype |
</ask_user>

---

## Dependencies

```bash
uv init plagiarism-detector
cd plagiarism-detector
uv add pymilvus sentence-transformers
```

---

## Detection Types

| Type | Description | Detection Method |
|------|-------------|------------------|
| Exact copy | Complete copy | Hash matching |
| Paraphrase | Same meaning, different words | Semantic vectors |
| Sentence shuffle | Reordered sentences | Per-sentence detection |

---

## End-to-End Implementation

### Step 1: Configure Models

```python
from sentence_transformers import SentenceTransformer
import hashlib
import re

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
DIMENSION = 1024

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()

def normalize_text(text: str) -> str:
    """Normalize text for hashing."""
    text = ''.join(text.split())
    text = re.sub(r'[.,!?;:"\'\[\]<>]', '', text)
    return text.lower()

def hash_text(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode()).hexdigest()

def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'[.!?\n]', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("documents.db")

schema = milvus.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("source", DataType.VARCHAR, max_length=512)
schema.add_field("author", DataType.VARCHAR, max_length=128)
schema.add_field("doc_type", DataType.VARCHAR, max_length=32)

index_params = milvus.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index("content_hash", index_type="TRIE")

milvus.create_collection("document_library", schema=schema, index_params=index_params)
```

### Step 3: Add Documents to Library

```python
import uuid
import time

def add_to_library(content: str, source: str, author: str = "", doc_type: str = "article"):
    """Add document to reference library."""
    chunks = [content[i:i+500] for i in range(0, len(content), 450)]

    data = []
    for chunk in chunks:
        data.append({
            "content_hash": hash_text(chunk),
            "embedding": embed([chunk])[0],
            "content": chunk,
            "source": source,
            "author": author,
            "doc_type": doc_type
        })

    milvus.insert(collection_name="document_library", data=data)
    return len(data)
```

### Step 4: Check for Plagiarism

```python
def check_document(content: str, chunk_size: int = 500) -> dict:
    """Check document for plagiarism."""
    # 1. Exact match check (hash)
    content_hash = hash_text(content)
    exact_match = milvus.query(
        collection_name="document_library",
        filter=f'content_hash == "{content_hash}"',
        output_fields=["source", "author"],
        limit=1
    )
    if exact_match:
        return {
            "is_plagiarized": True,
            "type": "exact_copy",
            "similarity": 1.0,
            "match_source": exact_match[0]["source"],
            "match_author": exact_match[0]["author"]
        }

    # 2. Chunk-level detection
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    plagiarized_chunks = []

    for i, chunk in enumerate(chunks):
        embedding = embed([chunk])[0]
        results = milvus.search(
            collection_name="document_library",
            data=[embedding],
            limit=1,
            output_fields=["content", "source"]
        )

        if results[0] and results[0][0]["distance"] > 0.85:
            plagiarized_chunks.append({
                "chunk_index": i,
                "chunk_content": chunk[:100] + "...",
                "similarity": results[0][0]["distance"],
                "match_source": results[0][0]["entity"]["source"],
                "match_content": results[0][0]["entity"]["content"][:100] + "..."
            })

    # 3. Sentence-level detection (finer)
    sentences = split_sentences(content)
    plagiarized_sentences = []

    for sent in sentences:
        embedding = embed([sent])[0]
        results = milvus.search(
            collection_name="document_library",
            data=[embedding],
            limit=1,
            output_fields=["source"]
        )

        if results[0] and results[0][0]["distance"] > 0.9:
            plagiarized_sentences.append({
                "sentence": sent,
                "similarity": results[0][0]["distance"],
                "match_source": results[0][0]["entity"]["source"]
            })

    # Calculate overall ratio
    chunk_ratio = len(plagiarized_chunks) / len(chunks) if chunks else 0
    sentence_ratio = len(plagiarized_sentences) / len(sentences) if sentences else 0
    overall_ratio = max(chunk_ratio, sentence_ratio)

    return {
        "is_plagiarized": overall_ratio > 0.3,
        "type": "partial" if overall_ratio > 0 else "original",
        "overall_similarity": overall_ratio,
        "chunk_analysis": {
            "total": len(chunks),
            "plagiarized": len(plagiarized_chunks),
            "ratio": chunk_ratio,
            "details": plagiarized_chunks[:5]
        },
        "sentence_analysis": {
            "total": len(sentences),
            "plagiarized": len(plagiarized_sentences),
            "ratio": sentence_ratio,
            "details": plagiarized_sentences[:10]
        }
    }
```

---

## Run Example

```python
# Add original document to library
add_to_library(
    content="Deep learning is a branch of machine learning that uses multi-layer neural networks to learn hierarchical representations of data.",
    source="Introduction to AI",
    author="John Smith"
)

# Check new document (paraphrased)
result = check_document(
    "Deep learning belongs to the machine learning subfield, with its core being multi-layer neural network structure to learn data representations."
)

if result["is_plagiarized"]:
    print(f"Plagiarism detected! Similarity: {result['overall_similarity']:.1%}")
    print("\nSuspicious passages:")
    for detail in result["sentence_analysis"]["details"]:
        print(f"  - {detail['sentence'][:50]}... ({detail['similarity']:.1%})")
        print(f"    Source: {detail['match_source']}")
else:
    print("No plagiarism detected.")
```

---

## Threshold Guidelines

| Scenario | Sentence | Chunk | Overall Judgment |
|----------|----------|-------|------------------|
| Academic | 0.90 | 0.85 | >15% suspicious |
| News | 0.85 | 0.80 | >30% suspicious |
| Assignments | 0.88 | 0.82 | >20% suspicious |
