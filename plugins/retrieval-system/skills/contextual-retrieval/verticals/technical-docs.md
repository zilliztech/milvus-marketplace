# Contextual Retrieval for Technical Documentation

> Retrieve code snippets and API references with surrounding explanation.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Documentation Language

<ask_user>
What language is your documentation in?

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
| **Local Model** | Free, offline | Model download required |
</ask_user>

### 3. Local Model Selection (if local)

<ask_user>
| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Better quality |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3GB | Best quality |
</ask_user>

### 4. Data Scale

<ask_user>
How many documentation pages do you have?

- Each page ≈ 5-20 chunks
- Example: 500 pages ≈ 5K-10K vectors

| Vector Count | Recommended Milvus |
|--------------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 5. Project Setup

<ask_user>
| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production |
| **pip** | Quick prototype |
</ask_user>

---

## Dependencies

```bash
uv init tech-docs-search
cd tech-docs-search
uv add pymilvus openai
# Or for local embedding:
uv add pymilvus sentence-transformers
```

---

## Why Contextual Retrieval for Technical Docs

Technical documentation benefits from contextual retrieval because:
1. **Code snippets need explanation** - A matched code block without surrounding prose is often useless
2. **Parameters reference other sections** - API params often link to type definitions elsewhere
3. **Step-by-step instructions** - Matching one step without context breaks the workflow

---

## End-to-End Implementation

### Step 1: Configure Embedding

```python
# === Choose ONE embedding approach ===

# Option A: OpenAI API
from openai import OpenAI
client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

# Option B: Local Model
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("BAAI/bge-base-en-v1.5")
# def embed(texts): return model.encode(texts, normalize_embeddings=True).tolist()
# DIMENSION = 768
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("tech_docs.db")

schema = milvus.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("child_content", DataType.VARCHAR, max_length=65535)
schema.add_field("parent_content", DataType.VARCHAR, max_length=65535)
schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
schema.add_field("section_title", DataType.VARCHAR, max_length=256)
schema.add_field("content_type", DataType.VARCHAR, max_length=32)  # text/code/table

index_params = milvus.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

milvus.create_collection("tech_docs", schema=schema, index_params=index_params)
```

### Step 3: Markdown-Aware Chunking

```python
import re

def chunk_markdown(text: str, parent_size: int = 1500, child_size: int = 300):
    """Chunk markdown preserving structure.

    Returns (child_chunk, parent_chunk, content_type) tuples.
    """
    # Split by headers
    header_pattern = r'\n(?=#{1,3}\s)'
    sections = re.split(header_pattern, text)

    results = []
    for section in sections:
        if len(section.strip()) < 50:
            continue

        # Extract section title
        title_match = re.match(r'^(#{1,3})\s*(.+)', section)
        title = title_match.group(2).strip() if title_match else ""

        # Parent is the full section (up to parent_size)
        parent = section[:parent_size].strip()

        # Find code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', section)

        # Create children
        start = 0
        while start < len(section):
            child = section[start:start + child_size].strip()
            if child:
                # Detect content type
                if '```' in child:
                    content_type = "code"
                elif '|' in child and '---' in child:
                    content_type = "table"
                else:
                    content_type = "text"

                results.append((child, parent, title, content_type))
            start += child_size - 50  # 50 char overlap

    return results

def chunk_preserving_code(text: str) -> list[tuple]:
    """Alternative: ensure code blocks are not split."""
    # Replace code blocks with placeholders
    code_blocks = []
    def save_code(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    text_with_placeholders = re.sub(r'```[\s\S]*?```', save_code, text)

    # Chunk the text
    chunks = chunk_markdown(text_with_placeholders)

    # Restore code blocks
    results = []
    for child, parent, title, ctype in chunks:
        for i, code in enumerate(code_blocks):
            child = child.replace(f"__CODE_BLOCK_{i}__", code)
            parent = parent.replace(f"__CODE_BLOCK_{i}__", code)
        results.append((child, parent, title, "code" if "```" in child else ctype))

    return results
```

### Step 4: Index Documentation

```python
def index_doc(doc_text: str, doc_id: str):
    """Index a documentation page."""
    chunks = chunk_markdown(doc_text)
    children = [c[0] for c in chunks]
    embeddings = embed(children)

    data = [
        {
            "embedding": emb,
            "child_content": child,
            "parent_content": parent,
            "doc_id": doc_id,
            "section_title": title,
            "content_type": ctype
        }
        for (child, parent, title, ctype), emb in zip(chunks, embeddings)
    ]

    milvus.insert(collection_name="tech_docs", data=data)
    print(f"Indexed {len(data)} chunks from {doc_id}")
```

### Step 5: Contextual Search

```python
def search_docs(query: str, content_type: str = None, top_k: int = 5):
    """Search docs and return with parent context."""
    query_embedding = embed([query])[0]

    filter_expr = None
    if content_type:
        filter_expr = f'content_type == "{content_type}"'

    results = milvus.search(
        collection_name="tech_docs",
        data=[query_embedding],
        filter=filter_expr,
        limit=top_k,
        output_fields=["child_content", "parent_content", "section_title", "content_type"]
    )

    return [{
        "matched_text": r["entity"]["child_content"],
        "context": r["entity"]["parent_content"],
        "section": r["entity"]["section_title"],
        "content_type": r["entity"]["content_type"],
        "score": r["distance"]
    } for r in results[0]]
```

---

## Run Example

```python
api_doc = """
## Authentication

All API requests require a Bearer token in the Authorization header.

### Getting a Token

POST /api/v1/auth/token

Request body:
```json
{
  "client_id": "your_client_id",
  "client_secret": "your_client_secret"
}
```

Response:
```json
{
  "access_token": "eyJhbGc...",
  "expires_in": 3600,
  "token_type": "Bearer"
}
```

### Using the Token

Include the token in all subsequent requests:

```bash
curl -H "Authorization: Bearer eyJhbGc..." https://api.example.com/v1/users
```

Tokens expire after 1 hour. Refresh before expiry to avoid interruption.
"""

# Index
index_doc(api_doc, doc_id="api_auth")

# Search - returns full authentication section context
results = search_docs("how to authenticate")

for r in results:
    print(f"Section: {r['section']}")
    print(f"Match: {r['matched_text'][:100]}...")
    print(f"Full Context: {r['context'][:300]}...")
```

---

## Best Practices

1. **Pre-process code blocks** - Ensure code blocks aren't split mid-function
2. **Use markdown-aware chunking** - Split on headers when possible
3. **Index multiple granularities** - Sometimes users want just the code, sometimes the explanation
4. **Store content types** - Enable filtering by code/text/table
