# Legal Document Search

> Search legal documents, cases, and statutes by legal concept or fact pattern.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Document Language

<ask_user>
What language are your legal documents in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** | English legal models |
| **Chinese** | Chinese-optimized models |
| **Bilingual** | Multilingual models |
</ask_user>

### 2. Document Type

<ask_user>
What types of legal documents?

| Type | Notes |
|------|-------|
| **Contracts** | Commercial agreements, terms |
| **Case Law** | Court decisions, judgments |
| **Statutes/Regulations** | Laws, rules, policies |
| **Mixed** | All types |
</ask_user>

### 3. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key, data sent to cloud |
| **Local Model** | Free, data stays local | Model download required |

Note: For sensitive legal documents, local models keep data on-premise.
</ask_user>

### 4. Local Model Selection (if local)

<ask_user>
**For English legal documents:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | General, good quality |
| `nlpaueb/legal-bert-base-uncased` | 768 | 440MB | Legal domain trained |

**For Chinese legal documents:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-base-zh-v1.5` | 768 | 400MB | General Chinese |
| `BAAI/bge-large-zh-v1.5` | 1024 | 1.3GB | Best quality |
</ask_user>

### 5. Data Scale

<ask_user>
How many documents do you have?

- Each document ≈ 20-100 chunks
- Example: 500 contracts ≈ 25K-50K vectors

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
uv init legal-search
cd legal-search
uv add pymilvus openai
```

### Local Model + uv
```bash
uv init legal-search
cd legal-search
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
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()

DIMENSION = 768
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("legal.db")  # Milvus Lite

schema = client.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("doc_type", DataType.VARCHAR, max_length=32)  # contract/case/statute
schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
schema.add_field("doc_title", DataType.VARCHAR, max_length=256)
schema.add_field("chunk_index", DataType.INT32)
schema.add_field("jurisdiction", DataType.VARCHAR, max_length=64)
schema.add_field("date", DataType.VARCHAR, max_length=32)
schema.add_field("parties", DataType.VARCHAR, max_length=512)

index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("legal_docs", schema=schema, index_params=index_params)
```

### Step 3: Chunk Legal Documents

```python
import re

def chunk_legal_doc(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Chunk legal document preserving clause boundaries."""
    # Try to split by legal section markers
    section_pattern = r'\n(?=\d+\.\s|Article\s+\d+|Section\s+\d+|ARTICLE\s+)'
    sections = re.split(section_pattern, text)

    chunks = []
    current = ""

    for section in sections:
        if len(current) + len(section) < chunk_size:
            current += section
        else:
            if current.strip():
                chunks.append(current.strip())
            current = section

    if current.strip():
        chunks.append(current.strip())

    return chunks

def index_legal_document(doc: dict):
    """Index a legal document.

    doc: {"text": "...", "doc_type": "contract", "doc_id": "...", ...}
    """
    chunks = chunk_legal_doc(doc["text"])
    texts = chunks
    embeddings = embed(texts)

    data = [
        {
            "embedding": emb,
            "content": chunk[:5000],
            "doc_type": doc.get("doc_type", ""),
            "doc_id": doc["doc_id"],
            "doc_title": doc.get("doc_title", ""),
            "chunk_index": i,
            "jurisdiction": doc.get("jurisdiction", ""),
            "date": doc.get("date", ""),
            "parties": doc.get("parties", "")
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    client.insert(collection_name="legal_docs", data=data)
    return len(chunks)
```

### Step 4: Search

```python
def search_legal(query: str, top_k: int = 10,
                 doc_type: str = None,
                 jurisdiction: str = None):
    """Search legal documents."""
    query_embedding = embed([query])[0]

    filters = []
    if doc_type:
        filters.append(f'doc_type == "{doc_type}"')
    if jurisdiction:
        filters.append(f'jurisdiction == "{jurisdiction}"')

    filter_expr = " and ".join(filters) if filters else None

    results = client.search(
        collection_name="legal_docs",
        data=[query_embedding],
        filter=filter_expr,
        limit=top_k,
        output_fields=["content", "doc_type", "doc_title", "jurisdiction", "parties"]
    )
    return results[0]

def search_similar_cases(case_facts: str, top_k: int = 10):
    """Find similar legal cases based on fact pattern."""
    return search_legal(case_facts, top_k=top_k, doc_type="case")

def search_relevant_statutes(legal_issue: str, jurisdiction: str = None, top_k: int = 10):
    """Find relevant statutes for a legal issue."""
    return search_legal(legal_issue, top_k=top_k, doc_type="statute", jurisdiction=jurisdiction)

def print_results(results):
    for i, r in enumerate(results, 1):
        e = r["entity"]
        print(f"\n{'='*60}")
        print(f"#{i} [{e['doc_type'].upper()}] {e['doc_title']}")
        if e["jurisdiction"]:
            print(f"    Jurisdiction: {e['jurisdiction']}")
        if e["parties"]:
            print(f"    Parties: {e['parties'][:100]}...")
        print(f"    Score: {r['distance']:.3f}")
        print(f"\n{e['content'][:300]}...")
```

---

## Run Example

```python
# Index legal documents
contract = {
    "text": "This Agreement is entered into between Party A and Party B...",
    "doc_type": "contract",
    "doc_id": "contract_001",
    "doc_title": "Service Agreement",
    "jurisdiction": "California",
    "date": "2024-01-15",
    "parties": "Acme Corp, Beta Inc"
}
index_legal_document(contract)

# Search examples
results = search_legal("termination clause breach of contract")
print_results(results)

results = search_similar_cases("employer terminated employee without notice")
print_results(results)

results = search_relevant_statutes("data privacy requirements", jurisdiction="California")
print_results(results)
```

---

## Advanced: Clause Classification

```python
CLAUSE_TYPES = [
    "termination", "liability", "confidentiality",
    "payment", "warranty", "indemnification",
    "force majeure", "dispute resolution"
]

def classify_clause(text: str) -> str:
    """Classify a clause by type."""
    clause_embeddings = embed(CLAUSE_TYPES)
    text_embedding = embed([text])[0]

    import numpy as np
    similarities = [np.dot(text_embedding, ce) for ce in clause_embeddings]
    best_idx = np.argmax(similarities)

    return CLAUSE_TYPES[best_idx] if similarities[best_idx] > 0.3 else "other"
```
