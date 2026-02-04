# Contextual Retrieval for Legal Contracts

> Retrieve legal clauses with surrounding context for better understanding.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Contract Language

<ask_user>
What language are your contracts in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** | English models |
| **Chinese** | Chinese models |
| **Bilingual** | Multilingual models |
</ask_user>

### 2. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key, data sent to cloud |
| **Local Model** | Free, data stays local | Model download required |

Note: For sensitive legal documents, local models keep data on-premise.
</ask_user>

### 3. Local Model Selection (if local)

<ask_user>
**For English contracts:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Good balance |
| `nlpaueb/legal-bert-base-uncased` | 768 | 440MB | Legal domain trained |

**For Chinese contracts:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-base-zh-v1.5` | 768 | 400MB | Good balance |
| `BAAI/bge-large-zh-v1.5` | 1024 | 1.3GB | Best quality |
</ask_user>

### 4. Data Scale

<ask_user>
How many contracts do you have?

- Each contract ≈ 20-100 chunks
- Example: 500 contracts ≈ 25K-50K vectors

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
uv init legal-retrieval
cd legal-retrieval
uv add pymilvus openai
# Or for local embedding:
uv add pymilvus sentence-transformers
```

---

## Why Contextual Retrieval for Legal Docs

Legal documents require special handling because:
1. **Clauses reference other clauses** - "Subject to Section 5.2..."
2. **Defined terms** - Terms defined once, used throughout
3. **Context changes meaning** - Same phrase means different things in different sections

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

milvus = MilvusClient("legal_contracts.db")

schema = milvus.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("child_content", DataType.VARCHAR, max_length=65535)   # Child chunk for matching
schema.add_field("parent_content", DataType.VARCHAR, max_length=65535)  # Parent chunk for context
schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
schema.add_field("section", DataType.VARCHAR, max_length=256)
schema.add_field("clause_type", DataType.VARCHAR, max_length=64)

index_params = milvus.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

milvus.create_collection("legal_contracts", schema=schema, index_params=index_params)
```

### Step 3: Contextual Chunking

```python
import re

def chunk_contract(text: str, parent_size: int = 2048, child_size: int = 512):
    """Chunk contract preserving clause boundaries.

    Returns (child_chunk, parent_chunk) pairs.
    """
    # Split by legal section markers
    section_pattern = r'\n(?=\d+\.\s|Article\s+\d+|Section\s+\d+|ARTICLE\s+)'
    sections = re.split(section_pattern, text)

    pairs = []
    for section in sections:
        if len(section.strip()) < 50:
            continue

        # Section is the parent (up to parent_size)
        parent = section[:parent_size].strip()

        # Create overlapping children
        start = 0
        while start < len(section):
            child = section[start:start + child_size].strip()
            if child:
                pairs.append((child, parent))
            start += child_size - 100  # 100 char overlap

    return pairs

def detect_clause_type(text: str) -> str:
    """Detect clause type."""
    text_lower = text.lower()
    if any(t in text_lower for t in ["terminat", "cancel"]):
        return "termination"
    if any(t in text_lower for t in ["confidential", "non-disclosure"]):
        return "confidentiality"
    if any(t in text_lower for t in ["indemn", "liability"]):
        return "indemnification"
    if any(t in text_lower for t in ["payment", "fee", "price"]):
        return "payment"
    return "general"
```

### Step 4: Index Contracts

```python
def index_contract(contract_text: str, doc_id: str):
    """Index a contract with contextual chunks."""
    pairs = chunk_contract(contract_text)
    children = [p[0] for p in pairs]
    embeddings = embed(children)

    data = [
        {
            "embedding": emb,
            "child_content": child,
            "parent_content": parent,
            "doc_id": doc_id,
            "section": "",
            "clause_type": detect_clause_type(child)
        }
        for (child, parent), emb in zip(pairs, embeddings)
    ]

    milvus.insert(collection_name="legal_contracts", data=data)
    print(f"Indexed {len(data)} chunks from {doc_id}")
```

### Step 5: Contextual Search

```python
def search_contracts(query: str, clause_type: str = None, top_k: int = 5):
    """Search contracts and return with parent context."""
    query_embedding = embed([query])[0]

    filter_expr = None
    if clause_type:
        filter_expr = f'clause_type == "{clause_type}"'

    results = milvus.search(
        collection_name="legal_contracts",
        data=[query_embedding],
        filter=filter_expr,
        limit=top_k,
        output_fields=["child_content", "parent_content", "doc_id", "clause_type"]
    )

    return [{
        "matched_text": r["entity"]["child_content"],
        "context": r["entity"]["parent_content"],
        "doc_id": r["entity"]["doc_id"],
        "clause_type": r["entity"]["clause_type"],
        "score": r["distance"]
    } for r in results[0]]
```

---

## Run Example

```python
contract = """
ARTICLE 5 - TERMINATION

5.1 Termination for Convenience
Either party may terminate this Agreement upon thirty (30) days prior
written notice to the other party.

5.2 Termination for Cause
Either party may terminate this Agreement immediately upon written notice
if the other party:
(a) Materially breaches any provision of this Agreement and fails to cure
    such breach within fifteen (15) days after receiving written notice; or
(b) Becomes insolvent or files for bankruptcy protection.

5.3 Effect of Termination
Upon termination:
(a) All licenses granted hereunder shall immediately terminate;
(b) Each party shall return or destroy Confidential Information;
(c) Sections 6, 7, and 8 shall survive termination.
"""

# Index
index_contract(contract, doc_id="service_agreement_001")

# Search - returns full Article 5 context even for specific match
results = search_contracts("what happens if they go bankrupt")

for r in results:
    print(f"Match: {r['matched_text'][:100]}...")
    print(f"Context: {r['context'][:200]}...")
    print(f"Clause Type: {r['clause_type']}")
```

---

## Best Practices

1. **Preserve section numbering** - Critical for legal reference
2. **Don't split defined terms** - Keep definitions with their terms
3. **Higher parent sizes** - Legal context often spans paragraphs
4. **Store clause types** - Enable filtering by clause category
