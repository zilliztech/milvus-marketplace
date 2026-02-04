# Contract Search

> Search legal contracts by clause, term, or concept.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Contract Language

<ask_user>
What language are your contracts in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** | English-optimized models |
| **Chinese** | Chinese-optimized models |
| **Bilingual** (中英双语) | Multilingual models |
</ask_user>

### 2. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality, fast | Requires API key, costs money |
| **Local Model** | Free, offline, data stays local | Model download required |

Note: For sensitive legal documents, local models keep data on-premise.
</ask_user>

### 3. Local Model Selection (if local)

<ask_user>
**For English contracts:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast prototyping |
| `BAAI/bge-base-en-v1.5` (recommended) | 768 | 440MB | Good quality |
| `nlpaueb/legal-bert-base-uncased` | 768 | 440MB | Legal domain trained |

**For Chinese contracts:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-base-zh-v1.5` (recommended) | 768 | 400MB | Balanced |
| `BAAI/bge-large-zh-v1.5` | 1024 | 1.3GB | Best quality |

**For bilingual:**

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `BAAI/bge-m3` (recommended) | 1024 | 2.2GB | Best multilingual |
</ask_user>

### 4. Data Scale

<ask_user>
How many contracts do you have?

- Each contract ≈ 20-100 chunks (depending on length)
- Example: 100 contracts ≈ 5K-10K vectors

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
uv init contract-search
cd contract-search
uv add pymilvus openai pymupdf python-docx
```

### Local Model + uv
```bash
uv init contract-search
cd contract-search
uv add pymilvus sentence-transformers pymupdf python-docx
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

### Step 2: Load Contracts

```python
import os
import fitz  # pymupdf
from docx import Document
from pathlib import Path

def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_docx_text(file_path: str) -> str:
    """Extract text from DOCX."""
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)

def load_contracts(folder: str) -> list[dict]:
    """Load all contracts from a folder."""
    contracts = []

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        ext = Path(file).suffix.lower()

        try:
            if ext == ".pdf":
                text = extract_pdf_text(file_path)
            elif ext in [".docx", ".doc"]:
                text = extract_docx_text(file_path)
            else:
                continue

            if text.strip():
                contracts.append({
                    "filename": file,
                    "path": file_path,
                    "text": text
                })
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return contracts

contracts = load_contracts("./contracts")
print(f"Loaded {len(contracts)} contracts")
```

### Step 3: Extract Metadata & Chunk

```python
import re

def extract_contract_metadata(text: str) -> dict:
    """Extract key metadata from contract text."""
    metadata = {}

    # Try to find parties (简化版，实际可能需要更复杂的解析)
    party_pattern = r"(?:between|by and between)\s+(.+?)\s+(?:and|AND)\s+(.+?)(?:\.|,|\n)"
    match = re.search(party_pattern, text[:2000], re.IGNORECASE)
    if match:
        metadata["party_a"] = match.group(1).strip()[:100]
        metadata["party_b"] = match.group(2).strip()[:100]

    # Try to find date
    date_pattern = r"(?:dated?|effective)\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4}|\d{4}[-/]\d{2}[-/]\d{2})"
    match = re.search(date_pattern, text[:2000], re.IGNORECASE)
    if match:
        metadata["date"] = match.group(1)

    return metadata

def chunk_contract(contract: dict, chunk_size: int = 1000, overlap: int = 150) -> list[dict]:
    """Split contract into chunks, trying to preserve clause boundaries."""
    text = contract["text"]
    metadata = extract_contract_metadata(text)

    # Try to split by clause markers
    clause_pattern = r'\n(?=\d+\.\s|\([a-z]\)|\([0-9]+\)|Article\s+\d+|Section\s+\d+|ARTICLE\s+)'
    sections = re.split(clause_pattern, text)

    chunks = []
    current_chunk = ""

    for section in sections:
        if len(current_chunk) + len(section) < chunk_size:
            current_chunk += section
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = section

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Create chunk records
    result = []
    for i, chunk_text in enumerate(chunks):
        result.append({
            "filename": contract["filename"],
            "chunk_index": i,
            "text": chunk_text,
            **metadata
        })

    return result

def process_contracts(contracts: list[dict]) -> list[dict]:
    """Process all contracts into chunks."""
    all_chunks = []
    for contract in contracts:
        chunks = chunk_contract(contract)
        all_chunks.extend(chunks)
    return all_chunks

chunks = process_contracts(contracts)
print(f"Created {len(chunks)} chunks")
```

### Step 4: Index into Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient("contracts.db")  # Milvus Lite

client.create_collection(
    collection_name="contracts",
    dimension=DIMENSION,
    auto_id=True
)

def index_chunks(chunks: list[dict], batch_size: int = 50):
    """Embed and index contract chunks."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        vectors = embed(texts)

        data = [
            {
                "vector": vec,
                "filename": c["filename"],
                "chunk_index": c["chunk_index"],
                "text": c["text"][:1000],
                "party_a": c.get("party_a", ""),
                "party_b": c.get("party_b", ""),
                "date": c.get("date", "")
            }
            for vec, c in zip(vectors, batch)
        ]
        client.insert(collection_name="contracts", data=data)
        print(f"Indexed {i + len(batch)}/{len(chunks)}")

index_chunks(chunks)
```

### Step 5: Search

```python
def search_contracts(query: str, top_k: int = 5, filename: str = None):
    """Search contracts by semantic query."""
    query_vector = embed([query])[0]

    filter_expr = f'filename == "{filename}"' if filename else None

    results = client.search(
        collection_name="contracts",
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=["filename", "text", "party_a", "party_b"]
    )
    return results[0]

def print_results(results):
    for i, hit in enumerate(results, 1):
        e = hit["entity"]
        print(f"\n{'='*60}")
        print(f"#{i} [{e['filename']}] (score: {hit['distance']:.3f})")
        if e.get("party_a"):
            print(f"Parties: {e['party_a']} & {e['party_b']}")
        print(f"\n{e['text'][:300]}...")
```

---

## Run Example

```python
# Load, process, and index
contracts = load_contracts("./contracts")
chunks = process_contracts(contracts)
index_chunks(chunks)

# Search examples
print_results(search_contracts("termination clause"))
print_results(search_contracts("liability limitation"))
print_results(search_contracts("confidentiality and non-disclosure"))
print_results(search_contracts("payment terms and schedule"))
```

---

## Advanced: Clause Classification

```python
CLAUSE_TYPES = [
    "termination",
    "liability",
    "confidentiality",
    "payment",
    "warranty",
    "indemnification",
    "force majeure",
    "dispute resolution"
]

def classify_clause(text: str) -> str:
    """Classify a clause by comparing to clause type embeddings."""
    clause_vectors = embed(CLAUSE_TYPES)
    text_vector = embed([text])[0]

    # Find most similar clause type
    import numpy as np
    similarities = [np.dot(text_vector, cv) for cv in clause_vectors]
    best_idx = np.argmax(similarities)

    return CLAUSE_TYPES[best_idx] if similarities[best_idx] > 0.3 else "other"
```
