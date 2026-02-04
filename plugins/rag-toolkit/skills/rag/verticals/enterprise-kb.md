# Enterprise Knowledge Base

> Build a RAG system for internal company documents, policies, and procedures.

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
| **OpenAI API** | High quality | Data sent to cloud |
| **Local Model** | Data stays on-premise | Model download |

Note: For sensitive enterprise data, local models may be preferred.
</ask_user>

### 3. Local Model (if local)

<ask_user>
| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Higher quality |
| `BAAI/bge-m3` | 1024 | 2.2GB | Multilingual |
</ask_user>

### 4. LLM for Generation

<ask_user>
| Model | Notes |
|-------|-------|
| **GPT-4o** | Best quality, cloud |
| **GPT-4o-mini** | Cost-effective |
| **Local LLM** (Ollama) | On-premise |
</ask_user>

### 5. Data Scale

<ask_user>
How many documents do you have?

- Each document ≈ 50-200 chunks
- Example: 500 documents ≈ 50K-100K vectors

| Vector Count | Recommended Milvus |
|--------------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 6. Project Setup

<ask_user>
| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production |
| **pip** | Quick prototype |
</ask_user>

---

## Dependencies

```bash
uv init enterprise-kb
cd enterprise-kb
uv add pymilvus openai pymupdf python-docx
# Or for local embedding:
uv add pymilvus sentence-transformers openai pymupdf python-docx
```

---

## End-to-End Implementation

### Step 1: Configure Embedding & LLM

```python
# === Choose ONE embedding approach ===

# Option A: OpenAI
from openai import OpenAI
client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

# Option B: Local
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("BAAI/bge-base-en-v1.5")
# def embed(texts): return model.encode(texts, normalize_embeddings=True).tolist()
# DIMENSION = 768

def generate(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("enterprise_kb.db")

schema = milvus.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
schema.add_field("doc_title", DataType.VARCHAR, max_length=256)
schema.add_field("doc_type", DataType.VARCHAR, max_length=32)  # policy/procedure/guide
schema.add_field("department", DataType.VARCHAR, max_length=64)
schema.add_field("chunk_index", DataType.INT32)
schema.add_field("last_updated", DataType.VARCHAR, max_length=32)

index_params = milvus.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

milvus.create_collection("enterprise_kb", schema=schema, index_params=index_params)
```

### Step 3: Document Processing

```python
import fitz  # pymupdf
from docx import Document
import os

def extract_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def extract_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text(path: str) -> str:
    if path.endswith(".pdf"):
        return extract_pdf(path)
    elif path.endswith(".docx"):
        return extract_docx(path)
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]
```

### Step 4: Index Documents

```python
def index_document(file_path: str, doc_id: str, doc_title: str,
                   doc_type: str = "guide", department: str = "",
                   last_updated: str = ""):
    """Index a document into the knowledge base."""
    text = extract_text(file_path)
    chunks = chunk_text(text)
    embeddings = embed(chunks)

    data = [
        {
            "embedding": emb,
            "content": chunk[:5000],
            "doc_id": doc_id,
            "doc_title": doc_title,
            "doc_type": doc_type,
            "department": department,
            "chunk_index": i,
            "last_updated": last_updated
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    milvus.insert(collection_name="enterprise_kb", data=data)
    print(f"Indexed {len(chunks)} chunks from {doc_title}")

def index_folder(folder: str, department: str = ""):
    """Index all documents in a folder."""
    for file in os.listdir(folder):
        if file.endswith((".pdf", ".docx", ".txt")):
            path = os.path.join(folder, file)
            doc_id = file.rsplit(".", 1)[0]
            index_document(path, doc_id, file, department=department)
```

### Step 5: Search & Answer

```python
def search_kb(query: str, top_k: int = 5,
              doc_type: str = None, department: str = None):
    """Search the knowledge base."""
    query_embedding = embed([query])[0]

    filters = []
    if doc_type:
        filters.append(f'doc_type == "{doc_type}"')
    if department:
        filters.append(f'department == "{department}"')

    filter_expr = " and ".join(filters) if filters else None

    results = milvus.search(
        collection_name="enterprise_kb",
        data=[query_embedding],
        filter=filter_expr,
        limit=top_k,
        output_fields=["content", "doc_title", "doc_type", "department"]
    )
    return results[0]

def answer_question(question: str, department: str = None) -> str:
    """Answer a question using RAG."""
    results = search_kb(question, top_k=5, department=department)

    if not results:
        return "I couldn't find relevant information in the knowledge base."

    # Build context
    context = "\n\n".join([
        f"[{r['entity']['doc_title']}]\n{r['entity']['content'][:800]}"
        for r in results
    ])

    prompt = f"""You are an enterprise knowledge assistant. Answer based on the provided documents.

Documents:
{context}

Question: {question}

Instructions:
- Be accurate and cite the document name when possible
- If the information is not in the documents, say so
- Be concise and professional

Answer:"""

    return generate(prompt)
```

---

## Run Example

```python
# Index documents
index_document(
    "hr_policies.pdf",
    doc_id="HR-001",
    doc_title="HR Policies 2024",
    doc_type="policy",
    department="HR",
    last_updated="2024-01-01"
)

index_document(
    "expense_procedure.docx",
    doc_id="FIN-001",
    doc_title="Expense Reimbursement Procedure",
    doc_type="procedure",
    department="Finance"
)

# Answer questions
answer = answer_question("What is the vacation policy?")
print(answer)

answer = answer_question("How do I submit an expense report?", department="Finance")
print(answer)
```
