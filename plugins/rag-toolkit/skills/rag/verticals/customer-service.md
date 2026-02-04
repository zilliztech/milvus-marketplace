# Customer Service Knowledge Base

> Build a customer service chatbot with FAQ, documentation, and ticket history.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Knowledge Language

<ask_user>
What language is your knowledge base in?

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
| **OpenAI API** | High quality, fast | Requires API key |
| **Local Model** | Free, offline | Model download |
</ask_user>

### 3. LLM for Generation

<ask_user>
Choose LLM for answer generation:

| Model | Notes |
|-------|-------|
| **GPT-4o-mini** (recommended) | Cost-effective, fast |
| **GPT-4o** | Highest quality |
| **Local LLM** | Ollama, vLLM |
</ask_user>

### 4. Data Scale

<ask_user>
How much knowledge do you have?

- FAQ: ~1 vector per Q&A pair
- Docs: ~50-100 chunks per document
- Tickets: ~1 chunk per resolved ticket

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
uv init customer-service-rag
cd customer-service-rag
uv add pymilvus openai
# Or for local embedding:
uv add pymilvus sentence-transformers openai
```

---

## End-to-End Implementation

### Step 1: Configure Embedding & LLM

```python
from openai import OpenAI

client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

def generate_answer(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("customer_service.db")

schema = milvus.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("kb_type", DataType.VARCHAR, max_length=32)  # faq/doc/ticket
schema.add_field("category", DataType.VARCHAR, max_length=64)
schema.add_field("product", DataType.VARCHAR, max_length=128)
schema.add_field("question", DataType.VARCHAR, max_length=512)  # For FAQ
schema.add_field("answer", DataType.VARCHAR, max_length=65535)  # For FAQ

index_params = milvus.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

milvus.create_collection("customer_service", schema=schema, index_params=index_params)
```

### Step 3: Index Knowledge

```python
def add_faq(faqs: list[dict]):
    """Add FAQ entries. faqs: [{"question": "...", "answer": "...", "category": "..."}]"""
    questions = [f["question"] for f in faqs]
    embeddings = embed(questions)

    data = [
        {
            "embedding": emb,
            "content": f["question"],
            "kb_type": "faq",
            "category": f.get("category", ""),
            "product": f.get("product", ""),
            "question": f["question"],
            "answer": f["answer"]
        }
        for f, emb in zip(faqs, embeddings)
    ]
    milvus.insert(collection_name="customer_service", data=data)

def add_docs(docs: list[dict]):
    """Add documentation. docs: [{"content": "...", "product": "..."}]"""
    contents = [d["content"] for d in docs]
    embeddings = embed(contents)

    data = [
        {
            "embedding": emb,
            "content": d["content"][:5000],
            "kb_type": "doc",
            "category": d.get("category", ""),
            "product": d.get("product", ""),
            "question": "",
            "answer": ""
        }
        for d, emb in zip(docs, embeddings)
    ]
    milvus.insert(collection_name="customer_service", data=data)

def add_tickets(tickets: list[dict]):
    """Add resolved tickets. tickets: [{"issue": "...", "resolution": "..."}]"""
    issues = [t["issue"] for t in tickets]
    embeddings = embed(issues)

    data = [
        {
            "embedding": emb,
            "content": t["issue"],
            "kb_type": "ticket",
            "category": t.get("category", ""),
            "product": t.get("product", ""),
            "question": "",
            "answer": t["resolution"]
        }
        for t, emb in zip(tickets, embeddings)
    ]
    milvus.insert(collection_name="customer_service", data=data)
```

### Step 4: Multi-Source Retrieval

```python
def search_faq(query: str, limit: int = 3):
    embedding = embed([query])[0]
    return milvus.search(
        collection_name="customer_service",
        data=[embedding],
        filter='kb_type == "faq"',
        limit=limit,
        output_fields=["question", "answer"]
    )[0]

def search_docs(query: str, product: str = None, limit: int = 3):
    embedding = embed([query])[0]
    filter_expr = 'kb_type == "doc"'
    if product:
        filter_expr += f' and product == "{product}"'

    return milvus.search(
        collection_name="customer_service",
        data=[embedding],
        filter=filter_expr,
        limit=limit,
        output_fields=["content", "product"]
    )[0]

def search_tickets(query: str, limit: int = 2):
    embedding = embed([query])[0]
    return milvus.search(
        collection_name="customer_service",
        data=[embedding],
        filter='kb_type == "ticket"',
        limit=limit,
        output_fields=["content", "answer"]
    )[0]
```

### Step 5: Generate Answer

```python
def answer_question(question: str, product: str = None) -> dict:
    """Generate answer from multiple knowledge sources."""
    # Search FAQ first
    faq_results = search_faq(question, limit=3)

    # High-confidence FAQ match
    if faq_results and faq_results[0]["distance"] > 0.9:
        return {
            "answer": faq_results[0]["entity"]["answer"],
            "source": "faq",
            "confidence": "high"
        }

    # Search docs and tickets
    doc_results = search_docs(question, product, limit=3)
    ticket_results = search_tickets(question, limit=2)

    # Build context
    context_parts = []

    for r in faq_results:
        context_parts.append(f"FAQ: {r['entity']['question']}\nAnswer: {r['entity']['answer']}")

    for r in doc_results:
        context_parts.append(f"Documentation: {r['entity']['content'][:500]}")

    for r in ticket_results:
        context_parts.append(f"Similar Issue: {r['entity']['content']}\nResolution: {r['entity']['answer']}")

    context = "\n\n".join(context_parts)

    # Generate with LLM
    prompt = f"""You are a helpful customer service assistant. Answer based on the reference materials.

Reference Materials:
{context}

Customer Question: {question}

Instructions:
- Be concise and helpful
- If unsure, suggest contacting human support
- Provide next steps if applicable

Answer:"""

    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "source": "rag",
        "confidence": "medium"
    }
```

---

## Run Example

```python
# Add FAQ
faqs = [
    {"question": "How to reset password?", "answer": "Go to Settings > Security > Reset Password...", "category": "Account"},
    {"question": "What is the return policy?", "answer": "30-day return for unopened items...", "category": "Orders"},
]
add_faq(faqs)

# Add documentation
docs = [
    {"content": "To set up your device, first unbox and connect the power cable...", "product": "Smart Speaker"},
]
add_docs(docs)

# Answer questions
result = answer_question("How do I reset my password?")
print(f"Answer: {result['answer']}")
print(f"Source: {result['source']}, Confidence: {result['confidence']}")

result = answer_question("My speaker won't turn on", product="Smart Speaker")
print(f"Answer: {result['answer']}")
```
