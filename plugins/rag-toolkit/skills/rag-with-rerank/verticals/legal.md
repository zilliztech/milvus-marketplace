# Legal Consultation RAG with Rerank

> Build a legal knowledge base with precision reranking for accurate answers.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Legal Document Language

<ask_user>
What language are your legal documents in?

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
| **Local Model** | Free, data stays local | Model download |

Note: For sensitive legal data, local models may be preferred.
</ask_user>

### 3. Reranker Selection

<ask_user>
Choose reranker for precision:

| Model | Notes |
|-------|-------|
| `BAAI/bge-reranker-large` | Best for English |
| `BAAI/bge-reranker-v2-m3` | Multilingual |
| `Cohere rerank-v3.5` | API-based, high quality |
</ask_user>

### 4. LLM for Generation

<ask_user>
Choose LLM for answer generation:

| Model | Notes |
|-------|-------|
| **GPT-4o** | Best quality for legal |
| **GPT-4o-mini** | Cost-effective |
| **Claude 3** | Good for nuanced answers |
</ask_user>

### 5. Data Scale

<ask_user>
How many legal documents do you have?

- Each document ≈ 50-200 chunks
- Example: 1000 documents ≈ 100K vectors

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
uv init legal-rag
cd legal-rag
uv add pymilvus openai sentence-transformers
```

---

## Why Rerank for Legal

Legal consultation requires extremely high accuracy:
- **One word difference, different meaning** - Legal terminology requires precision
- **Context dependent** - Need to understand query and answer correlation
- **Recall ≠ relevance** - Vector similarity doesn't equal legal relevance

---

## End-to-End Implementation

### Step 1: Configure Models

```python
from openai import OpenAI
from sentence_transformers import CrossEncoder

client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

# Reranker
reranker = CrossEncoder('BAAI/bge-reranker-large')

def generate(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # Low temperature for accuracy
    )
    return resp.choices[0].message.content
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("legal_kb.db")

schema = milvus.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("doc_type", DataType.VARCHAR, max_length=32)      # law/regulation/case/opinion
schema.add_field("law_category", DataType.VARCHAR, max_length=64)  # civil/criminal/labor
schema.add_field("effectiveness", DataType.VARCHAR, max_length=32) # valid/invalid/amended
schema.add_field("source", DataType.VARCHAR, max_length=256)

index_params = milvus.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

milvus.create_collection("legal_kb", schema=schema, index_params=index_params)
```

### Step 3: RAG with Rerank Pipeline

```python
def retrieve(query: str, law_category: str = None, limit: int = 30):
    """Initial recall - get more candidates for reranking."""
    embedding = embed([query])[0]

    filter_expr = 'effectiveness == "valid"'
    if law_category:
        filter_expr += f' and law_category == "{law_category}"'

    results = milvus.search(
        collection_name="legal_kb",
        data=[embedding],
        filter=filter_expr,
        limit=limit,
        output_fields=["content", "doc_type", "source", "law_category"]
    )

    return results[0]

def rerank(query: str, candidates: list, top_k: int = 5):
    """Precision reranking with CrossEncoder."""
    pairs = [[query, c["entity"]["content"]] for c in candidates]
    scores = reranker.predict(pairs)

    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]

def answer_legal_question(question: str, law_category: str = None) -> dict:
    """Complete RAG + Rerank pipeline."""
    # 1. Recall (more candidates)
    candidates = retrieve(question, law_category, limit=30)

    # 2. Rerank (select best)
    reranked = rerank(question, candidates, top_k=8)

    # 3. Organize context by type
    laws = [c for c in reranked if c["entity"]["doc_type"] == "law"]
    cases = [c for c in reranked if c["entity"]["doc_type"] == "case"]
    opinions = [c for c in reranked if c["entity"]["doc_type"] == "opinion"]

    context_text = ""
    if laws:
        context_text += "[Relevant Law Articles]\n"
        for l in laws:
            context_text += f"- {l['entity']['source']}: {l['entity']['content']}\n"

    if cases:
        context_text += "\n[Relevant Cases]\n"
        for c in cases:
            context_text += f"- {c['entity']['source']}: {c['entity']['content'][:500]}...\n"

    if opinions:
        context_text += "\n[Legal Interpretations]\n"
        for o in opinions:
            context_text += f"- {o['entity']['source']}: {o['entity']['content']}\n"

    # 4. Generate answer
    prompt = f"""You are a professional legal consultant. Answer based on the legal materials below.

{context_text}

Question: {question}

Requirements:
1. Cite specific law articles with names and numbers
2. If relevant cases exist, briefly explain key points
3. Distinguish between definitive opinions and advisory suggestions
4. Recommend consulting a lawyer for complex matters

Answer:"""

    answer = generate(prompt)
    sources = list(set(c["entity"]["source"] for c in reranked))

    return {
        "answer": answer,
        "sources": sources,
        "contexts": [{"source": c["entity"]["source"],
                     "content": c["entity"]["content"][:200] + "...",
                     "type": c["entity"]["doc_type"],
                     "rerank_score": c["rerank_score"]}
                    for c in reranked]
    }
```

---

## Run Example

```python
# Index legal documents
milvus.insert(collection_name="legal_kb", data=[{
    "embedding": embed(["Labor Contract Law Article 82..."])[0],
    "content": "If an employer fails to conclude a written labor contract with an employee for more than one month...",
    "doc_type": "law",
    "law_category": "labor",
    "effectiveness": "valid",
    "source": "Labor Contract Law Article 82"
}])

# Query with reranking
result = answer_legal_question(
    "Can I claim double wages if the company didn't sign a labor contract?",
    law_category="labor"
)

print(f"Answer: {result['answer']}")
print(f"\nSources:")
for src in result['sources']:
    print(f"  - {src}")
```

---

## Important Notes

1. **Timeliness**: Laws get amended, ensure database is updated
2. **Jurisdiction**: Different regions may have different regulations
3. **Disclaimer**: AI answers are for reference only, consult lawyers for important matters
4. **Privacy**: Don't log sensitive case information
