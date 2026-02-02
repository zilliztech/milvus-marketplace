---
name: rag-with-rerank
description: "Use when user needs high-precision RAG with reranking. Triggers on: rag rerank, precise RAG, cross-encoder, reranking RAG, legal QA, medical QA, high-precision QA."
---

# RAG with Rerank - RAG with Reranking

Add a Rerank stage on top of basic RAG to significantly improve answer precision.

## Use Cases

- Legal consultation (requires precise law matching)
- Medical QA (no room for error)
- Financial research (high precision requirements)
- Any scenario requiring high answer accuracy

## Architecture

```
Query → Vector Retrieval(Top 50) → Rerank(Top 5) → LLM Generation → Answer
```

## Why Rerank is Needed

| Stage | Method | Speed | Precision |
|-------|--------|-------|-----------|
| Recall | Vector Search (Bi-Encoder) | Fast | Medium |
| Rerank | Rerank (Cross-Encoder) | Slow | High |

Cross-Encoder encodes query and document together, capturing finer-grained interactions.

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import CrossEncoder
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGWithRerank:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()
        self.reranker = CrossEncoder('BAAI/bge-reranker-large')
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        self.collection_name = "rag_rerank"
        self._init_collection()

    def _embed(self, texts: list) -> list:
        """Generate embeddings using OpenAI API"""
        if isinstance(texts, str):
            texts = [texts]
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 256})

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add_document(self, text: str, source: str = ""):
        """Add document"""
        chunks = self.splitter.split_text(text)
        embeddings = self._embed(chunks)
        data = [{"text": c, "source": source, "embedding": e} for c, e in zip(chunks, embeddings)]
        self.client.insert(collection_name=self.collection_name, data=data)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 50):
        """Stage 1: Vector recall"""
        embedding = self._embed(query)[0]
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=top_k,
            output_fields=["text", "source"]
        )
        return [{"text": hit["entity"]["text"], "source": hit["entity"]["source"]}
                for hit in results[0]]

    def rerank(self, query: str, candidates: list, top_k: int = 5):
        """Stage 2: Reranking"""
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        # Sort and take top_k
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [item[0] for item in ranked[:top_k]]

    def generate(self, query: str, contexts: list):
        """Stage 3: LLM generation"""
        context_text = "\n\n".join([
            f"[Source: {c['source']}]\n{c['text']}" for c in contexts
        ])

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""Answer the question based on the following references. Ensure accuracy, and indicate if uncertain.

References:
{context_text}

Question: {query}

Answer:"""
            }],
            temperature=0.1  # Low temperature, more deterministic
        )
        return response.choices[0].message.content

    def query(self, question: str, retrieve_k: int = 50, rerank_k: int = 5):
        """Complete QA pipeline"""
        # 1. Recall
        candidates = self.retrieve(question, top_k=retrieve_k)

        # 2. Rerank
        reranked = self.rerank(question, candidates, top_k=rerank_k)

        # 3. Generate
        answer = self.generate(question, reranked)

        return {
            "answer": answer,
            "sources": list(set(c["source"] for c in reranked)),
            "contexts": reranked
        }

# Usage
rag = RAGWithRerank()

# Add legal documents
rag.add_document("Section 1076 of the Civil Code: Both spouses voluntarily agree to divorce...", source="Civil Code")
rag.add_document("Article 40 of the Labor Law: Under any of the following circumstances, the employer may terminate the labor contract...", source="Labor Law")

# High-precision QA
result = rag.query("What are the conditions for divorce?")
print(result["answer"])
print("Sources:", result["sources"])
```

## Rerank Model Selection

| Model | Language | Features |
|-------|----------|----------|
| BAAI/bge-reranker-large | Chinese | Best for Chinese, recommended |
| BAAI/bge-reranker-v2-m3 | Multilingual | Best performance, slower |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | English | English, fast |

## Parameter Tuning

| Parameter | Suggested Value | Description |
|-----------|----------------|-------------|
| retrieve_k | 30-100 | Recall count, larger = more complete |
| rerank_k | 3-10 | Count kept after reranking |
| chunk_size | 256-512 | Use smaller chunks for precision |

```python
# High precision scenario
result = rag.query(question, retrieve_k=100, rerank_k=3)

# Balanced scenario
result = rag.query(question, retrieve_k=50, rerank_k=5)
```

## Vertical Applications

See detailed guides in `verticals/` directory:
- `legal.md` - Legal consultation
- `medical.md` - Medical QA
- `financial.md` - Financial research

## Related Tools

- Data processing orchestration: `core:ray`
- Reranking: `core:rerank`
- Document chunking: `core:chunking`
- Basic RAG: `rag-toolkit:rag`
