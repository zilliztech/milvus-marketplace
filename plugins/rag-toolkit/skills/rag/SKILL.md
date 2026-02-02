---
name: rag
description: "Use when user wants to build RAG, Q&A system, or knowledge base. Triggers on: RAG, retrieval augmented generation, Q&A system, knowledge base, document Q&A, chat with docs, ChatGPT for docs, LLM + retrieval."
---

# RAG - Retrieval Augmented Generation

Build intelligent Q&A systems based on documents.

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

class RAGSystem:
    def __init__(self, collection_name: str = "rag_kb", uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.openai = OpenAI()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        self._init_collection()

    def _embed(self, texts: list) -> list:
        """Generate embeddings using OpenAI API"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1536)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256}
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add_document(self, text: str, source: str = ""):
        """Add document"""
        chunks = self.splitter.split_text(text)
        embeddings = self._embed(chunks)

        data = [
            {"text": chunk, "source": source, "embedding": emb}
            for chunk, emb in zip(chunks, embeddings)
        ]
        self.client.insert(collection_name=self.collection_name, data=data)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve relevant chunks"""
        query_embedding = self._embed([query])

        results = self.client.search(
            collection_name=self.collection_name,
            data=query_embedding,
            limit=top_k,
            output_fields=["text", "source"]
        )

        return [
            {"text": hit["entity"]["text"], "source": hit["entity"]["source"]}
            for hit in results[0]
        ]

    def generate(self, query: str, contexts: list):
        """Generate answer"""
        context_text = "\n\n".join([
            f"[Source: {c['source']}]\n{c['text']}" for c in contexts
        ])

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""Answer the question based on the following reference materials. If there is no relevant information in the materials, please state so.

Reference materials:
{context_text}

Question: {query}

Answer:"""
            }],
            temperature=0.3
        )
        return response.choices[0].message.content

    def query(self, question: str):
        """Complete Q&A workflow"""
        contexts = self.retrieve(question)
        answer = self.generate(question, contexts)
        return {
            "answer": answer,
            "sources": list(set(c["source"] for c in contexts))
        }

# Usage
rag = RAGSystem()

# Add documents
rag.add_document("Milvus is an open-source vector database...", source="milvus_intro.md")
rag.add_document("RAG enhances LLM's answering capabilities through retrieval...", source="rag_guide.md")

# Q&A
result = rag.query("What is Milvus?")
print(result["answer"])
print("Sources:", result["sources"])
```

## Optimization Tips

### 1. Reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('BAAI/bge-reranker-large')

def rerank(query: str, contexts: list, top_k: int = 3):
    pairs = [[query, c["text"]] for c in contexts]
    scores = reranker.predict(pairs)
    sorted_results = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in sorted_results[:top_k]]

# Usage
contexts = rag.retrieve(question, top_k=10)
reranked = rerank(question, contexts, top_k=5)
answer = rag.generate(question, reranked)
```

### 2. Multi-turn Conversation

```python
def query_with_history(self, question: str, history: list = None):
    # Combine historical context
    if history:
        context_prompt = "\n".join([
            f"Q: {h['q']}\nA: {h['a']}" for h in history[-3:]
        ])
        question = f"Conversation history:\n{context_prompt}\n\nCurrent question: {question}"

    return self.query(question)
```

### 3. Streaming Output

```python
def stream_generate(self, query: str, contexts: list):
    context_text = "\n\n".join([c['text'] for c in contexts])

    stream = self.openai.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": f"...{context_text}...{query}"}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

## Chunking Strategy Recommendations

| Document Type | chunk_size | overlap |
|---------------|------------|---------|
| General documents | 512 | 50 |
| Technical docs | 1024 | 100 |
| FAQ | 256 | 0 |
| Legal contracts | 1024 | 200 |

## Related Tools

- Data processing orchestration: `core:ray`
- Document chunking: `core:chunking`
- Vectorization: `core:embedding`
- With reranking: `rag-toolkit:rag-with-rerank`
