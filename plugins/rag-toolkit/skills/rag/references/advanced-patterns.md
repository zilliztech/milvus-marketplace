# Advanced RAG Patterns

This reference covers production patterns beyond basic RAG: streaming, multi-turn conversations, hybrid search, and query enhancement.

## Streaming Output

Stream answers token-by-token for better UX:

```python
def stream_generate(self, query: str, contexts: list):
    """Stream answer generation for real-time display"""
    context_text = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in contexts])

    stream = self.openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Answer based on these references. Cite sources.

References:
{context_text}

Question: {query}"""
        }],
        temperature=0.3,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Usage with FastAPI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/query")
async def query_stream(q: str):
    rag = RAGSystem()
    contexts = rag.retrieve(q)

    def generate():
        for token in rag.stream_generate(q, contexts):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")
```

## Multi-Turn Conversation

Maintain conversation history for context-aware Q&A:

```python
class ConversationalRAG:
    def __init__(self, rag: RAGSystem, max_history: int = 5):
        self.rag = rag
        self.history = []
        self.max_history = max_history

    def query(self, question: str) -> dict:
        # Rewrite query with conversation context
        if self.history:
            rewritten = self._rewrite_query(question)
        else:
            rewritten = question

        # Retrieve with rewritten query
        contexts = self.rag.retrieve(rewritten)

        # Generate with full history
        answer = self._generate_with_history(question, contexts)

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

        return {"answer": answer, "rewritten_query": rewritten}

    def _rewrite_query(self, question: str) -> str:
        """Rewrite question to be standalone using conversation context"""
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in self.history[-4:]  # Last 2 turns
        ])

        response = self.rag.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Rewrite the follow-up question to be standalone.

Conversation:
{history_text}

Follow-up: {question}

Standalone question:"""
            }],
            temperature=0
        )
        return response.choices[0].message.content

    def _generate_with_history(self, question: str, contexts: list) -> str:
        """Generate answer considering conversation history"""
        context_text = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in contexts])

        messages = [
            {"role": "system", "content": f"""You are a helpful assistant. Answer based on these references:

{context_text}

If information is not in the references, say so."""}
        ]
        messages.extend(self.history)
        messages.append({"role": "user", "content": question})

        response = self.rag.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content

    def reset(self):
        """Clear conversation history"""
        self.history = []
```

## Hybrid Search

Combine vector similarity with keyword matching:

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType

class HybridRAG:
    def __init__(self, collection_name: str = "hybrid_kb", uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.openai = OpenAI()
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        # BM25 function for sparse vectors
        bm25_fn = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"]
        )
        schema.add_function(bm25_fn)

        # Index for dense vectors
        index_params = self.client.prepare_index_params()
        index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

        self.client.create_collection(self.collection_name, schema=schema, index_params=index_params)

    def retrieve(self, query: str, top_k: int = 5, dense_weight: float = 0.7):
        """Hybrid retrieval combining dense and sparse search"""
        from pymilvus import AnnSearchRequest, RRFRanker

        query_embedding = self._embed([query])

        # Dense search request
        dense_req = AnnSearchRequest(
            data=query_embedding,
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=top_k * 2
        )

        # Sparse search request (BM25)
        sparse_req = AnnSearchRequest(
            data=[query],  # Raw text for BM25
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=top_k * 2
        )

        # Combine with RRF (Reciprocal Rank Fusion)
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),  # k is RRF parameter
            limit=top_k,
            output_fields=["text", "source"]
        )

        return [
            {"text": hit["entity"]["text"], "source": hit["entity"]["source"]}
            for hit in results[0]
        ]
```

**When to use hybrid search**:
- Exact keyword matching matters (product names, codes, IDs)
- Mixed queries (some need semantics, some need keywords)
- Regulatory/legal domains where specific terms must match

## Query Enhancement

### Query Expansion

Expand query with synonyms and related terms:

```python
def expand_query(self, query: str) -> str:
    """Generate expanded query with related terms"""
    response = self.openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Expand this search query with synonyms and related terms.
Keep it as a single search query, not multiple queries.

Original: {query}

Expanded:"""
        }],
        temperature=0.3
    )
    return response.choices[0].message.content

# Example: "ML training" → "machine learning model training neural network optimization"
```

### HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer, then search for similar content:

```python
def hyde_retrieve(self, query: str, top_k: int = 5):
    """HyDE: Generate hypothetical answer, then retrieve similar content"""
    # Generate hypothetical answer
    response = self.openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Write a short paragraph that would answer this question.
Don't say you don't know - write what a good answer would look like.

Question: {query}

Answer paragraph:"""
        }],
        temperature=0.7
    )
    hypothetical = response.choices[0].message.content

    # Embed the hypothetical answer (not the query)
    hyde_embedding = self._embed([hypothetical])

    # Search with hypothetical answer embedding
    results = self.client.search(
        collection_name=self.collection_name,
        data=hyde_embedding,
        limit=top_k,
        output_fields=["text", "source"]
    )

    return [
        {"text": h["entity"]["text"], "source": h["entity"]["source"]}
        for h in results[0]
    ]
```

**When to use HyDE**:
- Questions are phrased very differently from documents
- Documents are dense/technical, questions are casual
- Standard retrieval misses obvious matches

## Metadata Filtering

Filter search results by metadata before or during retrieval:

```python
def filtered_retrieve(self, query: str, filters: dict, top_k: int = 5):
    """Retrieve with metadata filters"""
    query_embedding = self._embed([query])

    # Build filter expression
    filter_parts = []
    for key, value in filters.items():
        if isinstance(value, str):
            filter_parts.append(f'{key} == "{value}"')
        elif isinstance(value, list):
            values = ", ".join([f'"{v}"' for v in value])
            filter_parts.append(f'{key} in [{values}]')
        elif isinstance(value, dict):
            if "gte" in value:
                filter_parts.append(f'{key} >= {value["gte"]}')
            if "lte" in value:
                filter_parts.append(f'{key} <= {value["lte"]}')

    filter_expr = " and ".join(filter_parts) if filter_parts else ""

    results = self.client.search(
        collection_name=self.collection_name,
        data=query_embedding,
        filter=filter_expr,
        limit=top_k,
        output_fields=["text", "source", "category", "date"]
    )

    return [hit["entity"] for hit in results[0]]

# Usage
results = rag.filtered_retrieve(
    "security best practices",
    filters={
        "category": "documentation",
        "date": {"gte": 20240101},  # After Jan 1, 2024
    }
)
```

## Evaluation and Monitoring

### Retrieval Quality Metrics

```python
def evaluate_retrieval(rag, test_set):
    """
    test_set = [
        {"query": "What is X?", "relevant_chunks": ["chunk_id_1", "chunk_id_2"]},
        ...
    ]
    """
    metrics = {"recall@3": [], "recall@5": [], "mrr": []}

    for case in test_set:
        results = rag.retrieve(case["query"], top_k=5)
        retrieved_ids = [r["id"] for r in results]
        relevant = set(case["relevant_chunks"])

        # Recall@K
        for k in [3, 5]:
            hits = len(set(retrieved_ids[:k]) & relevant)
            metrics[f"recall@{k}"].append(hits / len(relevant))

        # MRR (Mean Reciprocal Rank)
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant:
                metrics["mrr"].append(1 / (i + 1))
                break
        else:
            metrics["mrr"].append(0)

    return {k: sum(v) / len(v) for k, v in metrics.items()}
```

### Answer Quality Evaluation

```python
def evaluate_answer(query: str, answer: str, reference: str) -> dict:
    """Use LLM to evaluate answer quality"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Evaluate this answer on a scale of 1-5 for each criterion.

Question: {query}
Answer: {answer}
Reference (ground truth): {reference}

Rate:
1. Factual Accuracy (1-5): Does the answer contain correct facts?
2. Completeness (1-5): Does it cover all key points from reference?
3. Relevance (1-5): Is the answer focused on the question?
4. Clarity (1-5): Is the answer clear and well-organized?

Respond in JSON: {{"accuracy": N, "completeness": N, "relevance": N, "clarity": N}}"""
        }],
        temperature=0
    )
    return json.loads(response.choices[0].message.content)
```

### Logging and Monitoring

```python
import logging
import time
from dataclasses import dataclass

@dataclass
class QueryLog:
    query: str
    rewritten_query: str
    num_results: int
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    sources: list[str]

class MonitoredRAG:
    def __init__(self, rag: RAGSystem):
        self.rag = rag
        self.logger = logging.getLogger("rag")

    def query(self, question: str) -> dict:
        start = time.time()

        # Retrieval
        retrieve_start = time.time()
        contexts = self.rag.retrieve(question)
        retrieve_time = (time.time() - retrieve_start) * 1000

        # Generation
        generate_start = time.time()
        result = self.rag.generate(question, contexts)
        generate_time = (time.time() - generate_start) * 1000

        total_time = (time.time() - start) * 1000

        # Log
        log = QueryLog(
            query=question,
            rewritten_query=question,
            num_results=len(contexts),
            retrieval_time_ms=retrieve_time,
            generation_time_ms=generate_time,
            total_time_ms=total_time,
            sources=[c["source"] for c in contexts]
        )
        self.logger.info(f"Query completed: {log}")

        return {"answer": result, "metrics": log.__dict__}
```

## Production Checklist

- [ ] **Streaming** enabled for better UX
- [ ] **Conversation history** for multi-turn
- [ ] **Hybrid search** if keywords matter
- [ ] **Metadata filtering** for access control
- [ ] **Query logging** for debugging
- [ ] **Evaluation set** for quality monitoring
- [ ] **Caching** for repeated queries
- [ ] **Rate limiting** for API costs
- [ ] **Error handling** for API failures
- [ ] **Fallback** when retrieval returns nothing
