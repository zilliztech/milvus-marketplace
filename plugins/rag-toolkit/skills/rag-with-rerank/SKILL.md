---
name: rag-with-rerank
description: "Use when user needs high-precision RAG with reranking for domains where accuracy is critical. Triggers on: rag rerank, precise RAG, cross-encoder, reranking RAG, legal QA, medical QA, high-precision QA, two-stage retrieval, semantic reranking, improve RAG accuracy, relevance scoring, document ranking."
---

# RAG with Rerank

Add a reranking stage to dramatically improve retrieval precision when accuracy matters more than speed.

## When to Activate

This skill should be activated when the user:
- Needs high-precision answers (legal, medical, financial domains)
- Complains that basic RAG returns irrelevant results
- Asks about "reranking", "cross-encoder", or "two-stage retrieval"
- Wants to improve RAG accuracy without changing the corpus
- Has domain-specific terminology that vector search misses

## Interactive Flow

Reranking adds latency and cost. Confirm it's needed before implementing.

### Step 1: Validate the Need

```
"Before adding reranking, let me understand the problem:

Are you experiencing:
A) Vector search returns irrelevant results in top-5
B) Results are relevant but not in optimal order
C) Domain-specific terms aren't matching well
D) Not sure, just want higher accuracy

Which one? (A/B/C/D)"
```

| Answer | Recommendation |
|--------|----------------|
| A (Irrelevant results) | May need better chunking first. "Can you show me an example query that fails?" |
| B (Wrong order) | Reranking is the right solution |
| C (Term mismatch) | Consider hybrid search first, then rerank |
| D (General improvement) | "What's your current accuracy? Reranking typically improves 10-20%" |

### Step 2: Domain and Language

```
"What's the primary language of your documents?"

A) English
B) Chinese
C) Mixed / Multilingual
D) Other: ___

This determines which reranker model to use.
```

| Language | Recommended Reranker |
|----------|---------------------|
| English | `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast) or `bge-reranker-v2-m3` (quality) |
| Chinese | `BAAI/bge-reranker-large` |
| Mixed | `BAAI/bge-reranker-v2-m3` |

### Step 3: Latency vs Accuracy Tradeoff

```
"What's your latency budget per query?"

A) <200ms (speed critical)
B) 200-500ms (balanced)
C) >500ms OK (accuracy critical)

Current RAG is ~100ms. Reranking adds 50-150ms depending on model.
```

| Budget | Configuration |
|--------|---------------|
| <200ms | Fast reranker (MiniLM), retrieve_k=30 |
| 200-500ms | Standard (bge-reranker-large), retrieve_k=50 |
| >500ms | Best quality (bge-reranker-v2-m3), retrieve_k=100 |

### Step 4: Confirm Configuration

```
"Proposed configuration:

- **Reranker**: [model from Step 2]
- **Retrieve K**: [from Step 3] candidates
- **Rerank K**: 5 final results
- **Expected latency**: +[X]ms over basic RAG

Proceed? (yes / adjust [what])"
```

### Red Flags - When NOT to Rerank

Before implementing, check these:

| Symptom | Better Solution |
|---------|----------------|
| "All results are irrelevant" | Fix chunking or embeddings first |
| Corpus < 100 documents | Skip reranking, not enough to matter |
| Latency budget < 150ms | Use hybrid search instead |
| "Results are fine, want perfect" | Diminishing returns warning |

```
"I notice [symptom]. Reranking may not be the best solution here.
Should we try [alternative] first? (yes / proceed anyway)"
```

## Core Concepts

### The Precision Problem

Vector search (bi-encoder) trades precision for speed:

```
Bi-Encoder (Fast, Less Precise):
Query  ──▶ [Encoder] ──▶ Query Vector  ─┐
                                        ├──▶ Cosine Similarity
Doc    ──▶ [Encoder] ──▶ Doc Vector   ─┘

Cross-Encoder (Slow, More Precise):
[Query, Doc] ──▶ [Joint Encoder] ──▶ Relevance Score
```

**Key insight**: Bi-encoders encode query and document independently — they can't capture fine-grained interactions. Cross-encoders see both together, enabling deeper semantic matching.

### Two-Stage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: RECALL (Fast, High Volume)                             │
│ Query ──▶ Vector Search ──▶ Top 50 candidates                   │
│ Purpose: Cast a wide net, don't miss relevant docs              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: RERANK (Slow, High Precision)                          │
│ 50 candidates ──▶ Cross-Encoder ──▶ Top 5 results               │
│ Purpose: Fine-grained relevance scoring                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: GENERATE                                               │
│ Top 5 ──▶ LLM ──▶ Answer                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Mental Model: Hiring Process

Think of it like hiring:
- **Stage 1 (Vector Search)** = Resume screening: Quick filter, high recall, some false positives
- **Stage 2 (Rerank)** = Phone interview: Deeper evaluation, removes false positives
- **Stage 3 (Generate)** = Final interview: Make decision with best candidates

## Why Rerank Over Alternatives

| Approach | Precision | Latency | Cost | When to Use |
|----------|-----------|---------|------|-------------|
| Basic RAG | Medium | Low | Low | General Q&A, speed matters |
| **RAG + Rerank** | High | Medium | Medium | Accuracy-critical domains |
| Hybrid Search | Medium-High | Low | Low | Keyword matching important |
| Fine-tuned Embeddings | High | Low | High (training) | Large scale, stable domain |

**Choose Rerank when**:
- Accuracy is non-negotiable (legal, medical, compliance)
- Vector search returns "close but not quite" results
- Domain has specific terminology or jargon
- You can tolerate 100-200ms extra latency

**Skip Rerank when**:
- Latency budget < 200ms
- Basic RAG already achieves >90% precision
- Cost per query is critical constraint

## Implementation

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
        if isinstance(texts, str):
            texts = [texts]
        response = self.openai.embeddings.create(model="text-embedding-3-small", input=texts)
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
        index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(self.collection_name, schema=schema, index_params=index_params)

    def add_document(self, text: str, source: str = ""):
        chunks = self.splitter.split_text(text)
        embeddings = self._embed(chunks)
        data = [{"text": c, "source": source, "embedding": e} for c, e in zip(chunks, embeddings)]
        self.client.insert(self.collection_name, data)
        return len(chunks)

    def query(self, question: str, retrieve_k: int = 50, rerank_k: int = 5):
        # Stage 1: Recall
        embedding = self._embed(question)[0]
        results = self.client.search(self.collection_name, [embedding], limit=retrieve_k,
                                     output_fields=["text", "source"])
        candidates = [{"text": h["entity"]["text"], "source": h["entity"]["source"]} for h in results[0]]

        # Stage 2: Rerank
        if candidates:
            pairs = [[question, c["text"]] for c in candidates]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            reranked = [item[0] for item in ranked[:rerank_k]]
        else:
            reranked = []

        # Stage 3: Generate
        context_text = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in reranked])
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""Answer based on these references. Be precise and cite sources.

References:
{context_text}

Question: {question}"""}],
            temperature=0.1
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": list(set(c["source"] for c in reranked)),
            "rerank_scores": [s for _, s in ranked[:rerank_k]] if candidates else []
        }
```

**Usage**:
```python
rag = RAGWithRerank()
rag.add_document(open("legal_docs/civil_code.txt").read(), source="Civil Code")
result = rag.query("What are the conditions for divorce?")
```

## Configuration Guide

### Retrieve K vs Rerank K

| Scenario | retrieve_k | rerank_k | Rationale |
|----------|------------|----------|-----------|
| High precision | 100 | 3 | Wide net, strict filter |
| Balanced | 50 | 5 | Good default |
| Speed priority | 30 | 5 | Less reranking work |
| Research/synthesis | 50 | 10 | More diverse context |

**Rule of thumb**: `retrieve_k` should be 10-20x `rerank_k`

### Reranker Model Selection

| Model | Language | Latency | Quality | Use Case |
|-------|----------|---------|---------|----------|
| BAAI/bge-reranker-large | Chinese/English | 50ms | High | Chinese, general |
| BAAI/bge-reranker-v2-m3 | Multilingual | 100ms | Highest | Multilingual, quality-first |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | English | 20ms | Good | English, speed-first |
| Cohere rerank-v3.0 | Multilingual | 80ms | High | API-based, no GPU |

See [references/reranker-comparison.md](references/reranker-comparison.md) for detailed benchmarks.

### Temperature Setting

| Domain | Temperature | Why |
|--------|-------------|-----|
| Legal | 0.0-0.1 | Precision critical, no creativity |
| Medical | 0.0-0.1 | Safety critical |
| Technical | 0.2-0.3 | Some flexibility for explanation |
| General | 0.3-0.5 | Balance |

## Common Pitfalls

### 1. Retrieve K Too Low
**Symptom**: Reranker can't find good results because recall missed them
**Fix**: Increase `retrieve_k` to 50-100

### 2. Retrieve K Too High
**Symptom**: Slow reranking, high latency
**Fix**: Reduce `retrieve_k` or use faster reranker model

### 3. Wrong Reranker Language
**Symptom**: Reranking doesn't improve (or hurts) quality
**Fix**: Match reranker to corpus language (bge-reranker-large for Chinese)

### 4. Reranking Tiny Corpus
**Symptom**: Wasted compute, no improvement
**Fix**: If corpus < 100 docs, skip reranking

### 5. Ignoring Rerank Scores
**Symptom**: Including irrelevant results with low scores
**Fix**: Add score threshold (e.g., only keep score > 0.5)

```python
# Add score filtering
threshold = 0.5
reranked = [(c, s) for c, s in zip(candidates, scores) if s > threshold]
reranked = sorted(reranked, key=lambda x: x[1], reverse=True)[:rerank_k]
```

## Performance Optimization

### Batch Reranking

```python
def batch_rerank(self, queries: list[str], candidates_list: list[list[dict]]):
    """Rerank multiple queries efficiently"""
    all_pairs = []
    indices = []
    for i, (query, candidates) in enumerate(zip(queries, candidates_list)):
        for c in candidates:
            all_pairs.append([query, c["text"]])
            indices.append(i)

    # Single batch prediction
    all_scores = self.reranker.predict(all_pairs)

    # Reconstruct results
    results = [[] for _ in queries]
    for idx, score, pair in zip(indices, all_scores, all_pairs):
        results[idx].append((pair[1], score))

    return [sorted(r, key=lambda x: x[1], reverse=True) for r in results]
```

### Caching Rerank Results

```python
import hashlib

class CachedReranker:
    def __init__(self, reranker):
        self.reranker = reranker
        self.cache = {}

    def predict(self, pairs):
        results = []
        to_compute = []
        indices = []

        for i, (query, doc) in enumerate(pairs):
            key = hashlib.md5(f"{query}|||{doc}".encode()).hexdigest()
            if key in self.cache:
                results.append(self.cache[key])
            else:
                to_compute.append((query, doc))
                indices.append(i)
                results.append(None)

        if to_compute:
            scores = self.reranker.predict(to_compute)
            for idx, (pair, score) in zip(indices, zip(to_compute, scores)):
                key = hashlib.md5(f"{pair[0]}|||{pair[1]}".encode()).hexdigest()
                self.cache[key] = score
                results[idx] = score

        return results
```

## When to Level Up

| Symptom | Solution | Skill |
|---------|----------|-------|
| Questions need multi-step reasoning | Multi-hop retrieval | [multi-hop-rag](../multi-hop-rag/SKILL.md) |
| Need dynamic, conversational retrieval | Agentic approach | [agentic-rag](../agentic-rag/SKILL.md) |
| Keyword matching important | Hybrid search | See [rag/references/advanced-patterns.md](../rag/references/advanced-patterns.md) |

## References

**Internal**:
- [references/reranker-comparison.md](references/reranker-comparison.md) - Model benchmarks and selection guide

**Related skills**:
- [rag](../rag/SKILL.md) - Basic RAG (when speed > precision)
- [agentic-rag](../agentic-rag/SKILL.md) - Agent-driven retrieval
- `core:rerank` - Reranking utilities

**Verticals**:
- [verticals/legal.md](verticals/legal.md) - Legal consultation
