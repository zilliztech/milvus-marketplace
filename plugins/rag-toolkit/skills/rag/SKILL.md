---
name: rag
description: "Use when user wants to build RAG, Q&A system, or knowledge base with documents. Triggers on: RAG, retrieval augmented generation, Q&A system, knowledge base, document Q&A, chat with docs, ChatGPT for docs, LLM + retrieval, semantic search over documents, ground LLM with facts, reduce hallucination, enterprise search."
---

# RAG - Retrieval Augmented Generation

Build intelligent Q&A systems that ground LLM responses in your documents, reducing hallucinations and enabling knowledge updates without retraining.

## When to Activate

This skill should be activated when the user:
- Wants to build a Q&A system over their documents
- Needs to reduce LLM hallucinations with factual grounding
- Asks about "chat with docs", "ChatGPT for my data", or similar
- Wants to keep knowledge up-to-date without fine-tuning
- Needs source attribution for answers

## Interactive Flow

Before implementing, gather requirements through focused questions.

### Step 1: Understand the Use Case

Ask ONE question at a time:

```
"What type of documents will users query?"

A) Internal knowledge base (policies, procedures, docs)
B) Customer-facing FAQ / support articles
C) Technical documentation (API docs, code)
D) Mixed / other

Which one? (A/B/C/D)
```

Based on answer, follow up:

| Answer | Next Question |
|--------|---------------|
| A (Internal KB) | "Do you need access control (different users see different docs)?" |
| B (Customer FAQ) | "Do you need multi-turn conversation or single Q&A?" |
| C (Technical) | "Will queries include code snippets or just natural language?" |
| D (Mixed) | "Can you describe the main document types?" |

### Step 2: Clarify Constraints

```
"What's your priority?"

A) Accuracy first (willing to accept slower responses)
B) Speed first (good-enough answers, fast)
C) Cost first (minimize API calls)

Choose A, B, or C.
```

| Priority | Recommendation |
|----------|----------------|
| Accuracy | Add reranking → suggest `rag-with-rerank` |
| Speed | Basic RAG with caching |
| Cost | Local embeddings (BGE), smaller LLM |

### Step 3: Confirm Before Implementation

```
"Based on your requirements, I'll build:

- **Document type**: [from Step 1]
- **Chunk size**: [512 for general, 256 for FAQ, 1024 for technical]
- **Embedding**: text-embedding-3-small
- **Retrieval**: Top-5 with AUTOINDEX
- **LLM**: gpt-4o-mini

Does this look right? (yes / adjust [what])"
```

### Decision Points During Implementation

| Checkpoint | Question |
|------------|----------|
| After chunking | "I've split into X chunks. Sample: [show 2]. Chunk size OK?" |
| After indexing | "Collection created with X documents. Ready to test?" |
| After first query | "Here's a test result. Quality acceptable?" |

## Core Concepts

### The RAG Paradigm

RAG decouples **knowledge storage** from **reasoning capability**:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Documents  │───▶│  Retrieval  │───▶│    LLM      │
│  (Facts)    │    │  (Relevance)│    │  (Reasoning)│
└─────────────┘    └─────────────┘    └─────────────┘
      ▲                   │                  │
      │                   ▼                  ▼
   Update            Top-K chunks        Answer with
   anytime           as context          citations
```

**Key insight**: LLMs are excellent reasoners but unreliable knowledge stores. RAG leverages their reasoning while externalizing knowledge to a retrievable corpus.

### Mental Model: Library + Librarian

Think of RAG as a **library** (vector database) with a **librarian** (retrieval) helping a **scholar** (LLM):
- The library stores books (document chunks) indexed by topic (embeddings)
- The librarian finds relevant books based on the question
- The scholar synthesizes an answer from the provided materials

## Why RAG Over Alternatives

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **RAG** | No retraining, instant updates, source attribution | Retrieval quality limits accuracy | Dynamic knowledge, audit requirements |
| **Fine-tuning** | Deep knowledge integration | Expensive, slow updates, no citations | Stable domain expertise |
| **Long context** | Simple, no chunking | Expensive per query, 128K limit | Small corpus, one-off analysis |
| **Pure prompting** | Zero setup | Knowledge cutoff, hallucinations | General knowledge only |

**Choose RAG when**:
- Knowledge changes frequently (docs updated weekly/monthly)
- Users need source attribution ("where did you get this?")
- Corpus exceeds context window (>100K tokens)
- Domain accuracy matters more than response speed

**Avoid RAG when**:
- Corpus is tiny (<10 pages) — just use long context
- Questions don't need specific facts — pure LLM suffices
- Latency is critical (<100ms) — consider caching or fine-tuning

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INDEXING PHASE                            │
├──────────────────────────────────────────────────────────────────┤
│  Documents  ──▶  Chunking  ──▶  Embedding  ──▶  Vector Store    │
│   (raw)         (512 tokens)   (1536-dim)      (Milvus)         │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        QUERY PHASE                               │
├──────────────────────────────────────────────────────────────────┤
│  Question  ──▶  Embed  ──▶  Search  ──▶  Top-K  ──▶  LLM  ──▶  Answer
│                           (HNSW)       chunks      (GPT-4)      │
└──────────────────────────────────────────────────────────────────┘
```

### Stage Breakdown

| Stage | Purpose | Key Decision |
|-------|---------|--------------|
| **Chunking** | Split docs into retrievable units | Chunk size (see [references/chunk-strategies.md](references/chunk-strategies.md)) |
| **Embedding** | Convert text to vectors | Model choice (accuracy vs cost vs speed) |
| **Indexing** | Enable fast similarity search | Index type (HNSW for most cases) |
| **Retrieval** | Find relevant chunks | Top-K value (recall vs precision) |
| **Generation** | Synthesize answer | Prompt design, temperature |

## Implementation

Core implementation with production-ready defaults:

```python
from pymilvus import MilvusClient, DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

class RAGSystem:
    def __init__(self, collection_name: str = "rag_kb", uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.openai = OpenAI()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        self._init_collection()

    def _embed(self, texts: list) -> list:
        response = self.openai.embeddings.create(model="text-embedding-3-small", input=texts)
        return [item.embedding for item in response.data]

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
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

    def query(self, question: str, top_k: int = 5):
        # Retrieve
        results = self.client.search(self.collection_name, self._embed([question]),
                                     limit=top_k, output_fields=["text", "source"])
        contexts = [{"text": h["entity"]["text"], "source": h["entity"]["source"]} for h in results[0]]

        # Generate
        context_text = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in contexts])
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""Answer based on these references. Cite sources.

References:
{context_text}

Question: {question}"""}],
            temperature=0.3
        )
        return {"answer": response.choices[0].message.content,
                "sources": list(set(c["source"] for c in contexts))}
```

**Usage**:
```python
rag = RAGSystem()
rag.add_document(open("docs/intro.md").read(), source="intro.md")
result = rag.query("What is Milvus?")
print(result["answer"])
```

For advanced patterns (streaming, multi-turn, hybrid search), see [references/advanced-patterns.md](references/advanced-patterns.md).

## Configuration Guide

### Chunk Size Selection

| Document Type | chunk_size | overlap | Rationale |
|---------------|------------|---------|-----------|
| General docs | 512 | 50 | Balance context vs precision |
| Technical docs | 1024 | 100 | Preserve code blocks, procedures |
| FAQ | 256 | 0 | One Q&A per chunk |
| Legal/contracts | 1024 | 200 | High overlap for clause continuity |

**Rule of thumb**: Start with 512/50, adjust based on answer quality.

### Top-K Selection

| Use Case | top_k | Why |
|----------|-------|-----|
| Precise factual Q&A | 3-5 | Less noise, focused context |
| Research/synthesis | 8-12 | More perspectives |
| With reranking | 20-50 | Recall high, reranker filters |

### Embedding Model Tradeoffs

| Model | Dim | Speed | Quality | Cost |
|-------|-----|-------|---------|------|
| text-embedding-3-small | 1536 | Fast | Good | $0.02/1M |
| text-embedding-3-large | 3072 | Medium | Better | $0.13/1M |
| BAAI/bge-large-en | 1024 | Local | Good | Free |

See [references/embedding-models.md](references/embedding-models.md) for detailed comparison.

## Common Pitfalls

### 1. Chunks Too Large
**Symptom**: Irrelevant information pollutes context
**Fix**: Reduce chunk_size, or use semantic chunking

### 2. Chunks Too Small
**Symptom**: Answers lack context, feel fragmented
**Fix**: Increase chunk_size or overlap

### 3. Wrong Embedding Model for Language
**Symptom**: Poor retrieval for non-English text
**Fix**: Use multilingual model (bge-m3) or language-specific model

### 4. Ignoring Metadata
**Symptom**: Can't filter by date, source, or category
**Fix**: Store metadata fields, use filtered search

### 5. No Source Attribution
**Symptom**: Users don't trust answers
**Fix**: Always return sources, include in prompt

## When to Level Up

| Symptom | Solution | Skill |
|---------|----------|-------|
| Top results aren't the best | Add reranking | [rag-with-rerank](../rag-with-rerank/SKILL.md) |
| Complex multi-step questions | Use agentic approach | [agentic-rag](../agentic-rag/SKILL.md) |
| Questions need cross-doc reasoning | Multi-hop retrieval | [multi-hop-rag](../multi-hop-rag/SKILL.md) |

## References

**Internal**:
- [references/chunk-strategies.md](references/chunk-strategies.md) - Detailed chunking approaches
- [references/embedding-models.md](references/embedding-models.md) - Model comparison and selection
- [references/advanced-patterns.md](references/advanced-patterns.md) - Streaming, multi-turn, hybrid search

**Core operators**:
- `core:chunking` - Document chunking utilities
- `core:embedding` - Embedding generation
- `core:ray` - Data processing at scale

**Verticals**:
- [verticals/enterprise-kb.md](verticals/enterprise-kb.md) - Enterprise knowledge base
- [verticals/customer-service.md](verticals/customer-service.md) - Customer service bot
