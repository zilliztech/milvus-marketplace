---
name: multi-hop-rag
description: "Use when user needs multi-step reasoning with iterative retrieval for complex questions. Triggers on: multi-hop, multi-step RAG, complex questions, chain of retrieval, iterative retrieval, complex reasoning, cross-document reasoning, question decomposition, research questions, fact synthesis, connecting information across documents."
---

# Multi-Hop RAG

Answer complex questions that require connecting information from multiple documents through iterative retrieval — each retrieval step informs the next.

## When to Activate

This skill should be activated when the user:
- Asks questions requiring information from multiple sources
- Needs to connect facts across documents ("A founded B, B acquired C, what's A's connection to C?")
- Has research questions requiring iterative discovery
- Asks "compare", "trace", "investigate", or "how did X lead to Y" questions
- Needs cross-document fact checking or verification

## Interactive Flow

Multi-hop adds complexity and latency. Validate the need and choose the right strategy.

### Step 1: Validate Multi-Hop Need

```
"Multi-hop retrieval is for questions that need information from multiple places.

Can you give me an example question users will ask?
___

I'll analyze if it truly needs multi-hop or if single retrieval suffices."
```

**Analysis criteria**:
| Question Pattern | Needs Multi-Hop? |
|------------------|------------------|
| "What is X?" | No - single retrieval |
| "Compare X and Y" | Yes - need both X and Y |
| "How did X lead to Y?" | Yes - need chain |
| "What are all the..." | Maybe - depends on distribution |

```
"Your example '[question]' [does/doesn't] need multi-hop because [reason].

Proceed with multi-hop? (yes / try basic RAG first)"
```

### Step 2: Choose Decomposition Strategy

```
"How should I break down complex questions?

A) **Upfront decomposition** - Split into sub-questions first, retrieve each
   Best for: Questions with clear parts ("Compare A and B")

B) **Iterative discovery** - Retrieve, evaluate, follow leads
   Best for: Questions where you don't know what you need ("Investigate X")

C) **Hybrid** - Decompose first, then iterate if needed
   Best for: Complex research questions

Which strategy? (A/B/C)"
```

| Strategy | Example Flow |
|----------|--------------|
| A (Upfront) | Q: "Compare pricing of A and B" → Sub-Q1: "A pricing" → Sub-Q2: "B pricing" → Combine |
| B (Iterative) | Q: "Who owns company X?" → Find X → X owned by Y → Find Y → Y owned by Z → Answer chain |
| C (Hybrid) | Decompose into parts, then iterate within each part |

### Step 3: Set Hop Limits

```
"How deep should the search go?

A) 2 hops (fast, simple chains)
B) 3 hops (balanced, most cases)
C) 4-5 hops (deep research, complex chains)

More hops = more complete but slower. Recommend B.

Which? (A/B/C)"
```

### Step 4: Sufficiency Criteria

```
"When should I stop searching and answer?

A) **Strict** - Only stop when ALL parts are fully answered
   Use for: Legal, compliance, fact-checking

B) **Lenient** - Stop when main information is found
   Use for: General research, exploratory questions

Which? (A/B)"
```

### Step 5: Confirm Configuration

```
"Multi-hop configuration:

- **Strategy**: [from Step 2]
- **Max hops**: [from Step 3]
- **Sufficiency**: [from Step 4]
- **Per-hop retrieval**: Top-5

Example flow for your question:
  Hop 1: [query] → find [what]
  Hop 2: [follow-up] → find [what]
  → Synthesize answer

Does this approach make sense? (yes / adjust [what])"
```

### Checkpoints During Execution

| Checkpoint | Question to User |
|------------|------------------|
| After decomposition | "I've split into these sub-questions: [list]. Look right?" |
| After each hop | "Hop [N] found: [summary]. Continue searching or enough info?" |
| Before synthesis | "I have info from [N] sources. Ready to synthesize?" |
| If stuck | "Hop [N] found nothing relevant. Try different terms or answer with what we have?" |

### Red Flags - When to Simplify

```
"I notice:
- Your question only needs info from one source
- OR: Each hop returns the same documents
- OR: The 'chain' is actually just one lookup

Basic RAG would be faster and simpler. Switch? (yes / keep multi-hop)"
```

### Strategy Selection Helper

If user is unsure, use this decision tree:

```
"Let me help choose the right approach:

Is the answer in ONE document?
  → Yes: Use basic `rag`
  → No: Continue...

Do you know what pieces you need upfront?
  → Yes: Upfront decomposition (Strategy A)
  → No: Continue...

Is it a chain (A→B→C) or parallel (A, B, C)?
  → Chain: Iterative discovery (Strategy B)
  → Parallel: Upfront decomposition (Strategy A)
  → Both: Hybrid (Strategy C)

Based on this: I recommend Strategy [X]. Agree?"
```

## Core Concepts

### The Multi-Hop Problem

Some questions can't be answered with a single retrieval:

```
Question: "Who is the CEO of the company that acquired our main competitor?"

Single-hop fails:
  Search: "CEO company acquired competitor" → No direct match

Multi-hop succeeds:
  Hop 1: "main competitor" → "TechCorp is our main competitor"
  Hop 2: "TechCorp acquisition" → "TechCorp was acquired by MegaInc in 2023"
  Hop 3: "MegaInc CEO" → "Jane Smith is CEO of MegaInc"
  Answer: "Jane Smith"
```

### Retrieval Chain

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Hop Retrieval                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Question ──▶ Decompose ──▶ [Sub-Q 1] ──▶ [Sub-Q 2] ──▶ ...    │
│                                 │             │                 │
│                                 ▼             ▼                 │
│                            Retrieve 1    Retrieve 2             │
│                                 │             │                 │
│                                 ▼             ▼                 │
│                            Facts 1   ───▶ Facts 2               │
│                            (inform next query)                  │
│                                                                 │
│                                     │                           │
│                                     ▼                           │
│                               Synthesize                        │
│                                     │                           │
│                                     ▼                           │
│                                 Answer                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Mental Model: Detective Investigation

Think of multi-hop RAG like a detective following leads:
- **Clue 1** leads to **Person A**
- **Person A** mentions **Location B**
- **Location B** reveals **Evidence C**
- **Evidence C** solves the case

Each finding opens new avenues of investigation.

## Why Multi-Hop Over Alternatives

| Approach | Single Query | Multi-Step | Information Linking | Use Case |
|----------|--------------|------------|---------------------|----------|
| Basic RAG | Yes | No | No | Simple factual Q&A |
| Agentic RAG | Yes | Yes (dynamic) | Limited | Conversational |
| **Multi-Hop RAG** | No | Yes (structured) | Yes | Complex research |

**Choose Multi-Hop when**:
- Answer requires connecting 2+ pieces of information
- Question involves relationships, comparisons, or chains
- Single retrieval returns partial information
- Research tasks need systematic exploration

**Skip Multi-Hop when**:
- Question is simple and factual
- All information is in one document
- Latency is critical (each hop adds ~200ms)

## Implementation

```python
from pymilvus import MilvusClient, DataType
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MultiHopRAG:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        self.collection_name = "multi_hop_rag"
        self._init_collection()

    def _embed(self, texts):
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

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = self._embed(query)[0]
        results = self.client.search(self.collection_name, [embedding], limit=top_k,
                                     output_fields=["text", "source"])
        return [{"text": h["entity"]["text"], "source": h["entity"]["source"]} for h in results[0]]

    def decompose(self, question: str) -> list[str]:
        """Decompose complex question into sub-questions"""
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Break this complex question into 2-4 simpler sub-questions that can be answered independently.
Each sub-question should retrieve one piece of information.

Question: {question}

Sub-questions (one per line, no numbering):"""
            }],
            temperature=0
        )
        subs = response.choices[0].message.content.strip().split("\n")
        return [s.strip() for s in subs if s.strip()]

    def multi_hop_retrieve(self, question: str, max_hops: int = 3) -> dict:
        """Iterative retrieval with information accumulation"""
        all_contexts = []
        queries = [question]
        hop_details = []

        for hop in range(max_hops):
            current_query = queries[-1]
            results = self.retrieve(current_query, top_k=5)
            all_contexts.extend(results)

            hop_details.append({
                "hop": hop + 1,
                "query": current_query,
                "results_count": len(results)
            })

            # Check sufficiency
            context_text = "\n".join([c["text"] for c in all_contexts])
            check = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"""Given the original question and retrieved information, determine if we have enough to answer.

Question: {question}

Retrieved information:
{context_text}

Answer "SUFFICIENT" if we can answer, or "NEED: <what's missing>" if more info needed:"""
                }],
                temperature=0
            )

            response_text = check.choices[0].message.content
            if "SUFFICIENT" in response_text.upper():
                break

            # Generate follow-up query based on what's missing
            if "NEED:" in response_text:
                missing = response_text.split("NEED:")[-1].strip()
                followup = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"""Generate a search query to find: {missing}

Context we already have:
{context_text}

Search query (keywords only):"""
                    }],
                    temperature=0
                )
                queries.append(followup.choices[0].message.content.strip())

        return {
            "contexts": all_contexts,
            "queries": queries,
            "hops": hop_details
        }

    def query(self, question: str, max_hops: int = 3) -> dict:
        """Complete multi-hop Q&A pipeline"""
        # Retrieve through multiple hops
        retrieval = self.multi_hop_retrieve(question, max_hops)

        # Deduplicate contexts
        seen = set()
        unique_contexts = []
        for c in retrieval["contexts"]:
            if c["text"] not in seen:
                seen.add(c["text"])
                unique_contexts.append(c)

        # Generate answer
        context_text = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in unique_contexts])
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Answer the question by synthesizing information from multiple sources. Show your reasoning.

Sources:
{context_text}

Question: {question}

Answer:"""
            }],
            temperature=0.3
        )

        return {
            "answer": response.choices[0].message.content,
            "hops": len(retrieval["hops"]),
            "queries": retrieval["queries"],
            "sources": list(set(c["source"] for c in unique_contexts)),
            "hop_details": retrieval["hops"]
        }
```

**Usage**:
```python
rag = MultiHopRAG()
rag.add_document(open("people.md").read(), source="people.md")
rag.add_document(open("companies.md").read(), source="companies.md")
rag.add_document(open("deals.md").read(), source="deals.md")

result = rag.query("Who founded the company that acquired TechCorp?")
print(f"Answer: {result['answer']}")
print(f"Hops taken: {result['hops']}")
print(f"Queries: {result['queries']}")
```

## Configuration Guide

### Max Hops

| Complexity | max_hops | Example Question |
|------------|----------|------------------|
| Simple chain | 2 | "Who is X's manager?" |
| Medium | 3 | "What company did X's former employer acquire?" |
| Complex research | 4-5 | "Trace the ownership history of product Y" |

**Rule**: Start with 3, increase only if answers are incomplete.

### Hop Strategy

| Strategy | When to Use | Implementation |
|----------|-------------|----------------|
| Question Decomposition | Clear sub-questions | Split upfront, retrieve each |
| Iterative Discovery | Unknown structure | Retrieve → evaluate → follow leads |
| Hybrid | Complex research | Decompose first, then iterate |

### Sufficiency Checking

```python
# Strict (for high-precision needs)
STRICT_CHECK = """Answer SUFFICIENT only if ALL parts of the question can be fully answered.
Any uncertainty = NEED: <what's uncertain>"""

# Lenient (for exploratory research)
LENIENT_CHECK = """Answer SUFFICIENT if we have the main information.
Only say NEED if critical information is completely missing."""
```

## Common Pitfalls

### 1. Too Many Hops
**Symptom**: Slow responses, redundant information
**Fix**: Lower max_hops, improve sufficiency check

### 2. Circular Retrieval
**Symptom**: Same documents retrieved repeatedly
**Fix**: Track seen documents, exclude from subsequent searches

```python
def retrieve_excluding(self, query: str, exclude_ids: set, top_k: int = 5):
    results = self.retrieve(query, top_k=top_k * 2)
    return [r for r in results if r["id"] not in exclude_ids][:top_k]
```

### 3. Query Drift
**Symptom**: Follow-up queries stray from original question
**Fix**: Always include original question in follow-up generation

### 4. Information Overload
**Symptom**: Too much context, LLM gets confused
**Fix**: Limit total context tokens, summarize intermediate results

```python
def summarize_context(self, contexts: list, question: str) -> str:
    """Summarize accumulated context to manage token budget"""
    if len(contexts) <= 5:
        return "\n".join([c["text"] for c in contexts])

    response = self.openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Summarize these {len(contexts)} passages into key facts relevant to: {question}

{chr(10).join([c['text'] for c in contexts])}

Key facts (bullet points):"""
        }]
    )
    return response.choices[0].message.content
```

### 5. Missing Connections
**Symptom**: Has all facts but doesn't connect them
**Fix**: Explicitly prompt for reasoning chain in generation

## Advanced Patterns

### Decompose-Then-Retrieve

```python
def query_decomposed(self, question: str) -> dict:
    """Decompose first, then retrieve for each sub-question"""
    # Step 1: Decompose
    sub_questions = self.decompose(question)

    # Step 2: Retrieve for each sub-question
    all_results = []
    for sub_q in sub_questions:
        results = self.retrieve(sub_q, top_k=3)
        all_results.append({
            "sub_question": sub_q,
            "contexts": results
        })

    # Step 3: Synthesize
    # ... combine all results and generate answer
```

### Parallel Multi-Hop

```python
import asyncio

async def parallel_retrieve(self, queries: list[str]) -> list[list[dict]]:
    """Retrieve for multiple queries in parallel"""
    tasks = [self._async_retrieve(q) for q in queries]
    return await asyncio.gather(*tasks)
```

### With Reranking

```python
def multi_hop_with_rerank(self, question: str) -> dict:
    """Add reranking at each hop for higher precision"""
    # ... retrieve as usual
    # At each hop, rerank before adding to context
    reranked = self.reranker.rerank(query, results, top_k=3)
    all_contexts.extend(reranked)
```

## When to Level Up

| Symptom | Solution | Recommendation |
|---------|----------|----------------|
| Need dynamic conversation | Agent-based approach | Use `rag-toolkit:agentic-rag` skill |
| Need higher precision per hop | Add reranking | Use `rag-toolkit:rag-with-rerank` skill |
| External data sources needed | Full agent framework | Consider LangChain/LlamaIndex |

## References

**Internal**:
- [references/decomposition-strategies.md](references/decomposition-strategies.md) - Question decomposition techniques

**Related skills**:
- `rag-toolkit:rag` - Basic RAG (simpler questions)
- `rag-toolkit:agentic-rag` - Dynamic, conversational retrieval
- `rag-toolkit:rag-with-rerank` - Higher precision per hop

**Verticals**:
- [verticals/troubleshooting.md](verticals/troubleshooting.md) - Multi-step diagnosis
