# Chunking Strategies for RAG

This reference covers chunking approaches in depth, helping you choose and configure the right strategy for your use case.

## Why Chunking Matters

Chunking is the **most impactful** decision in RAG pipeline design:

| Chunk Size | Retrieval Precision | Context Quality | Token Cost |
|------------|--------------------:|----------------:|-----------:|
| Too small (128) | High | Poor (fragments) | Low |
| Optimal (512) | Good | Good | Medium |
| Too large (2048) | Low | Rich but noisy | High |

**The tradeoff**: Smaller chunks → better retrieval precision, but less context per chunk. Larger chunks → more context, but retrieval returns irrelevant portions.

## Chunking Methods

### 1. Fixed-Size Chunking

Split by character/token count with overlap.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]  # Try these in order
)
chunks = splitter.split_text(document)
```

**Pros**: Simple, predictable chunk sizes
**Cons**: May split mid-sentence or mid-concept

**When to use**: General documents, when simplicity matters

### 2. Semantic Chunking

Split at natural boundaries (paragraphs, sections).

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Prioritize semantic boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=100,
    separators=[
        "\n## ",      # Markdown H2
        "\n### ",     # Markdown H3
        "\n\n",       # Paragraph
        "\n",         # Line
        ". ",         # Sentence
        " ",
        ""
    ]
)
```

**Pros**: Preserves semantic units
**Cons**: Variable chunk sizes, may exceed limits

**When to use**: Structured documents (markdown, HTML)

### 3. Sentence-Based Chunking

Split by sentences, group into chunks.

```python
import nltk
nltk.download('punkt')

def sentence_chunk(text: str, sentences_per_chunk: int = 5):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks
```

**Pros**: Never splits mid-sentence
**Cons**: Requires NLP library, variable sizes

**When to use**: Prose documents, when sentence integrity matters

### 4. Document-Specific Chunking

Custom logic per document type.

```python
def chunk_code(code: str):
    """Split code by function/class definitions"""
    import ast
    tree = ast.parse(code)
    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            chunks.append(ast.get_source_segment(code, node))
    return chunks

def chunk_markdown(md: str):
    """Split markdown by headers"""
    import re
    sections = re.split(r'\n(?=#{1,3} )', md)
    return [s.strip() for s in sections if s.strip()]
```

**Pros**: Optimal for specific formats
**Cons**: Requires custom code per type

**When to use**: Homogeneous corpus (all code, all markdown, etc.)

### 5. Sliding Window with Embedding Similarity

Advanced: Split when embedding similarity drops.

```python
from openai import OpenAI
import numpy as np

def semantic_split(text: str, threshold: float = 0.8):
    """Split when consecutive sentences become semantically distant"""
    client = OpenAI()
    sentences = text.split(". ")

    def embed(t):
        return client.embeddings.create(model="text-embedding-3-small", input=[t]).data[0].embedding

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    chunks, current = [], [sentences[0]]
    prev_emb = embed(sentences[0])

    for sent in sentences[1:]:
        curr_emb = embed(sent)
        if cosine_sim(prev_emb, curr_emb) < threshold:
            chunks.append(". ".join(current))
            current = [sent]
        else:
            current.append(sent)
        prev_emb = curr_emb

    chunks.append(". ".join(current))
    return chunks
```

**Pros**: True semantic boundaries
**Cons**: Expensive (embedding per sentence), slow

**When to use**: High-value documents where quality justifies cost

## Chunk Size Guidelines by Document Type

| Document Type | Recommended Size | Overlap | Rationale |
|---------------|------------------|---------|-----------|
| **General text** | 512 | 50 | Balanced default |
| **Technical docs** | 1024 | 100 | Preserve procedures, code |
| **FAQ** | 256 | 0 | One Q&A per chunk |
| **Legal contracts** | 1024 | 200 | High overlap for clauses |
| **Chat logs** | 512 | 100 | Preserve conversation flow |
| **Source code** | Function-based | 0 | One function per chunk |
| **Academic papers** | 1024 | 150 | Preserve argument structure |
| **Product reviews** | 256 | 0 | One review per chunk |

## Overlap Strategy

Overlap prevents information loss at chunk boundaries.

```
Without overlap:
[Chunk 1: "The process requires A and B."][Chunk 2: "C is also important."]
Query "B and C" might miss both chunks!

With overlap:
[Chunk 1: "The process requires A and B. C is also"][Chunk 2: "B. C is also important."]
Query "B and C" now matches Chunk 2!
```

**Rules of thumb**:
- 10% overlap for independent chunks (FAQ, reviews)
- 15-20% overlap for continuous prose
- 25%+ overlap for legal/regulatory (clause continuity critical)

## Measuring Chunk Quality

### Retrieval Precision Test

```python
def test_chunk_quality(rag, test_cases):
    """
    test_cases = [
        {"query": "What is X?", "expected_source": "doc_about_x.md"},
        ...
    ]
    """
    hits = 0
    for case in test_cases:
        results = rag.retrieve(case["query"], top_k=3)
        sources = [r["source"] for r in results]
        if case["expected_source"] in sources:
            hits += 1
    return hits / len(test_cases)
```

**Target**: >80% precision at top-3

### Chunk Coherence Check

```python
def check_coherence(chunks):
    """Flag chunks that start/end mid-sentence"""
    issues = []
    for i, chunk in enumerate(chunks):
        if chunk[0].islower():  # Starts mid-sentence
            issues.append(f"Chunk {i}: starts mid-sentence")
        if not chunk.rstrip()[-1] in ".!?\"'":  # Ends mid-sentence
            issues.append(f"Chunk {i}: ends mid-sentence")
    return issues
```

## Common Mistakes

### 1. One-Size-Fits-All
**Mistake**: Using same chunk size for all document types
**Fix**: Tune per document type, or use adaptive chunking

### 2. Ignoring Document Structure
**Mistake**: Chunking markdown/HTML without respecting headers
**Fix**: Use semantic separators that match format

### 3. Zero Overlap
**Mistake**: No overlap, losing boundary information
**Fix**: At least 10% overlap for prose

### 4. Chunking After Concatenation
**Mistake**: Combining multiple documents then chunking
**Fix**: Chunk each document separately to preserve source boundaries

## Quick Reference

```python
# Standard configurations

# General purpose
GENERAL = {"chunk_size": 512, "chunk_overlap": 50}

# Technical documentation
TECHNICAL = {"chunk_size": 1024, "chunk_overlap": 100}

# FAQ / Q&A pairs
FAQ = {"chunk_size": 256, "chunk_overlap": 0}

# Legal / Contracts
LEGAL = {"chunk_size": 1024, "chunk_overlap": 200}

# Conversational (chat logs)
CHAT = {"chunk_size": 512, "chunk_overlap": 100}
```
