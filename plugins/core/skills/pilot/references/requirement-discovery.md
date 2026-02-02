# Requirement Discovery Guide

Quickly gather requirements, then **directly orchestrate the toolchain** for users - don't let them struggle with choices.

## Two Core Questions

You only need to clarify these two things:

### 1. What does your data look like?

| Question | Options | Impact |
|----------|---------|--------|
| Data type | Text / Images / PDF / Mixed | Determines embedding model |
| Data volume | Thousands / Tens of thousands / Millions | Determines index type and storage |
| Any labels/categories | Yes / No | Determines if filter fields needed |

### 2. What does your query look like?

| Query Type | Example | Toolchain |
|------------|---------|-----------|
| Simple semantic search | "Find related articles" | embedding → vector search |
| Filtered search | "Find products in phone category" | embedding + scalar filtering |
| Image search | "Find similar images" | CLIP → vector search |
| Image-to-text search | "Find photos of cars turning right" | Image captioning → embedding → search |
| Q&A | "What does this document say" | RAG (retrieval + LLM) |

## Quick Decision Table

Based on data and query, **directly select the toolchain**:

### Data Processing

| Data Type | Processing Method | Tool |
|-----------|-------------------|------|
| Plain text | Direct embedding | sentence-transformers |
| PDF | Extract text → chunk → embedding | PyPDF + chunking |
| Images | CLIP embedding or image-to-text | clip-ViT-B-32 |
| Images needing content understanding | VLM generates description → embedding | GPT-4V / Qwen-VL |

### Embedding Model (Select directly, don't ask user)

| Scenario | Model | Dimensions |
|----------|-------|------------|
| Chinese text | BAAI/bge-large-zh-v1.5 | 1024 |
| English text | text-embedding-3-small | 1536 |
| Images/Image-text | clip-ViT-B-32 | 512 |

### Storage (Select directly, don't ask user)

| Scale | Storage | Index |
|-------|---------|-------|
| < 10k | Milvus Lite (local file) | FLAT |
| 10k - 1M | Zilliz Cloud Serverless | HNSW |
| > 1M | Zilliz Cloud Dedicated | IVF_PQ |

### Whether Additional Fields Needed

| Situation | Handling |
|-----------|----------|
| Need category filtering | Add category field |
| Need time filtering | Add timestamp field |
| Need to return original text | Add content field |
| Pure vector search | Only need id + embedding |

## Workflow

```
1. Ask about data → 2. Ask about query → 3. Auto-orchestrate toolchain → 4. Generate code → 5. User tests
```

**Don't ask about**:
- What language → Python
- What framework → FastAPI
- Which embedding model → Auto-select based on data type
- Which storage → Auto-select based on scale

## Example Conversations

**User**: I want to build an image search

**You**: Sure, let me ask two questions:
1. Approximately how many images?
2. How will users search? Upload an image to find similar ones, or search by text description?

**User**: About 100k product images, users upload images to find similar

**You**: Got it, I'll set up for you:
- CLIP model to convert images to vectors
- Store in Zilliz Cloud Serverless
- AUTOINDEX
- FastAPI for the API

Let me generate the code - you can try running it first.

---

**User**: I want AI to answer questions about our company documents

**You**: Sure, let me ask two questions:
1. What format are the documents? PDF, Word, or web pages?
2. Approximately how many documents?

**User**: PDF, about 200

**You**: Got it, this is a typical RAG scenario, I'll set up:
- Extract text from PDF, chunk at 512 characters
- BGE model for vectorization
- Store in Milvus Lite (sufficient for local use)
- Retrieve + GPT generates answers at query time

Let me generate the code.

## Iteration (60 → 80%)

Users may give feedback after testing, provide optimization suggestions:

| Problem | Optimization Suggestion |
|---------|------------------------|
| Irrelevant results | Add rerank, adjust embedding model |
| Search too slow | Tune index parameters, add caching |
| Want category filtering | Add scalar field for filtering |
| Want to see similarity scores | Return score field |
| Too few/too many results | Adjust limit and threshold |

**Key mindset**: Get users up and running first, then iterate based on feedback. Don't pursue perfection from the start.

## Next Steps

After requirements are clear:
- Directly orchestrate toolchain → `development-workflow.md`
- Match pre-built solution → `solution-matching.md`
