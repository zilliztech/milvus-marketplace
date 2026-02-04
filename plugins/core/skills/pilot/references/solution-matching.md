# Solution Matching Guide

How to match the best solution for user requirements.

## Quick Matching Table

| User Says | Actual Need | Recommended Solution |
|-----------|-------------|---------------------|
| "Search", "find similar" | Semantic search | `retrieval-system:semantic-search` |
| "Keyword + semantic", "full-text + vector" | Hybrid search | `retrieval-system:hybrid-search` |
| "Search with conditions", "filter by category" | Filtered search | `retrieval-system:filtered-search` |
| "Match across multiple fields" | Multi-vector search | `retrieval-system:multi-vector-search` |
| "Parent document", "hierarchical chunks", "small-to-big" | Contextual retrieval | `retrieval-system:contextual-retrieval` |
| "Q&A", "knowledge base", "RAG" | Knowledge Q&A | `rag-toolkit:rag` |
| "Document Q&A", "PDF Q&A" | Document QA (RAG variant) | `rag-toolkit:rag` |
| "High-precision Q&A", "results not accurate enough" | RAG with rerank | `rag-toolkit:rag-with-rerank` |
| "Complex question", "multi-step reasoning" | Multi-hop RAG | `rag-toolkit:multi-hop-rag` |
| "Smart assistant", "autonomous agent" | Agentic RAG | `rag-toolkit:agentic-rag` |
| "Image-to-image", "image search" | Image search | `multimodal-retrieval:image-search` |
| "Search images with text description" | Text-to-image search | `multimodal-retrieval:text-to-image-search` |
| "Recommend similar products", "related items" | Item-to-item recommendation | `rec-system:item-to-item` |
| "Personalized recommendations", "you might like" | User-to-item recommendation | `rec-system:user-to-item` |
| "Find duplicates", "deduplication" | Duplicate detection | `data-analytics:duplicate-detection` |
| "Clustering", "group similar items" | Clustering analysis | `data-analytics:clustering` |
| "Remember conversations", "chat history" | Chat memory | `memory-system:chat-memory` |
| "Mixed image-text documents", "product manuals" | Multimodal RAG | `multimodal-retrieval:multimodal-rag` |
| "Video search", "find video clips" | Video search | `multimodal-retrieval:video-search` |

## Requirement Recognition

### Semantic Search (`retrieval-system:semantic-search`)

**Typical expressions**:
- "Search for similar content"
- "Find related articles"
- "Semantic retrieval"
- "Full-text search but understand meaning"

**Core characteristics**:
- Input: Query text
- Output: Similar text list
- No answer generation needed

**Use cases**:
- Content retrieval
- Similar article recommendations
- Duplicate content detection

### Hybrid Search (`retrieval-system:hybrid-search`)

**Typical expressions**:
- "Keyword + semantic search"
- "Full-text search plus vector search"
- "Want both exact match and fuzzy match"

**Core characteristics**:
- Combines vector search with BM25 keyword search
- Score fusion (RRF or weighted)
- Better recall than pure vector search

**Use cases**:
- E-commerce product search
- Legal document retrieval
- Academic paper search

### Filtered Search (`retrieval-system:filtered-search`)

**Typical expressions**:
- "Search within a category"
- "Filter by price/date then search"
- "Search with conditions"

**Core characteristics**:
- Vector search + scalar field filtering
- Requires additional metadata fields
- Pre-filter or post-filter strategies

**Use cases**:
- E-commerce (category + similarity)
- Recruitment (location + skill match)
- Real estate (area + preference match)

### Multi-Vector Search (`retrieval-system:multi-vector-search`)

**Typical expressions**:
- "Match across title and content"
- "Search by multiple fields simultaneously"
- "Joint search on text and image"

**Core characteristics**:
- Multiple vector fields in one collection
- Weighted combination of similarity scores
- Different embedding models per field

**Use cases**:
- Product search (title + description + image)
- Academic papers (title + abstract)
- Resume matching (skills + experience)

### Contextual Retrieval (`retrieval-system:contextual-retrieval`)

**Typical expressions**:
- "Need broader context around matched chunks"
- "Parent document retrieval"
- "Hierarchical chunking"
- "Small chunks for search, big chunks for context"
- "Lost in the middle problem"

**Core characteristics**:
- Parent-child document structure
- Search on small child chunks for precision
- Return parent chunks for complete context
- Deduplication of parent contexts

**Use cases**:
- Legal contract Q&A (clause + surrounding sections)
- Technical documentation (code snippet + explanation)
- Long-form research papers (precise matching with context)
- Product manuals (specific instruction + full procedure)

### RAG (`rag-toolkit:rag`)

**Typical expressions**:
- "Knowledge base Q&A"
- "Let AI answer questions about my data"
- "Document-based conversation"
- "Private knowledge base"
- "Ask questions about this PDF"

**Core characteristics**:
- Input: User question
- Output: Answer based on knowledge base
- Requires LLM for answer generation

**Use cases**:
- Enterprise knowledge base
- Customer service bots
- Document assistant

### RAG with Rerank (`rag-toolkit:rag-with-rerank`)

**Typical expressions**:
- "RAG results not accurate enough"
- "Need higher precision Q&A"
- "Want to improve answer quality"

**Core characteristics**:
- Adds cross-encoder reranking after retrieval
- Higher precision at the cost of latency
- Typically retrieves more candidates then reranks

**Use cases**:
- Legal consulting
- Medical Q&A
- Financial report analysis

### Multi-hop RAG (`rag-toolkit:multi-hop-rag`)

**Typical expressions**:
- "Complex question that needs multiple lookups"
- "Compare information from different documents"
- "Multi-step reasoning"

**Core characteristics**:
- Multiple rounds of retrieval
- Previous results guide next retrieval
- Chain-of-thought reasoning

**Use cases**:
- Complex research questions
- Fact-checking across sources
- Troubleshooting workflows

### Agentic RAG (`rag-toolkit:agentic-rag`)

**Typical expressions**:
- "Smart assistant that decides what to search"
- "Agent that autonomously finds answers"
- "Intelligent research assistant"

**Core characteristics**:
- Agent autonomously decides retrieval strategy
- Can use multiple tools and data sources
- Dynamic query planning

**Use cases**:
- Smart assistants
- Research agents
- Complex task automation

### Image Search (`multimodal-retrieval:image-search`)

**Typical expressions**:
- "Image-to-image search"
- "Find similar images"
- "Image retrieval"
- "Visual search"

**Core characteristics**:
- Images as input or output
- Visual similarity
- CLIP multimodal

**Use cases**:
- E-commerce find similar
- Image library management
- Visual search

### Text-to-Image Search (`multimodal-retrieval:text-to-image-search`)

**Typical expressions**:
- "Search images by text description"
- "Find photos matching a description"
- "Text query for images"

**Core characteristics**:
- VLM generates captions for images
- Text embedding on captions
- Text query matches image descriptions

**Use cases**:
- Stock image search
- Surveillance video retrieval
- Medical image search

### Item-to-Item Recommendation (`rec-system:item-to-item`)

**Typical expressions**:
- "Recommend similar products"
- "Related articles"
- "Similar videos"

**Core characteristics**:
- Item vector similarity
- Content-based recommendation
- No user profile needed

**Use cases**:
- Similar products
- Related articles
- Similar videos

### User-to-Item Recommendation (`rec-system:user-to-item`)

**Typical expressions**:
- "Personalized recommendations"
- "You might like"
- "Personalized feed"

**Core characteristics**:
- User vector + item vector matching
- Based on user behavior/preferences
- Real-time requirements

**Use cases**:
- Personalized product recommendations
- Content feeds
- Job matching

### Duplicate Detection (`data-analytics:duplicate-detection`)

**Typical expressions**:
- "Find duplicate content"
- "Deduplication"
- "Plagiarism detection"

**Core characteristics**:
- Batch vector comparison
- Similarity threshold
- Pairwise matching

**Use cases**:
- Plagiarism detection
- Content deduplication
- Resume deduplication

### Clustering (`data-analytics:clustering`)

**Typical expressions**:
- "Group similar items"
- "Topic clustering"
- "Find natural groupings"

**Core characteristics**:
- Vector clustering analysis
- Unsupervised grouping
- Cluster labeling

**Use cases**:
- Topic clustering
- User segmentation
- Anomaly detection

### Chat Memory (`memory-system:chat-memory`)

**Typical expressions**:
- "Remember conversation history"
- "Long-term memory for chatbot"
- "Context-aware conversations"

**Core characteristics**:
- Conversation vectorization
- Time decay weighting
- Relevant memory retrieval

**Use cases**:
- Long-term assistants
- Customer service memory
- Game NPCs

### Multimodal RAG (`multimodal-retrieval:multimodal-rag`)

**Typical expressions**:
- "Documents with images and text"
- "Product manuals with diagrams"
- "Mixed media Q&A"

**Core characteristics**:
- Mixed image-text parsing
- VLM for image understanding
- Combined retrieval

**Use cases**:
- Product manuals
- Medical reports
- Financial reports with charts

### Video Search (`multimodal-retrieval:video-search`)

**Typical expressions**:
- "Search within videos"
- "Find specific video clips"
- "Video content retrieval"

**Core characteristics**:
- Frame extraction or transcription
- Embedding on frames/text
- Temporal indexing

**Use cases**:
- Educational video search
- Meeting recording search
- Surveillance playback

## No Match Situations

If no pre-built solution matches:

1. **Still in deep expertise domain**
   - Combine with core tools
   - Reference similar scenario patterns
   - Custom development

2. **In general domain**
   - Provide technical advice
   - Design architecture
   - Write code directly
   - No pre-built toolchain

## Mixed Requirements

User requirements may involve multiple scenarios:

**Example**: "I want to build an e-commerce platform with product search and recommendations"

**Breakdown**:
1. Product search → `retrieval-system:semantic-search` or `multimodal-retrieval:image-search`
2. Recommendation system → `rec-system:item-to-item` or `rec-system:user-to-item`

**Strategy**:
- Implement each module separately
- Share Milvus infrastructure
- Unified data model design

## Next Steps

After solution is determined:
- Enter development workflow → See `development-workflow.md`
- Return to controller → `pilot`
