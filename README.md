# Milvus Marketplace

A data retrieval development assistant based on Claude Code Skills.

## Core Philosophy

### Focus: Data Retrieval Domain

We specialize in the **data retrieval** vertical:

- Vector search, semantic search
- RAG, knowledge base Q&A
- Image search, recommendation systems
- Milvus / Zilliz Cloud

### Two-Layer Solution Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Scenario Plugins (6 plugins)               │
│   rag-toolkit, retrieval-system, multimodal-retrieval,  │
│   rec-system, memory-system, data-analytics             │
│   = Pre-built solutions = AI era caching mechanism      │
└─────────────────────────────────────────────────────────┘
                           ↑
                     Match / Combine
                           ↑
┌─────────────────────────────────────────────────────────┐
│                      core plugin                         │
│       Methodology (pilot) + Atomic operators            │
│              (embedding, chunking, ...)                 │
└─────────────────────────────────────────────────────────┘
```

**Core Ideas**:

1. **Scenario plugins = AI cache**: Pre-built scenario solutions act like cache - use directly if matched, no need to build from scratch
2. **core = methodology + operators**: Universal development methodology + composable atomic capabilities
3. **Ship 60% first, iterate to 80%**: Deliver working solutions quickly, then optimize based on feedback

### Scenario Classification Principles

Scenarios are classified by **architectural differences**, not by industry or model:

- **Different architecture = different scenario**: Fundamentally different processes and code structures
- **Only model/parameters differ = vertical applications within same scenario**: Covered through configuration tables

```
plugins/retrieval-system/skills/
├── semantic-search/          # Category: architecture definition
│   ├── SKILL.md              # Generic workflow + model selection table
│   └── verticals/            # Subcategory: vertical application guides
│       ├── legal.md          # Legal search
│       ├── academic.md       # Academic papers
│       └── ecommerce.md      # E-commerce search
```

### Workflow

```
User describes requirement
      │
      ▼
   pilot activates
      │
      ├─→ Clarify data and query
      │
      ├─→ Can match a scenario?
      │       ├─ Yes → Use pre-built solution
      │       └─ No → Combine core operators
      │
      ├─→ Generate code → User tests
      │
      └─→ Collect feedback → Iterate
```

## Installation

### 1. Add Marketplace

```bash
/plugin marketplace add zilliztech/milvus-marketplace
```

### 2. Install Plugins

```bash
# Core tools (required)
/plugin install core@milvus-marketplace

# Install scenario plugins as needed
/plugin install rag-toolkit@milvus-marketplace        # RAG solutions
/plugin install retrieval-system@milvus-marketplace   # Text search
/plugin install multimodal-retrieval@milvus-marketplace # Image/video/multimodal
/plugin install rec-system@milvus-marketplace         # Recommendations
/plugin install memory-system@milvus-marketplace      # Chat memory
/plugin install data-analytics@milvus-marketplace     # Duplicate detection, clustering
```

### 3. Start Using

Simply describe what you want to build:

```
"Help me build a document Q&A system"
"I want to implement semantic search"
"Build an image search application"
```

The pilot will automatically activate, clarify requirements, and help you orchestrate the toolchain and generate code.

## Plugin Overview

### core - Core Capabilities

| Type | Skill | Purpose |
|------|-------|---------|
| Controller | **pilot** | AI application navigator - understands requirements, orchestrates tools, delivers code |
| Operator | **embedding** | Text/image vectorization |
| Operator | **chunking** | Document chunking |
| Operator | **indexing** | Milvus index management |
| Operator | **data-ingestion** | Batch data import |
| Operator | **rerank** | Search result reranking |
| Operator | **pdf-extract** | PDF text extraction |
| Operator | **vlm-caption** | Image captioning (VLM) |
| Environment | **local-setup** | Local Milvus deployment |

### Scenario Plugins - Pre-built Solutions (6 Plugins, 17 Skills)

#### retrieval-system - Text Search

| Skill | Architecture | Vertical Applications |
|-------|-------------|----------------------|
| **semantic-search** | embedding → vector search | Legal, academic, news, e-commerce, code, patents |
| **hybrid-search** | vector + BM25 keyword + score fusion | E-commerce, legal, academic |
| **filtered-search** | vector search + scalar filtering | E-commerce, recruitment, real estate |
| **multi-vector-search** | multi-vector field joint search | Products, papers, resumes |

#### rag-toolkit - RAG / Q&A

| Skill | Architecture | Vertical Applications |
|-------|-------------|----------------------|
| **rag** | retrieval → LLM generation | Enterprise KB, product docs, policies, customer service |
| **rag-with-rerank** | retrieval → rerank → LLM | Legal consulting, medical Q&A, financial reports |
| **multi-hop-rag** | multi-round retrieval (results guide next round) | Complex research, fact-checking, troubleshooting |
| **agentic-rag** | Agent autonomously decides retrieval strategy | Smart assistants, research agents |

#### multimodal-retrieval - Image / Video / Multimodal

| Skill | Architecture | Vertical Applications |
|-------|-------------|----------------------|
| **image-search** | CLIP embedding → vector search | E-commerce visual search, fashion, face, logo, design assets |
| **text-to-image-search** | VLM captioning → text embedding | Stock images, surveillance retrieval, medical imaging |

#### rec-system - Recommendation

| Skill | Architecture | Vertical Applications |
|-------|-------------|----------------------|
| **item-to-item** | Item vector similarity | Similar products, related articles, similar videos |
| **user-to-item** | User vector + item vector + recall | Personalized recommendations, feeds, job matching |

#### data-analytics - Detection / Analytics

| Skill | Architecture | Vertical Applications |
|-------|-------------|----------------------|
| **duplicate-detection** | Batch vector comparison + threshold | Plagiarism detection, content deduplication, resume deduplication |
| **clustering** | Vector clustering analysis | Topic clustering, user segmentation, anomaly detection |

#### memory-system - Conversation / Memory

| Skill | Architecture | Vertical Applications |
|-------|-------------|----------------------|
| **chat-memory** | Conversation vectorization + time decay | Long-term assistants, customer service memory, game NPCs |

*Note: `multimodal-rag` and `video-search` are also in `multimodal-retrieval` plugin.*

## Tech Stack (Fixed)

No need to choose - we've already decided for you:

| Purpose | Technology | Reason |
|---------|------------|--------|
| Language | **Python** | Best ecosystem, seamless integration with Milvus/ML |
| External APIs | **FastAPI** | Simple and efficient, built-in Swagger docs |
| Data Processing | **Ray** | Faster than Spark, GPU support |
| Environment Management | **uv** | 100x faster than pip |
| Vector Storage | **Milvus / Zilliz Cloud** | Professional vector database |

## Quick Start Examples

### Semantic Search

```
I have 50,000 news headlines and want to do semantic search
```

The pilot will:
1. Confirm data (text, 50k items)
2. Confirm query (semantic search)
3. Auto-select: BGE model + Milvus Lite + HNSW index
4. Generate complete code

### Document Q&A

```
I have 200 PDF documents and want AI to answer questions about them
```

The pilot will:
1. Confirm data (PDF, 200 files)
2. Confirm query (Q&A)
3. Auto-select: PDF extraction + chunking + BGE + RAG
4. Generate complete code

### Image Search

```
100,000 product images, users upload images to find similar ones
```

The pilot will:
1. Confirm data (images, 100k)
2. Confirm query (image-to-image search)
3. Auto-select: CLIP model + Zilliz Cloud + HNSW
4. Generate complete code

## Directory Structure

```
milvus-marketplace/
├── .claude-plugin/
│   └── marketplace.json              # Marketplace definition (7 plugins)
├── plugins/
│   ├── core/                         # Core capabilities
│   │   ├── .claude-plugin/plugin.json
│   │   └── skills/
│   │       ├── pilot/                # Controller + methodology
│   │       ├── embedding/            # Atomic operators
│   │       ├── chunking/
│   │       ├── indexing/
│   │       ├── rerank/
│   │       ├── ray/
│   │       └── local-setup/
│   ├── rag-toolkit/                  # RAG solutions (4 skills)
│   │   ├── .claude-plugin/plugin.json
│   │   └── skills/
│   │       ├── rag/
│   │       ├── rag-with-rerank/
│   │       ├── multi-hop-rag/
│   │       └── agentic-rag/
│   ├── retrieval-system/             # Text search (4 skills)
│   │   ├── .claude-plugin/plugin.json
│   │   └── skills/
│   │       ├── semantic-search/
│   │       ├── hybrid-search/
│   │       ├── filtered-search/
│   │       └── multi-vector-search/
│   ├── multimodal-retrieval/         # Image/video/multimodal (4 skills)
│   │   ├── .claude-plugin/plugin.json
│   │   └── skills/
│   │       ├── image-search/
│   │       ├── text-to-image-search/
│   │       ├── video-search/
│   │       └── multimodal-rag/
│   ├── rec-system/                   # Recommendations (2 skills)
│   │   ├── .claude-plugin/plugin.json
│   │   └── skills/
│   │       ├── item-to-item/
│   │       └── user-to-item/
│   ├── memory-system/                # Long-term memory (1 skill)
│   │   ├── .claude-plugin/plugin.json
│   │   └── skills/
│   │       └── chat-memory/
│   └── data-analytics/               # Data analysis (2 skills)
│       ├── .claude-plugin/plugin.json
│       └── skills/
│           ├── duplicate-detection/
│           └── clustering/
└── README.md
```

## Installation Options

```bash
# Default: user-level, available to all projects
/plugin install core@milvus-marketplace

# Project-level: team sharing (committed to git)
claude plugin install core@milvus-marketplace --scope project

# Local-level: current project only (gitignored)
claude plugin install core@milvus-marketplace --scope local
```

## Managing Plugins

```bash
# View installed
/plugin

# Update marketplace
/plugin marketplace update milvus-marketplace

# Uninstall
/plugin uninstall core@milvus-marketplace

# Remove marketplace
/plugin marketplace remove milvus-marketplace
```

## Contributing

Contributions of new scenario solutions or core operators are welcome.

- New scenarios: Add under `plugins/<category>/skills/` (e.g., `plugins/rag-toolkit/skills/`)
- New operators: Add under `plugins/core/skills/`
- New vertical applications: Add under the corresponding scenario's `verticals/`
