# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Milvus Marketplace is a **Claude Code skills marketplace** for building AI applications in the data retrieval domain. It is not a traditional software project with source code — it is a structured collection of markdown-based skills, workflows, and code templates organized around the Milvus vector database.

The project is maintained by **Zilliz** and licensed under Apache 2.0.

## Architecture

### Multi-Plugin Marketplace

```
marketplace.json              → defines 7 plugins
├── plugins/core/             → pilot (controller) + atomic operators
├── plugins/rag-toolkit/      → RAG solutions (4 skills)
├── plugins/memory-system/    → Long-term memory (1 skill)
├── plugins/retrieval-system/ → Text search (5 skills)
├── plugins/multimodal-retrieval/ → Image/video/multimodal (4 skills)
├── plugins/rec-system/       → Recommendation (2 skills)
└── plugins/data-analytics/   → Data analysis (2 skills)
```

**Core plugin** (`plugins/core/skills/`): The `pilot` skill is the main controller that routes user requests. Other skills are atomic operators: `embedding`, `chunking`, `indexing`, `rerank`, `ray`, `local-setup`.

**Scenario plugins**: Pre-built solutions organized by use case:

| Plugin | Skills | Description |
|--------|--------|-------------|
| `rag-toolkit` | rag, rag-with-rerank, agentic-rag, multi-hop-rag | RAG and knowledge base solutions |
| `memory-system` | chat-memory | Long-term memory for chatbots |
| `retrieval-system` | semantic-search, filtered-search, hybrid-search, multi-vector-search, contextual-retrieval | Text search variations |
| `multimodal-retrieval` | image-search, text-to-image-search, video-search, multimodal-rag | Image, video, and multimodal |
| `rec-system` | item-to-item, user-to-item | Recommendation systems |
| `data-analytics` | duplicate-detection, clustering | Data analysis patterns |

### Skill File Structure

Each skill follows this pattern:
```
skills/<skill-name>/
├── SKILL.md              # Main skill definition (YAML frontmatter + markdown)
├── verticals/            # Optional: vertical application guides
│   ├── <vertical>.md
│   └── ...
└── references/           # Optional: detailed reference docs
```

The YAML frontmatter in `SKILL.md` contains `name` and `description` fields used for skill discovery and trigger matching.

### Documentation Organization Guidelines

The project uses a consistent documentation structure across all skills:

#### 1. SKILL.md (Required)

Every skill must have a `SKILL.md` file with:
- **YAML frontmatter**: Contains `name` and `description` for skill discovery
- **Main content**: Core concepts, workflow, implementation examples
- **Progressive disclosure**: Start simple (60%), provide optimization paths

Example frontmatter:
```yaml
---
name: semantic-search
description: "Use when user wants to build semantic/text search. Triggers on: semantic search, text search, natural language search, vector search."
---
```

#### 2. verticals/ Directory (Optional - Scenario Skills Only)

**When to create**:
- ✅ For **scenario skills** (rag, semantic-search, image-search, etc.)
- ❌ Not for **core operators** (chunking, indexing, rerank, etc.)
- ❌ Not for **pilot** (controller skill)
- ⚠️ Exception: `embedding` has verticals because model selection depends heavily on use case

**Purpose**: Provide industry/domain-specific configurations and examples.

**Verticals file format**:
Verticals use an **interactive setup format** with `<ask_user>` blocks for gathering user requirements:

```markdown
# Use Case Title

> Brief description

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Question Category

<ask_user>
What is your [specific choice]?

| Option | Notes |
|--------|-------|
| **Option A** | Description |
| **Option B** | Description |
</ask_user>

### 2. Next Question

<ask_user>
Choose your approach:

| Method | Pros | Cons |
|--------|------|------|
| **Method A** | Benefits | Limitations |
| **Method B** | Benefits | Limitations |
</ask_user>

---

## Dependencies

Based on choices above, show relevant commands...

## End-to-End Implementation

Complete working code with inline comments...
```

**Key features**:
- `<ask_user>` blocks present choices as tables
- Progressive questioning (gather requirements before implementation)
- Clear separation: questions → dependencies → implementation
- All verticals must be runnable end-to-end examples

**Naming convention**: Use lowercase with hyphens (e.g., `legal-contracts.md`, `ecommerce.md`, `code-search.md`)

#### 3. references/ Directory (Optional - Complex Skills Only)

**When to create**:
- ✅ For **complex scenario skills** that need architectural deep-dives
- ✅ For **pilot** (has multiple reference docs: requirement-discovery, solution-matching, etc.)
- ❌ Not for simple skills that can be fully explained in SKILL.md

**Purpose**: Provide in-depth technical references, comparison tables, and advanced patterns.

**Common reference topics**:
- `<topic>-strategies.md` - Different approaches and trade-offs
- `<component>-comparison.md` - Model/algorithm comparisons
- `advanced-patterns.md` - Optimization techniques
- `best-practices.md` - Production recommendations

**Examples**:
- `pilot/references/` - Has 6 reference docs (requirement-discovery, solution-matching, etc.)
- `rag/references/` - chunk-strategies, embedding-models, advanced-patterns
- `semantic-search/references/` - similarity-metrics
- `hybrid-search/references/` - fusion-strategies

**When NOT to create**: If the content fits in SKILL.md or can be a vertical, don't create references.

### Scenario Classification

Scenarios are classified by **architectural differences** (different code structure/pipeline), not by industry. Variations that only differ in model/parameters are handled as verticals within the same scenario.

### Routing Flow

The `pilot` skill asks exactly two questions (data type + query type), then auto-routes to either:
- A matching scenario skill (if one fits)
- A combination of core operators (if no scenario matches)

## Key Configuration Files

| File | Purpose |
|------|---------|
| `.claude-plugin/marketplace.json` | Root marketplace definition, lists all plugins |
| `plugins/<name>/.claude-plugin/plugin.json` | Plugin metadata |

## Fixed Tech Stack (for generated code)

All generated application code uses this stack — no user choice involved:

- **Python** — language
- **FastAPI** — API framework
- **Ray** — data processing
- **uv** — package management
- **Milvus / Zilliz Cloud** — vector storage

## Contributing New Content

- **New skill in existing category**: Add directory under `plugins/<category>/skills/<skill-name>/` with a `SKILL.md`
- **New core operator**: Add directory under `plugins/core/skills/<operator-name>/` with a `SKILL.md`
- **New vertical**: Add `<vertical>.md` under the relevant skill's `verticals/` directory
- **New category plugin**: Create `plugins/<category>/` with `.claude-plugin/plugin.json` and `skills/` directory, then add to `marketplace.json`
- Update the routing table in `plugins/core/skills/pilot/SKILL.md` when adding new skills

## Design Principles

- **Ship 60% first, iterate to 80%**: Skills should generate working code quickly, then offer optimization paths
- **Don't let users choose tech stack**: The pilot auto-selects models, storage, and index based on data characteristics
- **Scenarios as cache**: Pre-built solutions are reused directly when matched; only fall back to composing operators when no scenario fits
