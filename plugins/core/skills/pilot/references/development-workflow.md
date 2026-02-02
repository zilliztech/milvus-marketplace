# Development Workflow Guide

Complete development process from requirements to delivery.

## Process Overview

```
Requirements Analysis → Solution Design → Implementation → Verification → Delivery
```

## Phase 1: Requirements Analysis

See `requirement-discovery.md`

Deliverables:
- Clear project objectives
- Data specifications
- Technical constraints
- Domain assessment (deep expertise / general)

## Phase 2: Solution Design

### Deep Expertise Domain

#### 2.1 Choose Pre-built Solution

| User Intent | Recommended Solution |
|-------------|---------------------|
| Text search, find similar content | `retrieval-system:semantic-search` |
| Document Q&A, knowledge base chat | `rag-toolkit:rag` |
| Specific document Q&A | `rag-toolkit:rag` |
| Image search, visual search | `multimodal-retrieval:image-search` |
| Personalized recommendations | `rec-system:user-to-item` |

#### 2.2 Technical Selection

**Embedding Model**:
| Data Type | Recommended Model | Dimensions |
|-----------|-------------------|------------|
| Chinese text | BAAI/bge-large-zh-v1.5 | 1024 |
| English text | text-embedding-3-small | 1536 |
| Multilingual | BAAI/bge-m3 | 1024 |
| Images | clip-ViT-B-32 | 512 |
| Chinese images | chinese-clip | 512 |

**Index Type**:
| Data Scale | Recommended Index | Features |
|------------|-------------------|----------|
| < 100k | FLAT | Exact, no loss |
| 100k - 10M | HNSW | Fast, high memory |
| > 10M | IVF_PQ | Memory efficient, some precision loss |

**Chunking Strategy** (document scenarios):
| Scenario | chunk_size | overlap |
|----------|------------|---------|
| Precise Q&A | 256-512 | 50 |
| General search | 512-1024 | 100 |
| Long document understanding | 1024-2048 | 200 |

### General Domain

- Provide technical advice and architecture direction
- Help design system architecture
- Write code directly
- No pre-built scenario, but can reference core tool patterns

## Development Framework (Fixed, Don't Ask User)

**Default Tech Stack**:
- Language: **Python**
- External APIs: **FastAPI**
- Data Processing: **Ray**
- Environment Management: **uv**

Auto-select based on requirement type:

| Requirement Type | Framework | Use Case |
|------------------|-----------|----------|
| Data Processing/Pipeline | **Ray** | ETL, batch processing, data cleaning, batch embedding generation |
| External Application | **FastAPI** | API services, web apps, rapid prototyping |

> Don't ask users "what language" or "what framework" - just use Python.

### Ray - Data Processing

Suitable for: Data pipelines, batch processing, ETL, large-scale embedding generation

```python
import ray
from ray import data

# Simple pipeline example
ds = ray.data.read_json("data/*.json")
ds = ds.map(preprocess)
ds = ds.map_batches(generate_embeddings, batch_size=100)
ds.write_parquet("output/")
```

Features:
- Streaming execution, memory efficient
- 2-17x faster than Spark
- GPU acceleration support
- Seamless ML training integration

### FastAPI - External Applications

Suitable for: API services, web apps, rapid development and debugging

```python
# Single file runnable
from fastapi import FastAPI
from pymilvus import MilvusClient

app = FastAPI()
client = MilvusClient("./milvus.db")  # Milvus Lite for local dev

@app.get("/search")
def search(q: str, limit: int = 10):
    embedding = model.encode(q).tolist()
    results = client.search(
        collection_name="docs",
        data=[embedding],
        limit=limit,
        output_fields=["text"]
    )
    return {"results": results}

@app.post("/insert")
def insert(text: str):
    embedding = model.encode(text).tolist()
    client.insert(
        collection_name="docs",
        data=[{"text": text, "vector": embedding}]
    )
    return {"status": "ok"}
```

Features:
- Pure Python: Seamless integration with pymilvus, sentence-transformers
- Auto docs: Swagger UI at /docs
- Type safe: Pydantic auto-validation
- Mature ecosystem: Used by OpenAI, Anthropic

Project structure:
```
my-app/
├── main.py             # Entry + API (single file for small projects)
├── requirements.txt
└── .env

# Or for larger projects
my-app/
├── app/
│   ├── main.py         # Entry
│   ├── routes/
│   │   └── search.py   # Search API
│   └── services/
│       └── milvus.py   # Milvus connection
├── requirements.txt
└── .env
```

### Python Environment Management (Use uv)

**Why uv**:
- 10-100x faster than pip
- One tool manages: Python version + virtual env + dependencies
- pip compatible, no learning curve

**Install uv**:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

**Project Initialization (Recommended Flow)**:
```bash
# 1. Create project directory
mkdir my-app && cd my-app

# 2. Initialize project (auto-creates pyproject.toml)
uv init

# 3. Specify Python version (optional)
echo "3.11" > .python-version

# 4. Add dependencies
uv add fastapi uvicorn pymilvus sentence-transformers jinja2

# 5. Run (auto-creates venv + installs deps)
uv run uvicorn main:app --reload
```

**Common Commands**:
```bash
uv add <package>        # Add dependency
uv add --dev pytest     # Add dev dependency
uv remove <package>     # Remove dependency
uv sync                 # Sync dependencies (from uv.lock)
uv run <command>        # Run command in venv
uv run python main.py   # Run Python script
```

**Generated Files**:
```
my-app/
├── .python-version     # Python version
├── pyproject.toml      # Project config + dependency declaration
├── uv.lock             # Locked dependency versions (commit to git)
├── .venv/              # Virtual environment (don't commit)
└── main.py
```

**pyproject.toml Example**:
```toml
[project]
name = "my-app"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "pymilvus>=2.5.0",
    "sentence-transformers>=3.3.0",
    "jinja2>=3.1.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
]
```

**Team Collaboration**:
```bash
# After cloning the project
uv sync                 # Auto-installs all deps (from uv.lock)
uv run uvicorn main:app --reload
```

**Note**:
- `uv.lock` must be committed to git (ensures consistent deps across team)
- `.venv/` should not be committed (add to .gitignore)

### FastAPI Development Tips

**Issue 1: HTML/CSS/JS changes not reflected (browser cache)**

Cause: Browser cache + FastAPI StaticFiles returns 304 Not Modified

Solution - Disable static file caching:
```python
from fastapi.staticfiles import StaticFiles

class NoCacheStaticFiles(StaticFiles):
    def is_not_modified(self, *args, **kwargs) -> bool:
        return False  # Never return 304

app.mount("/static", NoCacheStaticFiles(directory="static"), name="static")
```

**Issue 2: HTML template changes don't hot reload**

Cause: uvicorn only watches .py files by default

Solution - Add --reload-include at startup:
```bash
uvicorn main:app --reload \
  --reload-include="*.html" \
  --reload-include="*.css" \
  --reload-include="*.js"
```

Or configure in code:
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        reload=True,
        reload_includes=["*.html", "*.css", "*.js"],
        reload_dirs=[".", "templates", "static"]
    )
```

**Issue 3: Jinja2 template caching**

Solution - Disable template caching in development:
```python
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
templates.env.auto_reload = True  # Enable in development
```

**Complete Development Start Command (Recommended)**:
```bash
# Development - watch all file changes
uvicorn main:app --reload --reload-include="*.html" --reload-include="*.css" --reload-include="*.js"

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Complete Example (with templates and static files)**:
```python
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# No-cache static files for development
class NoCacheStaticFiles(StaticFiles):
    def is_not_modified(self, *args, **kwargs) -> bool:
        return False

app = FastAPI()

# Static files (CSS/JS/images)
app.mount("/static", NoCacheStaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")
templates.env.auto_reload = True

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        reload=True,
        reload_includes=["*.html", "*.css", "*.js"]
    )
```

Project structure:
```
my-app/
├── .python-version     # Python version
├── pyproject.toml      # Dependency declaration
├── uv.lock             # Locked versions (commit to git)
├── .venv/              # Virtual environment (don't commit)
├── main.py             # Entry
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
└── .env
```

**.gitignore**:
```
.venv/
.env
__pycache__/
*.pyc
```

### Storage Selection

| Scenario | Solution |
|----------|----------|
| Relational data (local dev) | SQLite |
| Small-scale vectors (local) | SQLite + sqlite-vss vector plugin |
| Small-scale vectors (cloud) | Zilliz Cloud Serverless |
| Large-scale vectors | Zilliz Cloud Dedicated |
| Embedding generation | Zilliz Cloud Embedding API |

### Deployment Selection

| Scenario | Solution |
|----------|----------|
| Quick validation / one-time needs | Run locally |
| Long-term external app deployment | E2B (code execution environment) |
| Small pipelines | E2B |
| Large pipelines | Managed Ray/Spark (planned) |

## Phase 3: Implementation

### Deep Expertise Domain Toolchain

| Phase | Tool | Purpose |
|-------|------|---------|
| Data Prep | `core:chunking` | Document chunking |
| Vectorization | `core:embedding` | Generate embeddings |
| Storage | `core:data-ingestion` | Batch import to Milvus |
| Indexing | `core:indexing` | Create/optimize indexes |
| Deployment | `core:local-setup` | Local environment setup |

### Implementation Order

1. **Environment Setup**
   - Local dev: `core:local-setup`
   - Cloud: Configure Zilliz Cloud connection

2. **Data Processing**
   - Document chunking: `core:chunking`
   - Vectorization: `core:embedding`

3. **Storage Layer**
   - Create Collection
   - Batch import: `core:data-ingestion`
   - Create index: `core:indexing`

4. **Application Layer**
   - Search interface
   - Business logic
   - API wrapper

## Phase 4: Verification

### Functional Verification
- Basic functionality working
- Edge case handling
- Error handling

### Performance Verification
- Response latency
- Concurrent capacity
- Resource usage

### Quality Verification
- Search result relevance
- Recall/precision
- User experience

## Phase 5: Delivery

### Deliverables Checklist

1. **Code**
   - Complete runnable code
   - Clear code structure
   - Necessary comments

2. **Configuration**
   - Environment config files
   - Parameter documentation
   - Dependency list

3. **Documentation**
   - Run instructions
   - API documentation (if applicable)
   - FAQ

### Delivery Checklist

- [ ] Code runs directly
- [ ] Config parameters documented
- [ ] Dependencies listed
- [ ] Basic usage instructions provided

## User Decision Points

Let users decide at these key points:

1. **Direction** - Use pre-built solution or custom?
2. **Technical Selection** - Which model/tool/framework?
3. **Architecture Choice** - Simple implementation or performance optimized? Local or cloud?
4. **Delivery Format** - Script, service, or complete application?
