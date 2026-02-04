# Milvus Codebase Search

> Search Milvus source code by functionality, component, or implementation detail.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Code Language Focus

<ask_user>
Which part of Milvus codebase do you want to search?

| Component | Language | Description |
|-----------|----------|-------------|
| **Core engine** | Go | Main server, indexing, query |
| **Python SDK** | Python | pymilvus client library |
| **Both** | Go + Python | Full codebase |
</ask_user>

### 2. Search Granularity

<ask_user>
What level do you want to search?

| Granularity | Best For |
|-------------|----------|
| **Function/Method** (recommended) | Finding specific implementations |
| **File** | Understanding file purposes |
| **Package/Module** | Architecture overview |
</ask_user>

### 3. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download needed |
</ask_user>

### 4. Local Model (if local)

<ask_user>
Choose embedding model:

| Model | Notes |
|-------|-------|
| `BAAI/bge-base-en-v1.5` (recommended) | General text, works well for code |
| `microsoft/codebert-base` | Code-specific |
</ask_user>

### 5. Project Setup

<ask_user>
Choose project management:

| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production projects |
| **pip** | Quick prototypes |
</ask_user>

---

## Dependencies

### OpenAI + uv
```bash
uv init milvus-code-search
cd milvus-code-search
uv add pymilvus openai gitpython
```

### Local Model + uv
```bash
uv init milvus-code-search
cd milvus-code-search
uv add pymilvus sentence-transformers gitpython
```

---

## End-to-End Implementation

### Step 1: Configure Embedding

```python
# === Choose ONE ===

# Option A: OpenAI API
from openai import OpenAI
client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

# Option B: Local Model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()

DIMENSION = 768
```

### Step 2: Clone Milvus Repository

```python
import os
from git import Repo

def clone_milvus(target_dir: str = "milvus"):
    """Clone Milvus repository."""
    repo_url = "https://github.com/milvus-io/milvus"

    if os.path.exists(target_dir):
        print("Repository exists, pulling latest...")
        repo = Repo(target_dir)
        repo.remotes.origin.pull()
    else:
        print("Cloning Milvus (this may take a while)...")
        Repo.clone_from(repo_url, target_dir, depth=1)  # Shallow clone

    return target_dir

def clone_pymilvus(target_dir: str = "pymilvus"):
    """Clone PyMilvus repository."""
    repo_url = "https://github.com/milvus-io/pymilvus"

    if os.path.exists(target_dir):
        repo = Repo(target_dir)
        repo.remotes.origin.pull()
    else:
        Repo.clone_from(repo_url, target_dir, depth=1)

    return target_dir

milvus_path = clone_milvus()
pymilvus_path = clone_pymilvus()
```

### Step 3: Extract Go Functions

```python
import re

def extract_go_functions(file_path: str) -> list[dict]:
    """Extract functions from a Go file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    functions = []

    # Match Go function definitions
    # func (receiver) Name(params) returns { body }
    pattern = r'((?://[^\n]*\n)*)\s*func\s+(?:\([^)]+\)\s+)?(\w+)\s*\([^)]*\)[^{]*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'

    for match in re.finditer(pattern, content, re.MULTILINE):
        comment = match.group(1).strip()
        func_name = match.group(2)
        body = match.group(3)

        # Clean up comment
        comment = re.sub(r'^//\s*', '', comment, flags=re.MULTILINE)

        functions.append({
            "name": func_name,
            "comment": comment[:500],
            "body": body[:1000],
            "file_path": file_path
        })

    return functions

def scan_go_files(repo_path: str) -> list[dict]:
    """Scan all Go files in repository."""
    all_functions = []
    skip_dirs = {".git", "vendor", "build", "cmake_build", "third_party"}

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            if file.endswith(".go") and not file.endswith("_test.go"):
                file_path = os.path.join(root, file)
                try:
                    functions = extract_go_functions(file_path)
                    all_functions.extend(functions)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

    return all_functions
```

### Step 4: Extract Python Functions

```python
def extract_python_functions(file_path: str) -> list[dict]:
    """Extract functions and classes from a Python file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    functions = []

    # Match Python function definitions
    pattern = r'((?:^[ \t]*#[^\n]*\n)*(?:^[ \t]*"""[\s\S]*?"""\n)?)?^([ \t]*)def\s+(\w+)\s*\([^)]*\)[^:]*:(.*?)(?=\n\2(?:def|class)\s|\Z)'

    for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
        docstring = match.group(1) or ""
        func_name = match.group(3)
        body = match.group(4)

        # Extract docstring
        doc_match = re.search(r'"""(.*?)"""', docstring, re.DOTALL)
        doc = doc_match.group(1).strip() if doc_match else ""

        functions.append({
            "name": func_name,
            "comment": doc[:500],
            "body": body[:1000].strip(),
            "file_path": file_path
        })

    return functions

def scan_python_files(repo_path: str) -> list[dict]:
    """Scan all Python files in repository."""
    all_functions = []
    skip_dirs = {".git", "__pycache__", "venv", ".venv", "build", "dist", ".eggs"}

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            if file.endswith(".py") and not file.startswith("test_"):
                file_path = os.path.join(root, file)
                try:
                    functions = extract_python_functions(file_path)
                    all_functions.extend(functions)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

    return all_functions
```

### Step 5: Create Search Text

```python
def create_function_text(func: dict) -> str:
    """Create searchable text from function data."""
    parts = []

    # Function name
    name_words = re.sub(r'([a-z])([A-Z])', r'\1 \2', func["name"])
    name_words = name_words.replace("_", " ")
    parts.append(f"Function: {name_words}")

    # File path for context
    rel_path = func["file_path"].split("/", 1)[-1] if "/" in func["file_path"] else func["file_path"]
    parts.append(f"File: {rel_path}")

    # Comment/docstring
    if func["comment"]:
        parts.append(f"Description: {func['comment']}")

    # Code body
    parts.append(f"Code:\n{func['body'][:500]}")

    return "\n".join(parts)
```

### Step 6: Index into Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient("milvus_code.db")  # Milvus Lite

client.create_collection(
    collection_name="milvus_code",
    dimension=DIMENSION,
    auto_id=True
)

def index_functions(functions: list[dict], batch_size: int = 50):
    """Embed and index functions."""
    for i in range(0, len(functions), batch_size):
        batch = functions[i:i+batch_size]
        texts = [create_function_text(f) for f in batch]
        vectors = embed(texts)

        data = [
            {
                "vector": vec,
                "name": f["name"],
                "file_path": f["file_path"],
                "comment": f["comment"][:500],
                "body": f["body"][:1000]
            }
            for vec, f in zip(vectors, batch)
        ]
        client.insert(collection_name="milvus_code", data=data)
        print(f"Indexed {i + len(batch)}/{len(functions)}")

# Scan and index
go_functions = scan_go_files(milvus_path)
py_functions = scan_python_files(pymilvus_path)
all_functions = go_functions + py_functions

print(f"Found {len(go_functions)} Go functions, {len(py_functions)} Python functions")
index_functions(all_functions)
```

### Step 7: Search

```python
def search_code(query: str, top_k: int = 5, language: str = None):
    """Search Milvus codebase by natural language query."""
    query_vector = embed([query])[0]

    # Filter by file extension to filter language
    filter_expr = None
    if language == "go":
        filter_expr = 'file_path like "%.go"'
    elif language == "python":
        filter_expr = 'file_path like "%.py"'

    results = client.search(
        collection_name="milvus_code",
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=["name", "file_path", "comment", "body"]
    )
    return results[0]

def print_results(results):
    for i, hit in enumerate(results, 1):
        e = hit["entity"]
        print(f"\n{'='*60}")
        print(f"#{i} {e['name']} (score: {hit['distance']:.3f})")
        print(f"File: {e['file_path']}")
        if e["comment"]:
            print(f"Doc: {e['comment'][:150]}...")
        print(f"\n{e['body'][:300]}...")
```

---

## Run Example

```python
# Clone and index
milvus_path = clone_milvus()
pymilvus_path = clone_pymilvus()

go_functions = scan_go_files(milvus_path)
py_functions = scan_python_files(pymilvus_path)
all_functions = go_functions + py_functions
index_functions(all_functions)

# Search examples
print_results(search_code("create collection"))
print_results(search_code("insert vectors"))
print_results(search_code("build index", language="go"))
print_results(search_code("connect to server", language="python"))
```

---

## Advanced: Search by Component

```python
MILVUS_COMPONENTS = {
    "proxy": "internal/proxy",
    "datanode": "internal/datanode",
    "querynode": "internal/querynode",
    "indexnode": "internal/indexnode",
    "rootcoord": "internal/rootcoord",
    "datacoord": "internal/datacoord"
}

def search_component(query: str, component: str, top_k: int = 5):
    """Search within a specific Milvus component."""
    path_prefix = MILVUS_COMPONENTS.get(component, component)
    filter_expr = f'file_path like "%{path_prefix}%"'

    query_vector = embed([query])[0]
    results = client.search(
        collection_name="milvus_code",
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=["name", "file_path", "comment", "body"]
    )
    return results[0]

# Search in specific component
print_results(search_component("segment loading", "querynode"))
```
