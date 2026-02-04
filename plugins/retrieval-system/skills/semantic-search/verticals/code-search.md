# Code Search

> Search code by natural language description or find similar code snippets.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Code Language

<ask_user>
What programming language(s) is your codebase?

| Option | Notes |
|--------|-------|
| **Python** | Most model support |
| **Go** | Good support |
| **JavaScript/TypeScript** | Good support |
| **Java** | Good support |
| **Multi-language** | Use general model |
</ask_user>

### 2. Search Granularity

<ask_user>
What level do you want to search at?

| Granularity | Best For |
|-------------|----------|
| **Function/Method** (recommended) | Finding specific implementations |
| **File** | Understanding file purposes |
| **Code Block** (by lines) | Fine-grained search |
</ask_user>

### 3. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality, fast | Requires API key, costs money |
| **Local Model** | Free, offline | Model download required |

Do you have an OpenAI API Key?
</ask_user>

### 4. Code Embedding Model (if local)

<ask_user>
Choose a code embedding model:

| Model | Size | Notes |
|-------|------|-------|
| `BAAI/bge-base-en-v1.5` (recommended) | 440MB | General text, works well for code |
| `microsoft/codebert-base` | 500MB | Code-specific, understands structure |
| `Salesforce/codet5p-110m-embedding` | 440MB | Code-specific, lighter |
| `jinaai/jina-embeddings-v2-base-code` | 550MB | Optimized for code search |
</ask_user>

### 5. Data Source

<ask_user>
Where is your code?

| Source | Setup |
|--------|-------|
| **Local Repository** | Point to directory |
| **GitHub Repository** | Clone URL |
| **CodeSearchNet Dataset** | For benchmarking |
</ask_user>

### 6. Data Scale

<ask_user>
How large is your codebase?

- Each function ≈ 1 vector
- Example: 5000 functions = 5K vectors

| Vector Count | Recommended Milvus |
|--------------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 7. Project Setup

<ask_user>
Choose project management:

| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production projects |
| **pip** | Quick prototypes |
</ask_user>

---

## Dependencies

Based on your choices:

### OpenAI + uv
```bash
uv init code-search
cd code-search
uv add pymilvus openai gitpython
```

### Local Model + uv
```bash
uv init code-search
cd code-search
uv add pymilvus sentence-transformers gitpython
```

### With tree-sitter for better parsing (optional)
```bash
uv add tree-sitter tree-sitter-python  # or tree-sitter-go, etc.
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

### Step 2: Extract Code (Simple Regex Approach)

```python
import os
import re
from pathlib import Path

def extract_python_functions(file_path: str) -> list[dict]:
    """Extract functions from a Python file using regex."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Simple pattern for Python functions
    pattern = r'(def\s+(\w+)\s*\([^)]*\).*?(?=\ndef\s|\nclass\s|\Z))'
    matches = re.findall(pattern, content, re.DOTALL)

    functions = []
    for func_code, func_name in matches:
        # Extract docstring if exists
        docstring_match = re.search(r'"""(.*?)"""', func_code, re.DOTALL)
        docstring = docstring_match.group(1).strip() if docstring_match else ""

        functions.append({
            "name": func_name,
            "code": func_code.strip(),
            "docstring": docstring,
            "file_path": file_path
        })
    return functions

def scan_repository(repo_path: str, extensions: list[str] = [".py"]) -> list[dict]:
    """Scan repository and extract all functions."""
    all_functions = []

    for root, dirs, files in os.walk(repo_path):
        # Skip common non-code directories
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", "venv", ".venv"}]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    functions = extract_python_functions(file_path)
                    all_functions.extend(functions)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return all_functions
```

### Step 3: Clone from GitHub (optional)

```python
from git import Repo

def clone_repo(url: str, target_dir: str = "repo") -> str:
    """Clone a GitHub repository."""
    if os.path.exists(target_dir):
        print(f"Directory {target_dir} exists, pulling latest...")
        repo = Repo(target_dir)
        repo.remotes.origin.pull()
    else:
        print(f"Cloning {url}...")
        Repo.clone_from(url, target_dir)
    return target_dir

# Example
repo_path = clone_repo("https://github.com/milvus-io/pymilvus")
functions = scan_repository(repo_path)
```

### Step 4: Create Search Text

```python
def create_search_text(func: dict) -> str:
    """Create text representation for embedding."""
    parts = []

    # Function name (split camelCase/snake_case)
    name = func["name"]
    name_words = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)  # camelCase
    name_words = name_words.replace("_", " ")  # snake_case
    parts.append(f"Function: {name_words}")

    # Docstring
    if func["docstring"]:
        parts.append(f"Description: {func['docstring']}")

    # Code (first 500 chars for context)
    parts.append(f"Code:\n{func['code'][:500]}")

    return "\n".join(parts)
```

### Step 5: Index into Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient("code.db")  # Milvus Lite
# client = MilvusClient(uri="http://localhost:19530")  # Standalone

client.create_collection(
    collection_name="code",
    dimension=DIMENSION,
    auto_id=True
)

def index_functions(functions: list[dict], batch_size: int = 50):
    """Embed and index functions."""
    for i in range(0, len(functions), batch_size):
        batch = functions[i:i+batch_size]
        texts = [create_search_text(f) for f in batch]
        vectors = embed(texts)

        data = [
            {
                "vector": vec,
                "name": f["name"],
                "file_path": f["file_path"],
                "code": f["code"][:2000],  # Truncate for storage
                "docstring": f["docstring"][:500]
            }
            for vec, f in zip(vectors, batch)
        ]
        client.insert(collection_name="code", data=data)

    print(f"Indexed {len(functions)} functions")

# Index
functions = scan_repository("./repo")
index_functions(functions)
```

### Step 6: Search

```python
def search_code(query: str, top_k: int = 5):
    """Search code by natural language query."""
    query_vector = embed([query])[0]
    results = client.search(
        collection_name="code",
        data=[query_vector],
        limit=top_k,
        output_fields=["name", "file_path", "code", "docstring"]
    )
    return results[0]

def print_results(results):
    """Pretty print search results."""
    for i, hit in enumerate(results, 1):
        entity = hit["entity"]
        print(f"\n{'='*60}")
        print(f"#{i} {entity['name']} (score: {hit['distance']:.3f})")
        print(f"File: {entity['file_path']}")
        if entity["docstring"]:
            print(f"Doc: {entity['docstring'][:100]}...")
        print(f"\n{entity['code'][:300]}...")
```

---

## Run Example

```python
# Clone and index a repository
repo_path = clone_repo("https://github.com/milvus-io/pymilvus")
functions = scan_repository(repo_path)
index_functions(functions)

# Natural language code search
print_results(search_code("connect to milvus server"))
print_results(search_code("insert vectors into collection"))
print_results(search_code("search with filter expression"))
```

---

## Advanced: Using CodeSearchNet Dataset

```python
from datasets import load_dataset

def load_codesearchnet(language: str = "python", split: str = "train", max_samples: int = 10000):
    """Load CodeSearchNet dataset from HuggingFace."""
    ds = load_dataset("code_search_net", language, split=split)

    functions = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        functions.append({
            "name": item["func_name"],
            "code": item["func_code_string"],
            "docstring": item["func_documentation_string"] or "",
            "file_path": item["func_code_url"]
        })
    return functions

# Load and index
functions = load_codesearchnet("python", max_samples=5000)
index_functions(functions)
```
