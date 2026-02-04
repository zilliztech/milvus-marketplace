# GitHub Repository Search

> Search open source projects by description, features, or use case.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Search Scope

<ask_user>
What do you want to search?

| Scope | Description |
|-------|-------------|
| **README content** | Search by project documentation |
| **Repository metadata** | Search by name, description, topics |
| **Both** (recommended) | Full-text + metadata search |
</ask_user>

### 2. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality, fast | Requires API key |
| **Local Model** | Free, offline | Model download needed |
</ask_user>

### 3. Local Model (if local)

<ask_user>
Choose embedding model:

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast prototyping |
| `BAAI/bge-base-en-v1.5` (recommended) | 768 | 440MB | Higher quality |
</ask_user>

### 4. Data Source

<ask_user>
How do you want to collect repositories?

| Source | Notes |
|--------|-------|
| **GitHub Search API** | Search trending/popular repos |
| **Curated list** | Awesome lists, specific topics |
| **Organization repos** | All repos from an org |
</ask_user>

### 5. Data Scale

<ask_user>
How many repositories do you want to index?

- Each repo = 1-5 vectors (metadata + README chunks)

| Repo Count | Recommended Milvus |
|------------|-------------------|
| < 50K | **Milvus Lite** |
| 50K - 1M | **Milvus Standalone** |
| > 1M | **Zilliz Cloud** |
</ask_user>

### 6. Project Setup

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
uv init github-search
cd github-search
uv add pymilvus openai PyGithub requests
```

### Local Model + uv
```bash
uv init github-search
cd github-search
uv add pymilvus sentence-transformers PyGithub requests
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

### Step 2: Fetch Repositories from GitHub

```python
from github import Github
import os
import time

# Initialize GitHub client
# Get token from https://github.com/settings/tokens (optional but recommended for rate limits)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
g = Github(GITHUB_TOKEN) if GITHUB_TOKEN else Github()

def search_github_repos(query: str, max_repos: int = 100) -> list[dict]:
    """Search GitHub repositories."""
    repos = []

    results = g.search_repositories(query=query, sort="stars", order="desc")

    for i, repo in enumerate(results):
        if i >= max_repos:
            break

        try:
            # Get README content
            readme_content = ""
            try:
                readme = repo.get_readme()
                readme_content = readme.decoded_content.decode("utf-8")[:5000]  # Limit size
            except:
                pass

            repos.append({
                "name": repo.full_name,
                "description": repo.description or "",
                "url": repo.html_url,
                "stars": repo.stargazers_count,
                "language": repo.language or "",
                "topics": repo.get_topics(),
                "readme": readme_content
            })

            # Rate limiting
            if i % 30 == 0 and i > 0:
                time.sleep(1)

        except Exception as e:
            print(f"Error fetching {repo.full_name}: {e}")

    return repos

def fetch_trending_repos(languages: list[str] = None, max_per_lang: int = 100) -> list[dict]:
    """Fetch trending repos across languages."""
    all_repos = []

    if languages is None:
        languages = ["python", "javascript", "go", "rust", "java"]

    for lang in languages:
        print(f"Fetching {lang} repositories...")
        query = f"language:{lang} stars:>100"
        repos = search_github_repos(query, max_repos=max_per_lang)
        all_repos.extend(repos)
        time.sleep(2)  # Rate limiting between queries

    return all_repos

# Fetch trending repos
repos = fetch_trending_repos(languages=["python", "go"], max_per_lang=50)
print(f"Fetched {len(repos)} repositories")
```

### Step 3: Create Search Text

```python
def create_repo_text(repo: dict) -> str:
    """Create searchable text from repository data."""
    parts = []

    # Name and description
    parts.append(f"Repository: {repo['name']}")
    if repo["description"]:
        parts.append(f"Description: {repo['description']}")

    # Topics
    if repo["topics"]:
        parts.append(f"Topics: {', '.join(repo['topics'])}")

    # Language
    if repo["language"]:
        parts.append(f"Language: {repo['language']}")

    # README excerpt
    if repo["readme"]:
        # Take first 1000 chars of README
        readme_excerpt = repo["readme"][:1000].replace("\n", " ")
        parts.append(f"README: {readme_excerpt}")

    return "\n".join(parts)
```

### Step 4: Index into Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient("github_repos.db")  # Milvus Lite

client.create_collection(
    collection_name="repos",
    dimension=DIMENSION,
    auto_id=True
)

def index_repos(repos: list[dict], batch_size: int = 50):
    """Embed and index repositories."""
    for i in range(0, len(repos), batch_size):
        batch = repos[i:i+batch_size]
        texts = [create_repo_text(r) for r in batch]
        vectors = embed(texts)

        data = [
            {
                "vector": vec,
                "name": r["name"],
                "description": r["description"][:500] if r["description"] else "",
                "url": r["url"],
                "stars": r["stars"],
                "language": r["language"],
                "topics": ",".join(r["topics"][:10])
            }
            for vec, r in zip(vectors, batch)
        ]
        client.insert(collection_name="repos", data=data)
        print(f"Indexed {i + len(batch)}/{len(repos)}")

index_repos(repos)
```

### Step 5: Search

```python
def search_repos(query: str, top_k: int = 10, language: str = None, min_stars: int = None):
    """Search repositories by natural language query."""
    query_vector = embed([query])[0]

    # Build filter
    filters = []
    if language:
        filters.append(f'language == "{language}"')
    if min_stars:
        filters.append(f'stars >= {min_stars}')
    filter_expr = " and ".join(filters) if filters else None

    results = client.search(
        collection_name="repos",
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=["name", "description", "url", "stars", "language", "topics"]
    )
    return results[0]

def print_results(results):
    for i, hit in enumerate(results, 1):
        e = hit["entity"]
        print(f"\n{'='*60}")
        print(f"#{i} {e['name']} ⭐ {e['stars']}")
        print(f"URL: {e['url']}")
        print(f"Language: {e['language']} | Topics: {e['topics']}")
        print(f"Score: {hit['distance']:.3f}")
        if e["description"]:
            print(f"\n{e['description']}")
```

---

## Run Example

```python
# Fetch and index
repos = fetch_trending_repos(["python", "javascript", "go"], max_per_lang=100)
index_repos(repos)

# Search examples
print_results(search_repos("machine learning framework"))
print_results(search_repos("web scraping library"))
print_results(search_repos("database ORM", language="python"))
print_results(search_repos("real-time chat", min_stars=1000))
```

---

## Advanced: Fetch from Awesome Lists

```python
import requests
import re

def fetch_awesome_list(url: str) -> list[str]:
    """Parse an Awesome list and extract repository URLs."""
    resp = requests.get(url)
    content = resp.text

    # Find GitHub repo links
    pattern = r'https://github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)'
    matches = re.findall(pattern, content)

    return list(set(matches))

def fetch_repos_from_awesome(awesome_url: str, max_repos: int = 100) -> list[dict]:
    """Fetch repositories from an Awesome list."""
    repo_names = fetch_awesome_list(awesome_url)[:max_repos]

    repos = []
    for name in repo_names:
        try:
            repo = g.get_repo(name)
            readme_content = ""
            try:
                readme = repo.get_readme()
                readme_content = readme.decoded_content.decode("utf-8")[:5000]
            except:
                pass

            repos.append({
                "name": repo.full_name,
                "description": repo.description or "",
                "url": repo.html_url,
                "stars": repo.stargazers_count,
                "language": repo.language or "",
                "topics": repo.get_topics(),
                "readme": readme_content
            })
            time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching {name}: {e}")

    return repos

# Example: Fetch from awesome-python
repos = fetch_repos_from_awesome(
    "https://raw.githubusercontent.com/vinta/awesome-python/master/README.md",
    max_repos=100
)
```
