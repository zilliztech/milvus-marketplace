# Topic & Document Clustering

> Automatically discover and organize topics from document collections.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Document Language

<ask_user>
What language are your documents in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** | English models |
| **Chinese** | Chinese models |
| **Multilingual** | Multilingual models |
</ask_user>

### 2. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download |

Local options:
- `BAAI/bge-large-en-v1.5` (1024d, English)
- `BAAI/bge-large-zh-v1.5` (1024d, Chinese)
</ask_user>

### 3. Clustering Algorithm

<ask_user>
Choose clustering algorithm:

| Algorithm | Best For |
|-----------|----------|
| **KMeans** | Known number of topics |
| **DBSCAN** | Unknown topics, auto-discover |
| **HDBSCAN** | Hierarchical clustering |
</ask_user>

### 4. Data Scale

<ask_user>
How many documents do you have?

| Document Count | Recommended Milvus |
|----------------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 5. Project Setup

<ask_user>
| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production |
| **pip** | Quick prototype |
</ask_user>

---

## Dependencies

```bash
uv init topic-clustering
cd topic-clustering
uv add pymilvus openai scikit-learn numpy
```

---

## End-to-End Implementation

### Step 1: Configure Models

```python
from openai import OpenAI
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

client = OpenAI()

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([e.embedding for e in resp.data])

DIMENSION = 1536
```

### Step 2: Clustering Functions

```python
def cluster_kmeans(embeddings: np.ndarray, n_clusters: int):
    """KMeans clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_

def cluster_dbscan(embeddings: np.ndarray, eps: float = 0.3, min_samples: int = 5):
    """DBSCAN clustering (auto-discover topics)."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = dbscan.fit_predict(embeddings)
    return labels

def estimate_k(embeddings: np.ndarray, max_k: int = 20) -> int:
    """Estimate optimal k using silhouette score."""
    from sklearn.metrics import silhouette_score

    scores = []
    for k in range(2, min(max_k, len(embeddings))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append((k, score))

    return max(scores, key=lambda x: x[1])[0]

def name_cluster(documents: list[str]) -> str:
    """Generate topic name using LLM."""
    sample = documents[:10]

    prompt = f"""Below are related document summaries. Give this topic a short name (2-5 words).

Documents:
{chr(10).join(['- ' + doc[:100] for doc in sample])}

Topic name:"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return resp.choices[0].message.content.strip()
```

### Step 3: Full Clustering Pipeline

```python
def cluster_documents(documents: list[str], method: str = "kmeans", n_clusters: int = None):
    """Cluster documents and analyze topics."""
    # 1. Embed documents
    embeddings = embed(documents)

    # 2. Cluster
    if method == "kmeans":
        if n_clusters is None:
            n_clusters = estimate_k(embeddings)
        labels, centers = cluster_kmeans(embeddings, n_clusters)
    else:
        labels = cluster_dbscan(embeddings)

    # 3. Organize by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:  # Noise (DBSCAN)
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(documents[i])

    # 4. Name each cluster
    results = []
    for label, docs in clusters.items():
        topic_name = name_cluster(docs)
        results.append({
            "cluster_id": int(label),
            "topic_name": topic_name,
            "document_count": len(docs),
            "sample_documents": docs[:5]
        })

    return {
        "total_documents": len(documents),
        "total_clusters": len(clusters),
        "noise_count": sum(1 for l in labels if l == -1),
        "clusters": sorted(results, key=lambda x: x["document_count"], reverse=True)
    }
```

### Step 4: Incremental Clustering

```python
def assign_to_clusters(new_documents: list[str], cluster_centers: np.ndarray, threshold: float = 0.7):
    """Assign new documents to existing clusters or flag as new topic."""
    embeddings = embed(new_documents)

    assignments = []
    new_topic_candidates = []

    for i, emb in enumerate(embeddings):
        similarities = np.dot(cluster_centers, emb) / (
            np.linalg.norm(cluster_centers, axis=1) * np.linalg.norm(emb)
        )
        max_sim = similarities.max()
        best_cluster = int(similarities.argmax())

        if max_sim >= threshold:
            assignments.append({
                "document": new_documents[i],
                "cluster_id": best_cluster,
                "similarity": float(max_sim)
            })
        else:
            new_topic_candidates.append(new_documents[i])

    return {
        "assigned": assignments,
        "new_topic_candidates": new_topic_candidates
    }
```

---

## Run Example

```python
# Sample news articles
news = [
    "Apple releases new iPhone 16 with A18 chip",
    "Samsung Galaxy S25 series officially announced",
    "Google Pixel 9 features Tensor G4 processor",
    "National team loses 0-3 to Japan",
    "World Cup qualifiers results summary",
    "Football coach resignation rumors",
    "OpenAI releases GPT-5",
    "Google Gemini 2.0 launched",
    "LLM competition intensifies",
]

result = cluster_documents(news, method="kmeans", n_clusters=3)

print(f"Found {result['total_clusters']} topics:\n")
for cluster in result["clusters"]:
    print(f"Topic: {cluster['topic_name']} ({cluster['document_count']} articles)")
    for doc in cluster["sample_documents"][:3]:
        print(f"  - {doc[:50]}...")
    print()
```

Output:
```
Found 3 topics:

Topic: Smartphone Releases (3 articles)
  - Apple releases new iPhone 16 with A18 chip...
  - Samsung Galaxy S25 series officially announced...
  - Google Pixel 9 features Tensor G4 processor...

Topic: Football News (3 articles)
  - National team loses 0-3 to Japan...
  - World Cup qualifiers results summary...
  - Football coach resignation rumors...

Topic: AI Models (3 articles)
  - OpenAI releases GPT-5...
  - Google Gemini 2.0 launched...
  - LLM competition intensifies...
```

---

## Visualization (Optional)

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, topic_names: list):
    """Visualize clusters in 2D."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10')

    handles, _ = scatter.legend_elements()
    plt.legend(handles, topic_names, title="Topics")

    plt.title("Document Clustering")
    plt.savefig("clusters.png")
    plt.show()
```
