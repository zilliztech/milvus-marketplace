---
name: clustering
description: "Use when user needs to group similar items together. Triggers on: clustering, group similar, topic modeling, user segmentation, categorization, automatic classification, unsupervised grouping."
---

# Clustering

Automatically group similar content into clusters using vector embeddings — discover hidden patterns and categories in your data.

## When to Activate

Activate this skill when:
- User wants to **group similar items** automatically
- User mentions "clustering", "segmentation", "topic modeling"
- User needs to **discover categories** in unlabeled data
- User wants to **organize content** without predefined labels

**Do NOT activate** when:
- User needs to find duplicates → use `duplicate-detection`
- User has predefined categories → use `filtered-search`
- User needs recommendations → use `rec-system`

## Interactive Flow

### Step 1: Understand Clustering Goal

"What do you want to achieve with clustering?"

A) **Topic discovery** (documents, articles)
   - Find themes in text corpus
   - Group by subject matter

B) **User segmentation** (behavioral data)
   - Group users by behavior
   - Marketing personas

C) **Anomaly detection**
   - Find outliers
   - Fraud detection

D) **Content organization**
   - Auto-categorization
   - Product grouping

Which describes your goal? (A/B/C/D)

### Step 2: Determine Number of Clusters

"Do you know how many clusters you want?"

| If You Know | Algorithm | Configuration |
|-------------|-----------|---------------|
| **Yes, exactly N** | KMeans | `n_clusters=N` |
| **Roughly N** | KMeans + silhouette | Find best K around N |
| **No idea** | DBSCAN/HDBSCAN | Auto-discovers |

### Step 3: Confirm Configuration

"Based on your requirements:

- **Algorithm**: KMeans (you specified 5 clusters)
- **Embedding**: BGE-large
- **Metric**: COSINE similarity

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Sorting a Library

Think of clustering as **a librarian organizing books without labels**:
- Look at each book's content
- Group similar topics together
- Name each section after grouping

```
┌─────────────────────────────────────────────────────────┐
│                    Clustering Pipeline                   │
│                                                          │
│  Unlabeled Documents                                     │
│  ┌─────┬─────┬─────┬─────┬─────┐                       │
│  │Doc1 │Doc2 │Doc3 │Doc4 │Doc5 │ ...                   │
│  └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘                       │
│     │     │     │     │     │                           │
│     ▼     ▼     ▼     ▼     ▼                           │
│  ┌─────────────────────────────┐                        │
│  │    Embedding Model (BGE)     │                        │
│  │    Text → Vector             │                        │
│  └──────────────┬──────────────┘                        │
│                 │                                        │
│                 ▼                                        │
│  [vec1] [vec2] [vec3] [vec4] [vec5] ...                 │
│                 │                                        │
│                 ▼                                        │
│  ┌─────────────────────────────┐                        │
│  │   Clustering Algorithm       │                        │
│  │   (KMeans / DBSCAN)         │                        │
│  └──────────────┬──────────────┘                        │
│                 │                                        │
│     ┌───────────┼───────────┐                           │
│     │           │           │                           │
│     ▼           ▼           ▼                           │
│  ┌─────┐    ┌─────┐    ┌─────┐                         │
│  │ C1  │    │ C2  │    │ C3  │  (Clusters)             │
│  │Tech │    │Sport│    │Food │  (Named by LLM)         │
│  └─────┘    └─────┘    └─────┘                         │
└─────────────────────────────────────────────────────────┘
```

### KMeans vs DBSCAN

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **KMeans** | Fast, predictable clusters | Must specify K | Known cluster count |
| **DBSCAN** | Auto-discovers K, finds outliers | Sensitive to eps | Unknown clusters |
| **HDBSCAN** | More robust than DBSCAN | Slower | Large datasets |

## Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

class VectorClustering:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.collection_name = "clustering"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("cluster_id", DataType.INT32)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="cluster_id", index_type="STL_SORT")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add_data(self, items: list):
        """Add data (unclustered)
        items: [{"id": "...", "content": "..."}]
        """
        contents = [item["content"] for item in items]
        embeddings = self.model.encode(contents).tolist()

        data = [{"id": item["id"], "content": item["content"],
                 "cluster_id": -1, "embedding": emb}
                for item, emb in zip(items, embeddings)]

        self.client.insert(collection_name=self.collection_name, data=data)

    def cluster_kmeans(self, n_clusters: int = 10) -> dict:
        """KMeans clustering - use when you know the number of clusters"""
        all_data = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["id", "content", "embedding"],
            limit=100000
        )

        if len(all_data) < n_clusters:
            raise ValueError(f"Data count {len(all_data)} < cluster count {n_clusters}")

        ids = [item["id"] for item in all_data]
        embeddings = np.array([item["embedding"] for item in all_data])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Update cluster labels in Milvus
        for item_id, label in zip(ids, labels):
            self.client.upsert(
                collection_name=self.collection_name,
                data=[{"id": item_id, "cluster_id": int(label)}]
            )

        # Organize results
        clusters = {}
        for item, label in zip(all_data, labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({"id": item["id"], "content": item["content"]})

        return {
            "n_clusters": n_clusters,
            "clusters": clusters,
            "cluster_sizes": {k: len(v) for k, v in clusters.items()}
        }

    def cluster_dbscan(self, eps: float = 0.3, min_samples: int = 5) -> dict:
        """DBSCAN clustering - use when cluster count is unknown"""
        all_data = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["id", "content", "embedding"],
            limit=100000
        )

        embeddings = np.array([item["embedding"] for item in all_data])

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(embeddings)

        # Update labels
        for item, label in zip(all_data, labels):
            self.client.upsert(
                collection_name=self.collection_name,
                data=[{"id": item["id"], "cluster_id": int(label)}]
            )

        # Organize results
        clusters = {}
        noise_count = 0
        for item, label in zip(all_data, labels):
            label = int(label)
            if label == -1:
                noise_count += 1
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({"id": item["id"], "content": item["content"]})

        return {
            "n_clusters": len(clusters),
            "clusters": clusters,
            "cluster_sizes": {k: len(v) for k, v in clusters.items()},
            "noise_count": noise_count  # Outliers
        }

    def find_optimal_k(self, min_k: int = 2, max_k: int = 20) -> int:
        """Find optimal K using silhouette score"""
        from sklearn.metrics import silhouette_score

        all_data = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["embedding"],
            limit=100000
        )
        embeddings = np.array([item["embedding"] for item in all_data])

        scores = []
        for k in range(min_k, min(max_k + 1, len(embeddings))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            scores.append((k, score))

        best_k = max(scores, key=lambda x: x[1])[0]
        return best_k

    def assign_cluster(self, content: str) -> dict:
        """Assign new content to existing cluster"""
        embedding = self.model.encode(content).tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=5,
            output_fields=["cluster_id"]
        )

        # Vote for cluster
        cluster_votes = {}
        for hit in results[0]:
            cid = hit["entity"]["cluster_id"]
            if cid != -1:  # Ignore noise
                cluster_votes[cid] = cluster_votes.get(cid, 0) + 1

        if not cluster_votes:
            return {"cluster_id": -1, "confidence": 0}

        best_cluster = max(cluster_votes, key=cluster_votes.get)
        return {
            "cluster_id": best_cluster,
            "confidence": cluster_votes[best_cluster] / len(results[0])
        }

# Usage
clustering = VectorClustering()

clustering.add_data([
    {"id": "1", "content": "Python is great for data science"},
    {"id": "2", "content": "Machine learning needs lots of data"},
    {"id": "3", "content": "The weather is nice today"},
    {"id": "4", "content": "Deep learning revolutionized AI"},
    {"id": "5", "content": "Going hiking on weekends is relaxing"},
])

# Find optimal K
best_k = clustering.find_optimal_k(min_k=2, max_k=5)
print(f"Optimal K: {best_k}")

# Cluster
result = clustering.cluster_kmeans(n_clusters=best_k)
for cid, items in result['clusters'].items():
    print(f"\nCluster {cid} ({len(items)} items):")
    for item in items[:3]:
        print(f"  - {item['content'][:50]}...")
```

## Parameter Tuning

### KMeans: Choosing K

```python
# Elbow method visualization
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    scores.append(silhouette_score(embeddings, labels))

# Pick K with highest silhouette score
best_k = scores.index(max(scores)) + 2
```

### DBSCAN: Tuning eps

| eps Value | Effect |
|-----------|--------|
| Too small | Too many tiny clusters |
| Too large | Everything in one cluster |
| Just right | Meaningful groups + outliers |

```python
# Start with distance analysis
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(embeddings)
distances, _ = neighbors.kneighbors(embeddings)
# Plot distances and look for "elbow" to set eps
```

## Common Pitfalls

### ❌ Pitfall 1: Wrong K

**Problem**: Clusters don't make sense

**Why**: Arbitrary K choice

**Fix**: Use silhouette score or domain knowledge

### ❌ Pitfall 2: DBSCAN eps Too Sensitive

**Problem**: Small eps change dramatically changes results

**Why**: Density-based algorithm, data-dependent

**Fix**: Try HDBSCAN (more robust) or normalize embeddings

### ❌ Pitfall 3: Ignoring Outliers

**Problem**: Forcing outliers into clusters degrades quality

**Why**: Not all data belongs to a cluster

**Fix**: Use DBSCAN to identify noise (label=-1)

### ❌ Pitfall 4: Clusters Without Names

**Problem**: Cluster IDs meaningless to users

**Fix**: Use LLM to name clusters based on samples
```python
def name_cluster(samples):
    prompt = f"Name this group based on samples: {samples}"
    return llm.generate(prompt)
```

## When to Level Up

| Need | Upgrade To |
|------|------------|
| Find duplicates | `duplicate-detection` |
| Hierarchical clusters | Use HDBSCAN |
| Real-time clustering | Add incremental clustering |
| Large scale | Add `core:ray` for distributed |

## References

- Topic modeling: `verticals/topic.md`
- User segmentation: `verticals/user-segmentation.md`
- Anomaly detection: `verticals/anomaly.md`
