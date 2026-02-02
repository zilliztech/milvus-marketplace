---
name: clustering
description: "Use when user needs to group similar items together. Triggers on: clustering, group similar, topic modeling, user segmentation, categorization, automatic classification."
---

# Clustering

Automatically group similar content into clusters.

## Use Cases

- Topic/theme clustering
- User segmentation
- Review sentiment clustering
- Anomaly pattern discovery
- Automatic category labeling

## Architecture

```
Data → Vectorize → Clustering algorithm → Cluster labels → Analysis/Naming
```

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

class VectorClustering:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
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
        index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 256})
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
        """KMeans clustering"""
        # Get all data
        all_data = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["id", "content", "embedding"],
            limit=100000
        )

        if len(all_data) < n_clusters:
            raise ValueError(f"Data count {len(all_data)} is less than cluster count {n_clusters}")

        # Extract embeddings
        ids = [item["id"] for item in all_data]
        embeddings = np.array([item["embedding"] for item in all_data])

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Update cluster labels
        for item_id, label in zip(ids, labels):
            self.client.upsert(
                collection_name=self.collection_name,
                data=[{"id": item_id, "cluster_id": int(label)}]
            )

        # Return clustering results
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
        """DBSCAN clustering (auto-discovers cluster count)"""
        all_data = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["id", "content", "embedding"],
            limit=100000
        )

        embeddings = np.array([item["embedding"] for item in all_data])

        # DBSCAN (density-based)
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
            "noise_count": noise_count
        }

    def get_cluster(self, cluster_id: int, limit: int = 100) -> list:
        """Get content of a cluster"""
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'cluster_id == {cluster_id}',
            output_fields=["id", "content"],
            limit=limit
        )
        return results

    def name_cluster(self, cluster_id: int, llm_client) -> str:
        """Name cluster using LLM"""
        samples = self.get_cluster(cluster_id, limit=10)
        sample_texts = "\n".join([s["content"][:200] for s in samples])

        response = llm_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""The following are content samples from the same category. Please describe this category's topic with a short label (2-5 words):

{sample_texts}

Category label:"""
            }],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def find_similar_cluster(self, content: str) -> dict:
        """Find cluster for new content"""
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
            cluster_votes[cid] = cluster_votes.get(cid, 0) + 1

        best_cluster = max(cluster_votes, key=cluster_votes.get)
        return {
            "cluster_id": best_cluster,
            "confidence": cluster_votes[best_cluster] / len(results[0])
        }

# Usage
clustering = VectorClustering()

# Add data
clustering.add_data([
    {"id": "1", "content": "Python is a great programming language"},
    {"id": "2", "content": "Java is popular in enterprise development"},
    {"id": "3", "content": "Machine learning requires lots of data"},
    {"id": "4", "content": "Deep learning is the core technology of AI"},
    {"id": "5", "content": "The weather is nice today, good for an outing"},
    {"id": "6", "content": "Going hiking on the weekend is a good choice"},
])

# KMeans clustering
result = clustering.cluster_kmeans(n_clusters=3)
print(f"Cluster count: {result['n_clusters']}")
for cid, items in result['clusters'].items():
    print(f"\nCluster {cid} ({len(items)} items):")
    for item in items[:3]:
        print(f"  - {item['content'][:50]}...")

# Classify new content
new_cluster = clustering.find_similar_cluster("Go language has great concurrency performance")
print(f"Assigned cluster: {new_cluster['cluster_id']}, Confidence: {new_cluster['confidence']:.2f}")
```

## Algorithm Selection

| Algorithm | Characteristics | Use Case |
|-----------|-----------------|----------|
| **KMeans** | Need to specify k, fast | Known category count |
| **DBSCAN** | Auto-discovers k, finds outliers | Unknown category count |
| **HDBSCAN** | Improved DBSCAN, more stable | Large-scale data |

## Parameter Tuning

```python
# KMeans: Use elbow method to choose k
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(embeddings)
    scores.append(silhouette_score(embeddings, labels))
best_k = scores.index(max(scores)) + 2

# DBSCAN: Adjust eps
# eps too small → too many clusters
# eps too large → too few clusters
```

## Vertical Applications

See `verticals/` directory for detailed guides:
- `topic.md` - Topic clustering
- `user-segmentation.md` - User segmentation
- `anomaly.md` - Anomaly detection

## Related Tools

- Vectorization: `core:embedding`
- Duplicate detection: `scenarios:duplicate-detection`
- Indexing: `core:indexing`
