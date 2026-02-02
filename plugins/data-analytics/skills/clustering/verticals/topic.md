# Topic/Document Clustering

## Use Cases

- News topic discovery
- User feedback classification
- Document auto-archiving
- Trending topic tracking

## Recommended Configuration

| Config | Recommended Value | Description |
|--------|------------------|-------------|
| Embedding | `text-embedding-3-small` | OpenAI embedding |
| Clustering Algorithm | KMeans | Known number of clusters |
| | DBSCAN | Auto-discover clusters |
| | HDBSCAN | Hierarchical clustering |
| Dimensionality Reduction (optional) | UMAP / PCA | Visualization |

## Implementation

```python
from pymilvus import MilvusClient
from openai import OpenAI
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

class TopicClustering:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()

    def embed_documents(self, documents: list) -> np.ndarray:
        """Batch vectorization using OpenAI API"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=documents
        )
        return np.array([item.embedding for item in response.data])

    def cluster_kmeans(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """KMeans clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        return labels, kmeans.cluster_centers_

    def cluster_dbscan(self, embeddings: np.ndarray, eps: float = 0.3,
                       min_samples: int = 5) -> np.ndarray:
        """DBSCAN clustering (auto-discover number of clusters)"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(embeddings)
        return labels

    def name_cluster(self, documents: list) -> str:
        """Name cluster with LLM"""
        sample = documents[:10]  # Take first 10 samples

        prompt = f"""Below is a group of related document summaries. Please name this topic in a short phrase (2-5 words).

Documents:
{chr(10).join(['- ' + doc[:100] for doc in sample])}

Topic name:"""

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    def cluster_and_analyze(self, documents: list, method: str = "kmeans",
                            n_clusters: int = None) -> dict:
        """Cluster and analyze"""
        # 1. Vectorize
        embeddings = self.embed_documents(documents)

        # 2. Cluster
        if method == "kmeans":
            if n_clusters is None:
                # Estimate optimal k with elbow method
                n_clusters = self._estimate_k(embeddings)
            labels, centers = self.cluster_kmeans(embeddings, n_clusters)
        else:
            labels = self.cluster_dbscan(embeddings)

        # 3. Organize results
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # DBSCAN noise points
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(documents[i])

        # 4. Name each cluster
        result = []
        for label, docs in clusters.items():
            topic_name = self.name_cluster(docs)
            result.append({
                "cluster_id": int(label),
                "topic_name": topic_name,
                "document_count": len(docs),
                "sample_documents": docs[:5]
            })

        return {
            "total_documents": len(documents),
            "total_clusters": len(clusters),
            "noise_count": sum(1 for l in labels if l == -1),
            "clusters": sorted(result, key=lambda x: x["document_count"], reverse=True)
        }

    def _estimate_k(self, embeddings: np.ndarray, max_k: int = 20) -> int:
        """Estimate optimal k with elbow method"""
        from sklearn.metrics import silhouette_score

        scores = []
        for k in range(2, min(max_k, len(embeddings))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            scores.append((k, score))

        # Return k with highest silhouette score
        best_k = max(scores, key=lambda x: x[1])[0]
        return best_k

    def incremental_cluster(self, new_documents: list, existing_centers: np.ndarray,
                            threshold: float = 0.7) -> dict:
        """Incremental clustering: assign new documents to existing clusters or create new ones"""
        embeddings = self.embed_documents(new_documents)

        assignments = []
        new_cluster_docs = []

        for i, emb in enumerate(embeddings):
            # Calculate similarity to each center
            similarities = np.dot(existing_centers, emb) / (
                np.linalg.norm(existing_centers, axis=1) * np.linalg.norm(emb)
            )
            max_sim = similarities.max()
            best_cluster = similarities.argmax()

            if max_sim >= threshold:
                assignments.append({
                    "document": new_documents[i],
                    "cluster_id": int(best_cluster),
                    "similarity": float(max_sim)
                })
            else:
                new_cluster_docs.append(new_documents[i])

        return {
            "assigned": assignments,
            "new_cluster_candidates": new_cluster_docs
        }
```

## Example: News Topic Clustering

```python
clustering = TopicClustering()

# News data
news = [
    "Apple releases new iPhone 16 with A18 chip",
    "Samsung Galaxy S25 series officially announced",
    "Google Pixel 9 features Tensor G4 processor",
    "National team loses 0-3 to Japan, playoffs hopes dim",
    "World Cup qualifiers results summary",
    "National football coach dismissal rumors",
    "OpenAI releases GPT-5 with major performance improvements",
    "Google Gemini 2.0 officially launched",
    "LLM competition enters heated phase",
]

result = clustering.cluster_and_analyze(news, method="kmeans", n_clusters=3)

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
  - National team loses 0-3 to Japan, playoffs hopes dim...
  - World Cup qualifiers results summary...
  - National football coach dismissal rumors...

Topic: AI Models (3 articles)
  - OpenAI releases GPT-5 with major performance improvements...
  - Google Gemini 2.0 officially launched...
  - LLM competition enters heated phase...
```

## Visualization

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, topic_names: list):
    """Cluster visualization"""
    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    coords = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10')

    # Add legend
    handles, _ = scatter.legend_elements()
    plt.legend(handles, topic_names, title="Topics")

    plt.title("Document Clustering Results")
    plt.savefig("cluster_visualization.png")
    plt.show()
```

## Real-time Topic Monitoring

```python
class RealtimeTopicMonitor:
    def __init__(self):
        self.clustering = TopicClustering()
        self.current_centers = None
        self.topics = {}

    def process_batch(self, documents: list):
        """Process a batch of new documents"""
        if self.current_centers is None:
            # Initial clustering
            result = self.clustering.cluster_and_analyze(documents)
            self.topics = {c["cluster_id"]: c["topic_name"] for c in result["clusters"]}
            # Calculate cluster centers
            embeddings = self.clustering.embed_documents(documents)
            self.current_centers = self._compute_centers(embeddings, result)
        else:
            # Incremental clustering
            result = self.clustering.incremental_cluster(
                documents, self.current_centers
            )

            # If there are new topic candidates
            if len(result["new_cluster_candidates"]) >= 5:
                print(f"Potential new topic discovered! Contains {len(result['new_cluster_candidates'])} documents")

        return result
```
