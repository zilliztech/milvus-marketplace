---
name: multi-vector-search
description: "Use when user needs to search across multiple vector fields. Triggers on: multi-vector, multiple embeddings, multi-field search, title + content, combined vectors."
---

# Multi-Vector Search

Perform joint search across multiple vector fields, fusing semantic information at different granularities.

## Use Cases

- Product search: title vector + description vector + review vector
- Paper search: title vector + abstract vector + body vector
- Resume matching: skills vector + experience vector + education vector
- News search: title vector + body vector

## Architecture

```
Query ─┬─→ Search title vector    ─┬─→ Score Fusion → Results
       ├─→ Search description vector ─┤
       └─→ Search review vector    ─┘
```

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer

class MultiVectorSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        self.dim = 1024
        self.collection_name = "multi_vector_search"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("title", DataType.VARCHAR, max_length=1024)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("title_embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("content_embedding", DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="title_embedding", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 256})
        index_params.add_index(field_name="content_embedding", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 256})

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add(self, items: list):
        """Add data
        items: [{"title": "...", "content": "..."}]
        """
        titles = [item["title"] for item in items]
        contents = [item["content"] for item in items]

        title_embeddings = self.model.encode(titles).tolist()
        content_embeddings = self.model.encode(contents).tolist()

        data = []
        for item, title_emb, content_emb in zip(items, title_embeddings, content_embeddings):
            data.append({
                "title": item["title"],
                "content": item["content"],
                "title_embedding": title_emb,
                "content_embedding": content_emb
            })

        self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query: str, limit: int = 10, weights: tuple = (0.6, 0.4)):
        """Multi-vector search
        weights: (title weight, content weight)
        """
        query_embedding = self.model.encode(query).tolist()

        # Title vector search
        title_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="title_embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit * 2
        )

        # Content vector search
        content_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="content_embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit * 2
        )

        # RRF fusion
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[title_req, content_req],
            ranker=RRFRanker(k=60),
            limit=limit,
            output_fields=["title", "content"]
        )

        return [{"title": hit["entity"]["title"],
                 "content": hit["entity"]["content"][:200] + "...",
                 "score": hit["distance"]} for hit in results[0]]

    def search_title_only(self, query: str, limit: int = 10):
        """Search title only"""
        embedding = self.model.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            anns_field="title_embedding",
            limit=limit,
            output_fields=["title", "content"]
        )
        return [{"title": hit["entity"]["title"], "score": hit["distance"]} for hit in results[0]]

# Usage
search = MultiVectorSearch()

# Add articles
search.add([
    {
        "title": "Milvus Vector Database Getting Started Guide",
        "content": "Milvus is an open-source vector database designed for AI applications..."
    },
    {
        "title": "RAG System Architecture Design",
        "content": "Retrieval-Augmented Generation (RAG) is a technique combining retrieval and generation..."
    }
])

# Multi-vector search (title + content)
results = search.search("how to use vector database")

# Title-only search (faster)
results = search.search_title_only("Milvus")
```

## Vector Field Design Recommendations

| Scenario | Vector Fields | Suggested Weights |
|----------|---------------|-------------------|
| Products | title, description, reviews | 0.4, 0.4, 0.2 |
| Papers | title, abstract, body | 0.3, 0.4, 0.3 |
| Resumes | skills, experience, education | 0.4, 0.4, 0.2 |
| News | title, body | 0.5, 0.5 |

## Fusion Strategies

```python
# RRF (recommended, doesn't require score normalization)
ranker = RRFRanker(k=60)

# Weighted fusion (requires score normalization)
from pymilvus import WeightedRanker
ranker = WeightedRanker(0.6, 0.4)  # Title 60%, content 40%
```

## Performance Optimization

```python
# 1. On-demand search (not all vectors every time)
def smart_search(self, query: str, mode: str = "full"):
    if mode == "quick":
        return self.search_title_only(query)  # Title only, faster
    else:
        return self.search(query)  # Full search

# 2. Short queries use title, long queries use content
def adaptive_search(self, query: str):
    if len(query) < 10:
        return self.search(query, weights=(0.8, 0.2))  # Favor title
    else:
        return self.search(query, weights=(0.3, 0.7))  # Favor content
```

## Vertical Applications

See `verticals/` directory for detailed guides:
- `ecommerce.md` - Products (title + description + reviews)
- `academic.md` - Papers (title + abstract + body)
- `recruitment.md` - Resumes (skills + experience)

## Related Tools

- Vectorization: `core:embedding`
- Indexing: `core:indexing`
- Reranking: `core:rerank`
