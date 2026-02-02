---
name: hybrid-search
description: "Use when user needs both keyword and semantic search combined. Triggers on: hybrid search, keyword + semantic, BM25, full-text search, combined search, lexical search."
---

# Hybrid Search

Combine vector semantic search and BM25 keyword search for improved recall quality.

## Use Cases

- E-commerce search (brand name + semantics)
- Legal retrieval (clause number + semantics)
- Academic search (author/year + semantics)
- When users want both exact keyword matching and semantic understanding

## Architecture

```
Query ─┬─→ Vector Search (semantic) ─┬─→ Score Fusion → Results
       └─→ BM25 Search (keyword) ────┘
```

## Complete Implementation

```python
from pymilvus import (
    MilvusClient, DataType, Function, FunctionType,
    AnnSearchRequest, RRFRanker
)
from sentence_transformers import SentenceTransformer

class HybridSearch:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        self.dim = 1024
        self.collection_name = "hybrid_search"
        self._init_collection()

    def _init_collection(self):
        # Check if already exists
        if self.client.has_collection(self.collection_name):
            return

        # Create schema
        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)  # Enable tokenization
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)  # BM25 sparse vector
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)  # Dense vector

        # BM25 function
        bm25_fn = Function(
            name="bm25",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25
        )
        schema.add_function(bm25_fn)

        # Create index
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add(self, texts: list):
        """Add documents"""
        dense_embeddings = self.model.encode(texts).tolist()
        data = [{"text": t, "dense": e} for t, e in zip(texts, dense_embeddings)]
        self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query: str, limit: int = 10, weights: tuple = (0.5, 0.5)):
        """Hybrid search"""
        dense_embedding = self.model.encode(query).tolist()

        # Vector search request
        dense_req = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit * 2
        )

        # BM25 search request
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=limit * 2
        )

        # RRF fusion
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),  # RRF parameter
            limit=limit,
            output_fields=["text"]
        )

        return [{"text": hit["entity"]["text"], "score": hit["distance"]} for hit in results[0]]

# Usage
search = HybridSearch()
search.add([
    "iPhone 15 Pro Max 256GB Deep Purple",
    "Latest Apple smartphone in Space Black",
    "Huawei Mate 60 Pro with Kunlun Glass",
])

# "iPhone phone" matches both keyword "iPhone" and semantic "phone"
results = search.search("iPhone phone")
```

## Score Fusion Strategies

| Strategy | Characteristics | Use Case |
|----------|----------------|----------|
| **RRF** (Reciprocal Rank Fusion) | Rank-based fusion, score-independent | General purpose, recommended |
| **Weighted** | Weighted sum | Requires tuning |

```python
# RRF fusion (recommended)
ranker = RRFRanker(k=60)  # Higher k gives more weight to lower-ranked results

# Weighted fusion
from pymilvus import WeightedRanker
ranker = WeightedRanker(0.7, 0.3)  # Vector 70%, BM25 30%
```

## Model Selection

| Data Type | Vector Model | Tokenizer |
|-----------|--------------|-----------|
| Chinese | BAAI/bge-large-zh-v1.5 | jieba (default) |
| English | text-embedding-3-small | standard |
| Multilingual | BAAI/bge-m3 | multilingual |

## Vertical Applications

See `verticals/` directory for detailed guides:
- `ecommerce.md` - E-commerce search (brand + semantics)
- `legal.md` - Legal retrieval (clause number + semantics)
- `academic.md` - Academic search (author/year + semantics)

## Related Tools

- Vectorization: `core:embedding`
- Indexing: `core:indexing`
- Reranking: `core:rerank`
