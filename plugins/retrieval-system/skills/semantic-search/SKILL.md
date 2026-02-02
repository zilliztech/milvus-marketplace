---
name: semantic-search
description: "Use when user wants to build semantic/text search. Triggers on: semantic search, text search, full-text search, natural language search, find similar text, vector search."
---

# Semantic Search

Build vector-based semantic search systems.

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, collection_name: str = "semantic_search", uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        self.dim = 1024
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add(self, texts: list):
        """Add documents"""
        embeddings = self.model.encode(texts).tolist()
        data = [{"text": text, "embedding": emb} for text, emb in zip(texts, embeddings)]
        self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query: str, limit: int = 10):
        """Search"""
        query_embedding = self.model.encode([query]).tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            data=query_embedding,
            limit=limit,
            output_fields=["text"]
        )

        return [
            {"text": hit["entity"]["text"], "score": hit["distance"]}
            for hit in results[0]
        ]

# Usage
search = SemanticSearch()
search.add(["Python is a programming language", "Java is also a programming language", "Machine learning is popular"])
results = search.search("what is programming")
for r in results:
    print(f"{r['score']:.3f}: {r['text']}")
```

## Search with Filtering

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="./milvus.db")

# Create schema with category field
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=256)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)

# Filter during search
results = client.search(
    collection_name="semantic_search",
    data=query_embedding,
    limit=10,
    filter='category == "tech"',  # Filter condition
    output_fields=["text", "category"]
)
```

## Hybrid Search (Vector + Keyword)

```python
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

client = MilvusClient(uri="./milvus.db")

# Vector search request
vector_req = AnnSearchRequest(
    data=query_embedding,
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=20
)

# Keyword search request (requires BM25 enabled)
text_req = AnnSearchRequest(
    data=[query],
    anns_field="text",
    param={"metric_type": "BM25"},
    limit=20
)

# Fusion with hybrid_search
results = client.hybrid_search(
    collection_name="semantic_search",
    reqs=[vector_req, text_req],
    ranker=RRFRanker(k=60),
    limit=10,
    output_fields=["text"]
)
```

## Performance Optimization

```python
# 1. GPU accelerated embedding
model = SentenceTransformer('...', device='cuda')

# 2. Batch search
queries = ["query1", "query2", "query3"]
embeddings = model.encode(queries).tolist()
results = client.search(
    collection_name="semantic_search",
    data=embeddings,
    limit=10,
    output_fields=["text"]
)

# 3. Adjust search_params for precision/speed tradeoff
# Higher ef = higher precision, slower speed
results = client.search(
    collection_name="semantic_search",
    data=embeddings,
    search_params={"ef": 128},  # High precision
    limit=10
)
```

## Related Tools

- Chunking: `core:chunking`
- Embedding: `core:embedding`
- Indexing: `core:indexing`
