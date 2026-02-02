---
name: indexing
description: "Use when user needs to create collections, indexes in Milvus. Triggers on: indexing, collection, create index, HNSW, IVF, schema, Milvus collection, vector storage."
---

# Indexing - Index Management

Create Collections and indexes in Milvus.

## Index Type Selection

| Data Scale | Recommended Index | Features |
|------------|-------------------|----------|
| <1M | HNSW | High precision, high memory |
| 1M-10M | IVF_FLAT | Balanced |
| >10M | IVF_PQ | Compressed storage |
| Need exact | FLAT | Brute force search |

## Code Examples

### Create Collection

```python
from pymilvus import MilvusClient, DataType

# Connect (use local file or remote server)
client = MilvusClient(uri="./milvus.db")  # Local file
# client = MilvusClient(uri="http://localhost:19530")  # Docker/Server

# Create schema
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)

# Prepare index params
index_params = client.prepare_index_params()
index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 256})

# Create collection with schema and index
client.create_collection(collection_name="my_collection", schema=schema, index_params=index_params)
```

### Index Types

**HNSW (Recommended, <1M data)**
```python
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="HNSW",
    metric_type="COSINE",  # or L2, IP
    params={"M": 16, "efConstruction": 256}
)
```

**IVF_FLAT (1M-10M)**
```python
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="IVF_FLAT",
    metric_type="L2",
    params={"nlist": 1024}
)
```

**IVF_PQ (>10M, memory efficient)**
```python
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="IVF_PQ",
    metric_type="L2",
    params={"nlist": 1024, "m": 8, "nbits": 8}
)
```

### Collection with Partitions

```python
# Create partitions
client.create_partition(collection_name="my_collection", partition_name="2024_01")
client.create_partition(collection_name="my_collection", partition_name="2024_02")

# Insert with partition
client.insert(collection_name="my_collection", data=data, partition_name="2024_01")

# Search with partition
client.search(collection_name="my_collection", data=query_embedding, partition_names=["2024_01"], limit=10)
```

### With Scalar Index

```python
# Add scalar index to index_params (speeds up filtering)
index_params = client.prepare_index_params()
index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE")
index_params.add_index(field_name="category", index_type="AUTOINDEX")  # Scalar index
```

## Distance Metric Selection

| Type | Use Case | Notes |
|------|----------|-------|
| COSINE | Text similarity | Normalized vectors |
| L2 | Euclidean distance | General purpose |
| IP | Inner product | Recommendation systems |

## Next Steps

After index creation:
- Import data: Use `core:data-ingestion`
- Start searching: Use corresponding scenario skill
