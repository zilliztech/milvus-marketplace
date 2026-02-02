---
name: local-setup
description: "Use when user needs to set up Milvus locally. Triggers on: local setup, install milvus, docker, docker-compose, dev environment, milvus standalone, milvus lite."
---

# Local Setup - Local Deployment

Set up a local Milvus development environment.

## Method Selection

| Method | Use Case | Resource Requirements |
|--------|----------|----------------------|
| Milvus Lite | Quick testing, learning | Minimal |
| Docker Standalone | Development, small scale | Medium |
| Docker Compose | Full features | Higher |

## Milvus Lite (Simplest)

No installation needed, use directly:

```python
from pymilvus import MilvusClient

# In-memory mode
client = MilvusClient()

# Or persist to file
client = MilvusClient("./milvus.db")

# Create collection
client.create_collection(
    collection_name="demo",
    dimension=768
)

# Insert
client.insert(
    collection_name="demo",
    data=[{"id": 1, "vector": [0.1]*768, "text": "hello"}]
)

# Search
results = client.search(
    collection_name="demo",
    data=[[0.1]*768],
    limit=3
)
```

## Docker Standalone

```bash
# Download docker-compose
wget https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh

# Start
bash standalone_embed.sh start

# Stop
bash standalone_embed.sh stop
```

Or manually:

```bash
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus:/var/lib/milvus \
  milvusdb/milvus:latest \
  milvus run standalone
```

Connect:

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
```

## Docker Compose (Full)

```yaml
# docker-compose.yml
version: '3.5'

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data

  milvus:
    image: milvusdb/milvus:latest
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
```

```bash
docker-compose up -d
```

## Verify Installation

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
print(f"Milvus version: {client.get_server_version()}")
print("Connected successfully!")
```

## Common Issues

**Port in use**
```bash
# Check
lsof -i :19530

# Change port
docker run -p 29530:19530 ...
```

**Out of memory**
```bash
# Limit memory
docker run -m 4g ...
```

**Data persistence**
```bash
# Mount volume
docker run -v /path/to/data:/var/lib/milvus ...
```

## Next Steps

After environment is ready:
- Requirements analysis: Use `core:pilot`
- Direct implementation: Use corresponding scenario skill
