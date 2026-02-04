# Personalized Feed Recommendation

> Build a personalized content feed based on user interests.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Content Language

<ask_user>
What language is your content in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** | English models |
| **Chinese** | Chinese models |
| **Multilingual** | Multilingual models |
</ask_user>

### 2. Content Embedding

<ask_user>
Choose content embedding:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download |

Local options:
- `BAAI/bge-large-en-v1.5` (1024d, English)
- `BAAI/bge-large-zh-v1.5` (1024d, Chinese)
</ask_user>

### 3. Data Scale

<ask_user>
How many content items and users?

- Each content = 1 vector
- Each user = 1 interest vector

| Total Vectors | Recommended Milvus |
|---------------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 4. Project Setup

<ask_user>
| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production |
| **pip** | Quick prototype |
</ask_user>

---

## Dependencies

```bash
uv init personalized-feed
cd personalized-feed
uv add pymilvus sentence-transformers numpy
```

---

## End-to-End Implementation

### Step 1: Configure Models

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import time

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
DIMENSION = 1024

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()
```

### Step 2: Create Collections

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("feed.db")

# Content collection
content_schema = milvus.create_schema(auto_id=True)
content_schema.add_field("id", DataType.INT64, is_primary=True)
content_schema.add_field("content_id", DataType.VARCHAR, max_length=64)
content_schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
content_schema.add_field("title", DataType.VARCHAR, max_length=512)
content_schema.add_field("content_type", DataType.VARCHAR, max_length=32)
content_schema.add_field("category", DataType.VARCHAR, max_length=64)
content_schema.add_field("author_id", DataType.VARCHAR, max_length=64)
content_schema.add_field("publish_time", DataType.INT64)
content_schema.add_field("click_rate", DataType.FLOAT)
content_schema.add_field("like_rate", DataType.FLOAT)

content_index = milvus.prepare_index_params()
content_index.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
milvus.create_collection("contents", schema=content_schema, index_params=content_index)

# User collection
user_schema = milvus.create_schema()
user_schema.add_field("user_id", DataType.VARCHAR, is_primary=True, max_length=64)
user_schema.add_field("interest_vector", DataType.FLOAT_VECTOR, dim=DIMENSION)
user_schema.add_field("total_reads", DataType.INT32)
user_schema.add_field("last_active", DataType.INT64)

user_index = milvus.prepare_index_params()
user_index.add_index("interest_vector", index_type="AUTOINDEX", metric_type="COSINE")
milvus.create_collection("users", schema=user_schema, index_params=user_index)
```

### Step 3: Update User Interests

```python
def update_user_interest(user_id: str, content_id: str, action: str, duration: float = 0):
    """Update user interest based on behavior."""
    # Action weights
    weights = {"view": 0.1, "click": 0.3, "like": 0.5, "share": 0.8, "collect": 0.7}
    weight = weights.get(action, 0.1)

    # Duration bonus
    if duration > 0:
        weight *= min(2.0, 1 + duration / 60)

    # Get content vector
    content = milvus.query(
        collection_name="contents",
        filter=f'content_id == "{content_id}"',
        output_fields=["embedding"],
        limit=1
    )
    if not content:
        return

    content_vector = np.array(content[0]["embedding"])

    # Get or create user vector
    user = milvus.query(
        collection_name="users",
        filter=f'user_id == "{user_id}"',
        output_fields=["interest_vector", "total_reads"],
        limit=1
    )

    if user:
        old_vector = np.array(user[0]["interest_vector"])
        total_reads = user[0]["total_reads"]

        # Moving average with decay
        decay = 0.95
        new_vector = decay * old_vector + weight * content_vector
        new_vector = new_vector / np.linalg.norm(new_vector)

        milvus.upsert(
            collection_name="users",
            data=[{
                "user_id": user_id,
                "interest_vector": new_vector.tolist(),
                "total_reads": total_reads + 1,
                "last_active": int(time.time())
            }]
        )
    else:
        # New user
        milvus.insert(
            collection_name="users",
            data=[{
                "user_id": user_id,
                "interest_vector": content_vector.tolist(),
                "total_reads": 1,
                "last_active": int(time.time())
            }]
        )
```

### Step 4: Get Recommendations

```python
def get_recommendations(user_id: str, top_k: int = 20, exclude_ids: list = None):
    """Get personalized recommendations."""
    user = milvus.query(
        collection_name="users",
        filter=f'user_id == "{user_id}"',
        output_fields=["interest_vector"],
        limit=1
    )

    if not user:
        return get_popular_content(top_k)

    user_vector = user[0]["interest_vector"]

    # Filters
    current_time = int(time.time())
    one_day_ago = current_time - 24 * 3600

    filters = [f'publish_time >= {one_day_ago}']

    if exclude_ids:
        exclude_expr = ' and '.join([f'content_id != "{eid}"' for eid in exclude_ids[:50]])
        filters.append(f'({exclude_expr})')

    filter_expr = ' and '.join(filters)

    results = milvus.search(
        collection_name="contents",
        data=[user_vector],
        filter=filter_expr,
        limit=top_k * 3,
        output_fields=["content_id", "title", "category", "click_rate", "like_rate", "publish_time"]
    )

    # Rank by relevance + quality + recency
    candidates = results[0]
    for c in candidates:
        relevance = c["distance"]
        quality = c["entity"]["click_rate"] * 0.3 + c["entity"]["like_rate"] * 0.7
        recency = 1 - (current_time - c["entity"]["publish_time"]) / (24 * 3600)
        recency = max(0, recency)

        c["final_score"] = relevance * 0.5 + quality * 0.3 + recency * 0.2

    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    return [{
        "content_id": c["entity"]["content_id"],
        "title": c["entity"]["title"],
        "category": c["entity"]["category"],
        "score": c["final_score"]
    } for c in candidates[:top_k]]

def get_popular_content(top_k: int):
    """Get popular content (cold start)."""
    current_time = int(time.time())
    one_day_ago = current_time - 24 * 3600

    results = milvus.query(
        collection_name="contents",
        filter=f'publish_time >= {one_day_ago}',
        output_fields=["content_id", "title", "category", "click_rate", "like_rate"],
        limit=top_k * 2
    )

    for r in results:
        r["hot_score"] = r["click_rate"] * 0.4 + r["like_rate"] * 0.6

    results.sort(key=lambda x: x["hot_score"], reverse=True)
    return results[:top_k]
```

---

## Run Example

```python
# Index content
milvus.insert(collection_name="contents", data=[{
    "content_id": "article_001",
    "embedding": embed(["Introduction to Machine Learning"])[0],
    "title": "Introduction to Machine Learning",
    "content_type": "article",
    "category": "Technology",
    "author_id": "author_001",
    "publish_time": int(time.time()),
    "click_rate": 0.15,
    "like_rate": 0.08
}])

# Track user behavior
update_user_interest("user_001", "article_001", "like", duration=120)

# Get recommendations
recommendations = get_recommendations("user_001", top_k=20, exclude_ids=["article_001"])

print("Recommended for you:")
for r in recommendations:
    print(f"  - {r['title']} [{r['category']}]")
```
