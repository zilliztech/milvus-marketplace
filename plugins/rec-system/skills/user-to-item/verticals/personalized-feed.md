# Personalized Feed Recommendation

## Use Cases

- News/content app homepage recommendations
- Short video recommendations
- Social media feed
- Content platform personalized homepage

## Core Architecture

```
User Behavior → User Profile Vector → Recall Candidates → Ranking → Display
                    ↑
              Real-time Update
```

## Recommendation Configuration

| Config | Recommended Value | Description |
|--------|------------------|-------------|
| Content Embedding | `BAAI/bge-large-en-v1.5` | Article/video title |
| User Vector Update | After each action | High real-time requirement |
| Recall Count | 100-500 | For ranking |
| Time Decay | Half-life 24h | News requires timeliness |

## Schema Design

### Content Database

```python
content_schema = client.create_schema()
content_schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
content_schema.add_field("title", DataType.VARCHAR, max_length=512)
content_schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

# Content properties
content_schema.add_field("content_type", DataType.VARCHAR, max_length=32)  # article/video/post
content_schema.add_field("category", DataType.VARCHAR, max_length=64)
content_schema.add_field("tags", DataType.VARCHAR, max_length=256)
content_schema.add_field("author_id", DataType.VARCHAR, max_length=64)
content_schema.add_field("publish_time", DataType.INT64)

# Quality metrics
content_schema.add_field("click_rate", DataType.FLOAT)
content_schema.add_field("like_rate", DataType.FLOAT)
content_schema.add_field("share_rate", DataType.FLOAT)
content_schema.add_field("avg_read_time", DataType.FLOAT)
```

### User Profile

```python
user_schema = client.create_schema()
user_schema.add_field("user_id", DataType.VARCHAR, is_primary=True, max_length=64)
user_schema.add_field("interest_vector", DataType.FLOAT_VECTOR, dim=1024)

# Interest tags
user_schema.add_field("interest_categories", DataType.VARCHAR, max_length=512)
user_schema.add_field("interest_tags", DataType.VARCHAR, max_length=1024)

# Behavior statistics
user_schema.add_field("total_reads", DataType.INT32)
user_schema.add_field("avg_read_time", DataType.FLOAT)
user_schema.add_field("last_active", DataType.INT64)
```

## Implementation

```python
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import numpy as np
import time

class PersonalizedFeed:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    def update_user_interest(self, user_id: str, content_id: str,
                             action: str, duration: float = 0):
        """Update user interest vector"""
        # Action weights
        action_weights = {
            "view": 0.1,
            "click": 0.3,
            "like": 0.5,
            "share": 0.8,
            "collect": 0.7,
            "comment": 0.6
        }
        weight = action_weights.get(action, 0.1)

        # Read duration weighting
        if duration > 0:
            # Extra weight for reading over 30 seconds
            weight *= min(2.0, 1 + duration / 60)

        # Get content vector
        content = self.client.get(
            collection_name="contents",
            ids=[content_id],
            output_fields=["embedding"]
        )
        if not content:
            return

        content_vector = np.array(content[0]["embedding"])

        # Get user's current vector
        user = self.client.get(
            collection_name="users",
            ids=[user_id],
            output_fields=["interest_vector", "total_reads"]
        )

        if user:
            # Update existing vector (moving average)
            old_vector = np.array(user[0]["interest_vector"])
            total_reads = user[0]["total_reads"]

            # Time decay: old interest weight decreases over time
            decay = 0.95  # Weight of old vector per update

            new_vector = decay * old_vector + weight * content_vector
            new_vector = new_vector / np.linalg.norm(new_vector)  # Normalize

            self.client.upsert(
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
            self.client.insert(
                collection_name="users",
                data=[{
                    "user_id": user_id,
                    "interest_vector": content_vector.tolist(),
                    "interest_categories": "",
                    "interest_tags": "",
                    "total_reads": 1,
                    "avg_read_time": duration,
                    "last_active": int(time.time())
                }]
            )

    def get_recommendations(self, user_id: str, limit: int = 20,
                           exclude_ids: list = None) -> list:
        """Get recommended content"""
        # Get user vector
        user = self.client.get(
            collection_name="users",
            ids=[user_id],
            output_fields=["interest_vector"]
        )

        if not user:
            # New user: return popular content
            return self._get_hot_contents(limit)

        user_vector = user[0]["interest_vector"]

        # Build filter conditions
        current_time = int(time.time())
        one_day_ago = current_time - 24 * 3600

        filter_parts = [f'publish_time >= {one_day_ago}']  # Only recommend last 24 hours

        if exclude_ids:
            # Exclude already viewed
            exclude_expr = ' and '.join([f'id != "{eid}"' for eid in exclude_ids[:100]])
            filter_parts.append(f'({exclude_expr})')

        filter_expr = ' and '.join(filter_parts)

        # Vector recall
        results = self.client.search(
            collection_name="contents",
            data=[user_vector],
            filter=filter_expr,
            limit=limit * 3,  # Recall more for ranking
            output_fields=["title", "category", "author_id", "publish_time",
                          "click_rate", "like_rate"]
        )

        # Ranking: combine relevance + quality + recency
        candidates = results[0]

        for c in candidates:
            relevance = c["distance"]
            quality = c["entity"]["click_rate"] * 0.3 + c["entity"]["like_rate"] * 0.7
            recency = 1 - (current_time - c["entity"]["publish_time"]) / (24 * 3600)
            recency = max(0, recency)

            c["final_score"] = relevance * 0.5 + quality * 0.3 + recency * 0.2

        candidates.sort(key=lambda x: x["final_score"], reverse=True)

        return [{
            "id": c["id"],
            "title": c["entity"]["title"],
            "category": c["entity"]["category"],
            "author_id": c["entity"]["author_id"],
            "score": c["final_score"]
        } for c in candidates[:limit]]

    def _get_hot_contents(self, limit: int) -> list:
        """Get popular content (cold start)"""
        current_time = int(time.time())
        one_day_ago = current_time - 24 * 3600

        results = self.client.query(
            collection_name="contents",
            filter=f'publish_time >= {one_day_ago}',
            output_fields=["title", "category", "click_rate", "like_rate"],
            limit=limit * 2
        )

        # Sort by popularity
        for r in results:
            r["hot_score"] = r["click_rate"] * 0.4 + r["like_rate"] * 0.6

        results.sort(key=lambda x: x["hot_score"], reverse=True)

        return results[:limit]

    def get_similar_contents(self, content_id: str, limit: int = 10) -> list:
        """Related recommendations (also viewed)"""
        content = self.client.get(
            collection_name="contents",
            ids=[content_id],
            output_fields=["embedding", "category"]
        )

        if not content:
            return []

        # Same category + vector similarity
        results = self.client.search(
            collection_name="contents",
            data=[content[0]["embedding"]],
            filter=f'id != "{content_id}" and category == "{content[0]["category"]}"',
            limit=limit,
            output_fields=["title", "category"]
        )

        return results[0]
```

## Examples

```python
feed = PersonalizedFeed()

# Report user behavior
feed.update_user_interest(
    user_id="user_001",
    content_id="article_123",
    action="like",
    duration=120  # Read for 2 minutes
)

# Get recommendations
recommendations = feed.get_recommendations(
    user_id="user_001",
    limit=20,
    exclude_ids=["article_123"]  # Exclude just viewed
)

print("Recommended for you:")
for r in recommendations:
    print(f"  - {r['title']} [{r['category']}]")

# Related recommendations
similar = feed.get_similar_contents("article_123")
print("\nRelated articles:")
for s in similar:
    print(f"  - {s['entity']['title']}")
```

## Optimization Strategies

### 1. Multi-path Recall

```python
def multi_recall(self, user_id: str, limit: int = 100):
    """Multi-path recall fusion"""
    results = []

    # Path 1: Interest vector recall
    interest_results = self.recall_by_interest(user_id, limit // 3)
    results.extend(interest_results)

    # Path 2: New content from followed authors
    follow_results = self.recall_by_following(user_id, limit // 3)
    results.extend(follow_results)

    # Path 3: Popular content
    hot_results = self._get_hot_contents(limit // 3)
    results.extend(hot_results)

    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)

    return unique
```

### 2. Exploration vs Exploitation Balance

```python
def get_recommendations_with_exploration(self, user_id: str, limit: int = 20,
                                         explore_ratio: float = 0.2):
    """Recommendations with exploration"""
    # Main recommendations (exploitation)
    main_count = int(limit * (1 - explore_ratio))
    main_results = self.get_recommendations(user_id, main_count)

    # Exploration: popular from random categories
    explore_count = limit - main_count
    explore_results = self._get_random_category_hot(explore_count)

    # Mix
    import random
    results = main_results + explore_results
    random.shuffle(results)

    return results
```
