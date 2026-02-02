---
name: user-to-item
description: "Use when user needs personalized recommendations based on user profile. Triggers on: personalized, user recommendation, personalized recommendations, for you, feed, user preference."
---

# User-to-Item Personalized Recommendation

Personalized item recommendations based on user profile/behavior vectors.

## Use Cases

- Personalized product recommendations (e-commerce homepage)
- Personalized feeds (news/video/content platforms)
- Job recommendations (recruitment platforms)
- Dating matching

## Architecture

```
User behavior history → Generate user vector → Vector search item library → Personalized recommendation list
```

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
import numpy as np

class UserToItemRecommender:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        self.dim = 1024
        self._init_collections()

    def _init_collections(self):
        # Item library
        if not self.client.has_collection("items"):
            schema = self.client.create_schema()
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("title", DataType.VARCHAR, max_length=1024)
            schema.add_field("category", DataType.VARCHAR, max_length=256)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                                   params={"M": 16, "efConstruction": 256})

            self.client.create_collection(collection_name="items", schema=schema, index_params=index_params)

        # User profile library
        if not self.client.has_collection("users"):
            schema = self.client.create_schema()
            schema.add_field("user_id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
            schema.add_field("updated_at", DataType.INT64)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                                   params={"M": 16, "efConstruction": 256})

            self.client.create_collection(collection_name="users", schema=schema, index_params=index_params)

    def add_items(self, items: list):
        """Add items"""
        texts = [f"{item['title']} {item.get('description', '')}" for item in items]
        embeddings = self.model.encode(texts).tolist()

        data = [{"id": item["id"], "title": item["title"],
                 "category": item.get("category", ""), "embedding": emb}
                for item, emb in zip(items, embeddings)]

        self.client.insert(collection_name="items", data=data)

    def update_user_profile(self, user_id: str, interactions: list, decay: float = 0.9):
        """Update user profile
        interactions: [{"item_id": "...", "action": "view/click/buy", "timestamp": ...}]
        """
        # Get embeddings of interacted items
        item_ids = [i["item_id"] for i in interactions]
        items = self.client.get(collection_name="items", ids=item_ids, output_fields=["embedding"])

        if not items:
            return

        # Weighted average (recent items have higher weight)
        weights = []
        embeddings = []
        for i, inter in enumerate(interactions):
            action_weight = {"buy": 3.0, "click": 1.5, "view": 1.0}.get(inter.get("action", "view"), 1.0)
            time_weight = decay ** i  # i=0 is most recent, highest weight
            weights.append(action_weight * time_weight)

        # Find corresponding embeddings
        item_emb_map = {item["id"]: item["embedding"] for item in items}
        for inter in interactions:
            if inter["item_id"] in item_emb_map:
                embeddings.append(item_emb_map[inter["item_id"]])

        if not embeddings:
            return

        # Weighted average
        weights = weights[:len(embeddings)]
        weights = np.array(weights) / sum(weights)
        user_embedding = np.average(embeddings, axis=0, weights=weights).tolist()

        # Update or insert user profile
        import time
        self.client.upsert(
            collection_name="users",
            data=[{"user_id": user_id, "embedding": user_embedding, "updated_at": int(time.time())}]
        )

    def recommend(self, user_id: str, limit: int = 10, exclude_ids: list = None, category: str = None):
        """Recommend items for user"""
        # Get user profile
        users = self.client.get(collection_name="users", ids=[user_id], output_fields=["embedding"])

        if not users:
            # New user, return popular items
            return self.get_popular_items(limit)

        user_embedding = users[0]["embedding"]

        # Build filter conditions
        filter_parts = []
        if exclude_ids:
            ids_str = ", ".join([f'"{id}"' for id in exclude_ids])
            filter_parts.append(f'id not in [{ids_str}]')
        if category:
            filter_parts.append(f'category == "{category}"')

        filter_expr = " and ".join(filter_parts) if filter_parts else ""

        # Search
        results = self.client.search(
            collection_name="items",
            data=[user_embedding],
            filter=filter_expr,
            limit=limit,
            output_fields=["id", "title", "category"]
        )

        return [{"id": hit["entity"]["id"],
                 "title": hit["entity"]["title"],
                 "category": hit["entity"]["category"],
                 "score": hit["distance"]} for hit in results[0]]

    def get_popular_items(self, limit: int = 10):
        """Get popular items (for cold start)"""
        # Simple implementation: return random items
        # Production should use actual popularity data
        results = self.client.query(
            collection_name="items",
            filter="",
            limit=limit,
            output_fields=["id", "title", "category"]
        )
        return results

# Usage
recommender = UserToItemRecommender()

# Add items
recommender.add_items([
    {"id": "a1", "title": "Python Programming for Beginners", "category": "Programming"},
    {"id": "a2", "title": "Machine Learning in Practice", "category": "AI"},
    {"id": "a3", "title": "Deep Learning Fundamentals", "category": "AI"},
    {"id": "a4", "title": "Advanced JavaScript Techniques", "category": "Programming"},
])

# Record user behavior
recommender.update_user_profile("user001", [
    {"item_id": "a2", "action": "buy"},
    {"item_id": "a3", "action": "click"},
])

# Personalized recommendations
recs = recommender.recommend("user001", limit=5)
# Returns AI-related content first
```

## User Profile Strategies

### 1. Real-time vs Offline

| Strategy | Characteristics | Use Case |
|----------|-----------------|----------|
| Real-time update | Update on every interaction | Small scale, strong real-time |
| Offline batch | Scheduled batch computation | Large scale |
| Hybrid | Real-time incremental + offline recalculation | Production environments |

### 2. Action Weights

```python
ACTION_WEIGHTS = {
    "purchase": 5.0,    # Purchase
    "add_cart": 3.0,    # Add to cart
    "favorite": 2.5,    # Favorite
    "click": 1.5,       # Click
    "view": 1.0,        # View
    "dislike": -2.0,    # Dislike
}
```

### 3. Time Decay

```python
def time_decay(timestamp, half_life_days=7):
    """Time decay with 7-day half-life"""
    import time
    days_ago = (time.time() - timestamp) / 86400
    return 0.5 ** (days_ago / half_life_days)
```

## Cold Start Handling

```python
def recommend_cold_start(self, user_info: dict):
    """New user cold start"""
    if user_info.get("interests"):
        # Has interest tags, recommend based on tags
        interest_text = " ".join(user_info["interests"])
        embedding = self.model.encode(interest_text).tolist()
        return self.search_items(embedding)
    else:
        # No information, return popular items
        return self.get_popular_items()
```

## Vertical Applications

See `verticals/` directory for detailed guides:
- `ecommerce.md` - E-commerce personalized recommendations
- `content.md` - Content feeds
- `recruitment.md` - Job recommendations

## Related Tools

- Similar items: `scenarios:item-to-item`
- Vectorization: `core:embedding`
- Indexing: `core:indexing`
