---
name: user-to-item
description: "Use when user needs personalized recommendations based on user profile. Triggers on: personalized, user recommendation, personalized recommendations, for you, feed, user preference, homepage recommendations."
---

# User-to-Item Personalized Recommendation

Build personalized recommendation feeds based on user behavior history — power "For You" and personalized homepage features.

## When to Activate

Activate this skill when:
- User needs **personalized recommendations** for individual users
- User mentions "for you", "personalized feed", "user preferences"
- User wants to build **homepage recommendations** based on history
- User's recommendation should be **user-centric** (not item-centric)

**Do NOT activate** when:
- User needs "similar items" for a product → use `item-to-item`
- User needs semantic search → use `semantic-search`
- User has no user behavior data

## Interactive Flow

### Step 1: Understand User Data

"What user behavior data do you have?"

| Data Type | Weight | Example |
|-----------|--------|---------|
| **Purchases** | High (3-5x) | User bought product X |
| **Add to cart** | Medium (2x) | User added X to cart |
| **Clicks** | Low (1x) | User clicked on X |
| **Views** | Lowest (0.5x) | User viewed X |
| **Dislikes** | Negative | User marked X as not interested |

Which data types do you have?

### Step 2: Cold Start Strategy

"How should we handle new users?"

A) **Popular items**: Show trending/popular items
B) **Category-based**: Ask for initial preferences
C) **Hybrid**: Popular initially, then personalize

### Step 3: Confirm Configuration

"Based on your requirements:

- **User profile**: Weighted average of interacted items
- **Time decay**: 7-day half-life
- **Cold start**: Popular items fallback

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Personal Shopper

Think of user-to-item as a **personal shopper who knows your taste**:
- Remembers what you bought, browsed, liked
- Learns your preferences over time
- Suggests items matching your taste profile

```
┌─────────────────────────────────────────────────────────┐
│            User-to-Item Recommendation                   │
│                                                          │
│  User History:                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Bought: ML Book (w=3)                            │   │
│  │ Clicked: Python Tutorial (w=1.5)                 │   │
│  │ Viewed: Data Science Course (w=1)                │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │         User Profile Vector                       │   │
│  │  = Weighted Average of Item Embeddings            │   │
│  │                                                   │   │
│  │  user_vec = (3×ML_vec + 1.5×Py_vec + 1×DS_vec)   │   │
│  │             / (3 + 1.5 + 1)                       │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Vector Search in Item Space               │   │
│  │         (exclude already interacted)              │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  Recommended Items:                                      │
│  ┌─────┬─────┬─────┬─────┐                             │
│  │AI   │Deep │Stats│NLP  │  (matches user taste)       │
│  │Book │Learn│Book │Course│                            │
│  └─────┴─────┴─────┴─────┘                             │
└─────────────────────────────────────────────────────────┘
```

### User Profile Construction

| Method | Pros | Cons |
|--------|------|------|
| **Weighted Average** | Simple, fast | No learning |
| **Last N items** | Captures recent interest | Ignores long-term |
| **Time-decayed** | Balances recency and history | Requires tuning |

## Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
import numpy as np
import time

class UserToItemRecommender:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.dim = 1024
        self._init_collections()

    def _init_collections(self):
        # Item collection
        if not self.client.has_collection("items"):
            schema = self.client.create_schema()
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("title", DataType.VARCHAR, max_length=1024)
            schema.add_field("category", DataType.VARCHAR, max_length=256)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")

            self.client.create_collection(collection_name="items", schema=schema, index_params=index_params)

        # User profile collection
        if not self.client.has_collection("user_profiles"):
            schema = self.client.create_schema()
            schema.add_field("user_id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
            schema.add_field("updated_at", DataType.INT64)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")

            self.client.create_collection(collection_name="user_profiles", schema=schema, index_params=index_params)

    def add_items(self, items: list):
        """Add items to catalog"""
        texts = [f"{item['title']} {item.get('description', '')}" for item in items]
        embeddings = self.model.encode(texts).tolist()

        data = [{"id": item["id"], "title": item["title"],
                 "category": item.get("category", ""), "embedding": emb}
                for item, emb in zip(items, embeddings)]

        self.client.insert(collection_name="items", data=data)

    def update_user_profile(self, user_id: str, interactions: list, decay_rate: float = 0.9):
        """Update user profile from interactions
        interactions: [{"item_id": "...", "action": "view/click/purchase", "timestamp": ...}]
        """
        # Action weights
        action_weights = {
            "purchase": 5.0,
            "add_cart": 3.0,
            "favorite": 2.5,
            "click": 1.5,
            "view": 1.0,
            "dislike": -2.0,
        }

        # Get item embeddings
        item_ids = [i["item_id"] for i in interactions]
        items = self.client.get(collection_name="items", ids=item_ids, output_fields=["embedding"])

        if not items:
            return

        item_emb_map = {item["id"]: item["embedding"] for item in items}

        # Calculate weighted user embedding
        embeddings = []
        weights = []

        for i, inter in enumerate(interactions):
            if inter["item_id"] not in item_emb_map:
                continue

            action = inter.get("action", "view")
            action_weight = action_weights.get(action, 1.0)

            # Time decay (more recent = higher weight)
            time_weight = decay_rate ** i  # i=0 is most recent

            embeddings.append(item_emb_map[inter["item_id"]])
            weights.append(action_weight * time_weight)

        if not embeddings:
            return

        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        user_embedding = np.average(embeddings, axis=0, weights=weights).tolist()

        # Store user profile
        self.client.upsert(
            collection_name="user_profiles",
            data=[{
                "user_id": user_id,
                "embedding": user_embedding,
                "updated_at": int(time.time())
            }]
        )

    def recommend(self, user_id: str, limit: int = 10,
                  exclude_ids: list = None, category: str = None) -> list:
        """Get personalized recommendations for user"""
        # Get user profile
        users = self.client.get(
            collection_name="user_profiles",
            ids=[user_id],
            output_fields=["embedding"]
        )

        if not users:
            # Cold start: return popular items
            return self.get_popular_items(limit)

        user_embedding = users[0]["embedding"]

        # Build filter
        filter_parts = []
        if exclude_ids:
            ids_str = ", ".join([f'"{id}"' for id in exclude_ids])
            filter_parts.append(f'id not in [{ids_str}]')
        if category:
            filter_parts.append(f'category == "{category}"')

        filter_expr = " and ".join(filter_parts) if filter_parts else None

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

    def get_popular_items(self, limit: int = 10) -> list:
        """Fallback for cold start users"""
        # In production, use actual popularity metrics
        results = self.client.query(
            collection_name="items",
            filter="",
            limit=limit,
            output_fields=["id", "title", "category"]
        )
        return results

    def recommend_with_exploration(self, user_id: str, limit: int = 10,
                                    explore_ratio: float = 0.2) -> list:
        """Balance exploitation (known preferences) with exploration (new content)"""
        exploit_count = int(limit * (1 - explore_ratio))
        explore_count = limit - exploit_count

        # Exploitation: based on user profile
        exploit_recs = self.recommend(user_id, limit=exploit_count)

        # Exploration: random diverse items
        explore_recs = self.get_diverse_items(limit=explore_count,
                                               exclude_ids=[r["id"] for r in exploit_recs])

        return exploit_recs + explore_recs

# Usage
recommender = UserToItemRecommender()

# Add items
recommender.add_items([
    {"id": "a1", "title": "Python Programming Guide", "category": "Programming"},
    {"id": "a2", "title": "Machine Learning Fundamentals", "category": "AI"},
    {"id": "a3", "title": "Deep Learning in Practice", "category": "AI"},
    {"id": "a4", "title": "JavaScript Mastery", "category": "Programming"},
    {"id": "a5", "title": "Cooking for Beginners", "category": "Lifestyle"},
])

# Record user behavior
recommender.update_user_profile("user001", [
    {"item_id": "a2", "action": "purchase"},
    {"item_id": "a3", "action": "click"},
    {"item_id": "a1", "action": "view"},
])

# Get personalized recommendations
recs = recommender.recommend("user001", limit=5)
print("For You:")
for r in recs:
    print(f"  {r['title']} ({r['score']:.2f})")
# Will prioritize AI content based on user's history
```

## User Profile Strategies

### Time Decay Function

```python
def time_decay_weight(timestamp: int, half_life_days: int = 7) -> float:
    """Exponential decay with configurable half-life"""
    days_ago = (time.time() - timestamp) / 86400
    return 0.5 ** (days_ago / half_life_days)
```

### Action Weights Table

| Action | Suggested Weight | Rationale |
|--------|-----------------|-----------|
| Purchase | 5.0 | Strongest intent signal |
| Add to cart | 3.0 | High purchase intent |
| Favorite/save | 2.5 | Explicit interest |
| Click | 1.5 | Some interest |
| View/impression | 1.0 | Baseline |
| Skip/hide | -1.0 | Negative signal |
| Dislike | -2.0 | Strong negative |

## Cold Start Strategies

### For New Users

```python
def recommend_cold_start(self, user_info: dict) -> list:
    """Handle users with no history"""
    # Option 1: Use explicit preferences
    if user_info.get("interests"):
        interest_text = " ".join(user_info["interests"])
        embedding = self.model.encode(interest_text).tolist()
        return self.search_by_embedding(embedding)

    # Option 2: Demographics-based
    if user_info.get("age_group"):
        return self.get_popular_by_demographic(user_info["age_group"])

    # Option 3: Global popular
    return self.get_popular_items()
```

### For New Items

```python
def handle_new_item(self, item_id: str) -> None:
    """Boost new items for discovery"""
    # Add to "new arrivals" category
    # Include in exploration recommendations
    # Use content-based similarity initially
    pass
```

## Common Pitfalls

### ❌ Pitfall 1: Filter Bubble

**Problem**: User only sees similar content, no discovery

**Fix**: Add exploration ratio
```python
# 80% based on preferences, 20% diverse exploration
recommendations = recommend_with_exploration(user_id, explore_ratio=0.2)
```

### ❌ Pitfall 2: Stale Profiles

**Problem**: Recommendations don't reflect changed interests

**Fix**: Time decay + recent activity boost
```python
# Recent actions weighted more heavily
time_weight = 0.9 ** days_since_action
```

### ❌ Pitfall 3: Recommending Already Purchased

**Problem**: User sees items they already bought

**Fix**: Always exclude previous interactions
```python
exclude_ids = get_user_purchase_history(user_id)
recommendations = recommend(user_id, exclude_ids=exclude_ids)
```

### ❌ Pitfall 4: Cold Start Black Hole

**Problem**: New users get nothing or irrelevant items

**Fix**: Have explicit cold start strategy
```python
if not user_has_history(user_id):
    return get_onboarding_recommendations(user_info)
```

## When to Level Up

| Need | Upgrade To |
|------|------------|
| Similar items (not personalized) | `item-to-item` |
| Real-time profile updates | Add streaming pipeline |
| Multi-objective optimization | Add re-ranking layer |
| A/B testing | Track engagement metrics |

## References

- Similar items: `rec-system:item-to-item`
- Vertical guides: `verticals/`
- Embedding models: `core:embedding`
