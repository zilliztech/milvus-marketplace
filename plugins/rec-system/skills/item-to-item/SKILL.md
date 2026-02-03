---
name: item-to-item
description: "Use when user needs to find similar items. Triggers on: similar items, related content, related products, more like this, similar products, related articles, content-based recommendation, you may also like."
---

# Item-to-Item Recommendation

Find similar items based on content similarity — power "Customers also viewed" and "Related articles" features.

## When to Activate

Activate this skill when:
- User needs **"More like this"** functionality
- User mentions "similar products", "related articles", "you may also like"
- User wants to build **content-based recommendations**
- User's recommendation should be **item-centric** (not user-centric)

**Do NOT activate** when:
- User needs personalized recommendations based on user history → use `user-to-item`
- User needs semantic search → use `semantic-search`
- User needs to find duplicates → use `duplicate-detection`

## Interactive Flow

### Step 1: Understand Item Type

"What type of items are you recommending?"

| Item Type | Key Features | Embedding Strategy |
|-----------|--------------|-------------------|
| **Products** | Title, description, specs | Text + optional image |
| **Articles** | Title, content | Text (title + abstract) |
| **Videos** | Title, description, thumbnail | Text + keyframe |
| **Music** | Title, artist, genre | Text + audio features |

Which item type? (or describe)

### Step 2: Determine Similarity Scope

"Should similar items be in the same category?"

A) **Same category only**: Similar phones among phones
B) **Cross-category**: Phone might recommend case, charger
C) **Configurable**: User chooses at runtime

### Step 3: Confirm Configuration

"Based on your requirements:

- **Embedding**: Title + Description (BGE-large)
- **Category filter**: Optional at search time
- **Diversity**: Enabled (avoid too-similar results)

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Store Shelf Arrangement

Think of item-to-item as **arranging a store shelf**:
- Put similar products near each other
- When customer picks one, show nearby items
- "If you like this, check out these neighbors"

```
┌─────────────────────────────────────────────────────────┐
│              Item-to-Item Recommendation                 │
│                                                          │
│  Current Item: iPhone 15 Pro                             │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────────────┐                                    │
│  │  Get Embedding  │                                    │
│  │  from storage   │                                    │
│  └────────┬────────┘                                    │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐                                    │
│  │  Vector Search  │  Find nearest neighbors            │
│  │  (exclude self) │  in embedding space                │
│  └────────┬────────┘                                    │
│           │                                              │
│           ▼                                              │
│  Similar Items:                                          │
│  ┌─────┬─────┬─────┬─────┐                             │
│  │iPh15│Pix 8│Sam24│iPh14│                             │
│  │0.95 │0.87 │0.85 │0.82 │  (similarity scores)        │
│  └─────┴─────┴─────┴─────┘                             │
│                                                          │
│  "Customers who viewed iPhone 15 Pro also viewed..."    │
└─────────────────────────────────────────────────────────┘
```

### Item-to-Item vs User-to-Item

| Aspect | Item-to-Item | User-to-Item |
|--------|--------------|--------------|
| **Input** | Current item | User profile |
| **Logic** | Find similar items | Match user preferences |
| **Use case** | "Related products" | "For you" homepage |
| **Cold start** | No issue | Needs user data |
| **Personalization** | None | High |

## Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

class ItemToItemRecommender:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.collection_name = "item_to_item"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("title", DataType.VARCHAR, max_length=1024)
        schema.add_field("description", DataType.VARCHAR, max_length=65535)
        schema.add_field("category", DataType.VARCHAR, max_length=256)
        schema.add_field("tags", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20, max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="category", index_type="TRIE")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add_items(self, items: list):
        """Add items
        items: [{"id": "...", "title": "...", "description": "...", "category": "...", "tags": [...]}]
        """
        texts = [f"{item['title']} {item['description']}" for item in items]
        embeddings = self.model.encode(texts).tolist()

        data = []
        for item, emb in zip(items, embeddings):
            data.append({
                "id": item["id"],
                "title": item["title"],
                "description": item["description"],
                "category": item.get("category", ""),
                "tags": item.get("tags", []),
                "embedding": emb
            })

        self.client.insert(collection_name=self.collection_name, data=data)

    def get_similar(self, item_id: str, limit: int = 10, same_category: bool = False) -> list:
        """Find similar items"""
        # Get current item's embedding
        results = self.client.get(
            collection_name=self.collection_name,
            ids=[item_id],
            output_fields=["embedding", "category"]
        )

        if not results:
            return []

        embedding = results[0]["embedding"]
        category = results[0]["category"]

        # Build filter (exclude self, optionally same category)
        filter_expr = f'id != "{item_id}"'
        if same_category and category:
            filter_expr += f' and category == "{category}"'

        # Search
        similar = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            filter=filter_expr,
            limit=limit,
            output_fields=["id", "title", "category", "tags"]
        )

        return [{"id": hit["entity"]["id"],
                 "title": hit["entity"]["title"],
                 "category": hit["entity"]["category"],
                 "score": hit["distance"]}
                for hit in similar[0]]

    def get_similar_by_text(self, text: str, limit: int = 10, category: str = None) -> list:
        """Find similar items by description (for items not in database)"""
        embedding = self.model.encode(text).tolist()

        filter_expr = f'category == "{category}"' if category else ""

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            filter=filter_expr if filter_expr else None,
            limit=limit,
            output_fields=["id", "title", "category"]
        )

        return [{"id": hit["entity"]["id"],
                 "title": hit["entity"]["title"],
                 "category": hit["entity"]["category"],
                 "score": hit["distance"]} for hit in results[0]]

    def get_diverse_similar(self, item_id: str, limit: int = 10, diversity_threshold: float = 0.85) -> list:
        """Find similar but diverse items (avoid too-similar results)"""
        # Get more candidates
        candidates = self.get_similar(item_id, limit=limit * 3)

        selected = []
        for candidate in candidates:
            # Check diversity against already selected
            is_diverse = True
            for s in selected:
                if candidate["score"] > diversity_threshold and s["category"] == candidate["category"]:
                    # Too similar AND same category, skip for diversity
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(candidate)
                if len(selected) >= limit:
                    break

        return selected

# Usage
recommender = ItemToItemRecommender()

recommender.add_items([
    {"id": "p001", "title": "iPhone 15 Pro", "description": "Apple flagship phone with A17 chip", "category": "Phones", "tags": ["Apple", "Flagship"]},
    {"id": "p002", "title": "iPhone 15", "description": "Apple phone with A16 chip", "category": "Phones", "tags": ["Apple"]},
    {"id": "p003", "title": "Samsung Galaxy S24", "description": "Samsung flagship with AI features", "category": "Phones", "tags": ["Samsung", "Flagship"]},
    {"id": "p004", "title": "AirPods Pro", "description": "Apple noise-canceling earbuds", "category": "Audio", "tags": ["Apple", "Earbuds"]},
])

# Basic similar items
similar = recommender.get_similar("p001", limit=5)
print("Similar to iPhone 15 Pro:")
for item in similar:
    print(f"  {item['title']} ({item['score']:.2f})")

# Same category only
similar_phones = recommender.get_similar("p001", limit=5, same_category=True)

# Diverse recommendations
diverse = recommender.get_diverse_similar("p001", limit=5)
```

## Optimization Strategies

### 1. Multi-Feature Fusion

```python
def create_rich_embedding(self, item: dict) -> list:
    """Combine text and image features"""
    # Text embedding
    text = f"{item['title']} {item['description']}"
    text_emb = self.text_model.encode(text)

    # Image embedding (if available)
    if item.get("image_path"):
        image = Image.open(item["image_path"])
        image_emb = self.image_model.encode(image)
        # Weighted fusion
        return np.concatenate([text_emb * 0.7, image_emb * 0.3]).tolist()

    return text_emb.tolist()
```

### 2. Popularity Boost

```python
def get_similar_with_popularity(self, item_id: str, limit: int = 10) -> list:
    """Boost popular items in recommendations"""
    similar = self.get_similar(item_id, limit=limit * 2)

    for item in similar:
        popularity = self.get_item_popularity(item["id"])  # 0-1
        item["final_score"] = item["score"] * 0.8 + popularity * 0.2

    similar.sort(key=lambda x: x["final_score"], reverse=True)
    return similar[:limit]
```

### 3. Business Rules

```python
def apply_business_rules(self, item_id: str, similar: list) -> list:
    """Apply business rules to recommendations"""
    current_item = self.get_item(item_id)

    filtered = []
    for item in similar:
        # Rule 1: Don't recommend out-of-stock items
        if not item.get("in_stock", True):
            continue

        # Rule 2: Don't recommend higher-priced items (upsell separately)
        if item.get("price", 0) > current_item.get("price", 0) * 1.5:
            continue

        filtered.append(item)

    return filtered
```

## Common Pitfalls

### ❌ Pitfall 1: Including Self in Results

**Problem**: Item recommends itself

**Fix**: Always exclude current item
```python
filter_expr = f'id != "{item_id}"'
```

### ❌ Pitfall 2: Too Similar Results

**Problem**: All recommendations are basically the same

**Fix**: Add diversity threshold
```python
# Skip if too similar to already selected items
if similarity > 0.95:
    continue
```

### ❌ Pitfall 3: Ignoring Availability

**Problem**: Recommending out-of-stock items

**Fix**: Add availability filter
```python
filter_expr = f'id != "{item_id}" and in_stock == true'
```

### ❌ Pitfall 4: Only Text-Based

**Problem**: Visually similar items not recommended

**Fix**: Include image embeddings for visual products

## When to Level Up

| Need | Upgrade To |
|------|------------|
| Personalized recommendations | `user-to-item` |
| Include multiple modalities | Combine text + image |
| Real-time updates | Add streaming pipeline |
| A/B testing | Track click-through rates |

## References

- Vertical guides: `verticals/`
- User recommendations: `rec-system:user-to-item`
- Embedding models: `core:embedding`
