---
name: item-to-item
description: "Use when user needs to find similar items. Triggers on: similar items, related content, related products, more like this, similar products, related articles, content-based recommendation."
---

# Item-to-Item Recommendation

Recommend related content based on item vector similarity.

## Use Cases

- Similar product recommendations ("Customers also viewed")
- Related article recommendations ("Related reading")
- Similar video recommendations
- Similar song recommendations

## Architecture

```
Current item → Get item vector → Vector search → Similar items list
```

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

class ItemToItemRecommender:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
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
        index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 256})
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
        # Generate embedding from title + description
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

    def get_similar(self, item_id: str, limit: int = 10, same_category: bool = False):
        """Get similar items"""
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

        # Search similar items
        filter_expr = f'category == "{category}"' if same_category else ""

        similar = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            filter=filter_expr,
            limit=limit + 1,  # Get one extra to exclude self
            output_fields=["id", "title", "category", "tags"]
        )

        # Exclude self
        return [{"id": hit["entity"]["id"],
                 "title": hit["entity"]["title"],
                 "category": hit["entity"]["category"],
                 "score": hit["distance"]}
                for hit in similar[0] if hit["entity"]["id"] != item_id][:limit]

    def get_similar_by_content(self, text: str, limit: int = 10, category: str = None):
        """Recommend based on content description"""
        embedding = self.model.encode(text).tolist()

        filter_expr = f'category == "{category}"' if category else ""

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            filter=filter_expr,
            limit=limit,
            output_fields=["id", "title", "category", "tags"]
        )

        return [{"id": hit["entity"]["id"],
                 "title": hit["entity"]["title"],
                 "category": hit["entity"]["category"],
                 "score": hit["distance"]} for hit in results[0]]

# Usage
recommender = ItemToItemRecommender()

# Add products
recommender.add_items([
    {"id": "p001", "title": "iPhone 15 Pro", "description": "Apple flagship phone, A17 chip", "category": "Phones", "tags": ["Apple", "Flagship"]},
    {"id": "p002", "title": "iPhone 15", "description": "Apple phone, A16 chip", "category": "Phones", "tags": ["Apple"]},
    {"id": "p003", "title": "Huawei Mate 60", "description": "Huawei flagship phone, Kirin chip", "category": "Phones", "tags": ["Huawei", "Flagship"]},
    {"id": "p004", "title": "AirPods Pro", "description": "Apple noise-canceling earbuds", "category": "Earbuds", "tags": ["Apple", "Noise-canceling"]},
])

# Find similar products
similar = recommender.get_similar("p001", limit=5)
# Returns: iPhone 15, Huawei Mate 60...

# Find similar in same category
similar = recommender.get_similar("p001", limit=5, same_category=True)
# Returns: iPhone 15, Huawei Mate 60 (all phones)

# Recommend by description
similar = recommender.get_similar_by_content("looking for Apple noise-canceling earbuds")
# Returns: AirPods Pro...
```

## Optimization Strategies

### 1. Multi-feature Fusion

```python
def create_item_embedding(self, item):
    """Fuse multiple features"""
    # Text features
    text = f"{item['title']} {item['description']}"
    text_emb = self.text_model.encode(text)

    # Image features (if available)
    if item.get("image_path"):
        image = Image.open(item["image_path"])
        image_emb = self.image_model.encode(image)
        # Fuse
        return np.concatenate([text_emb * 0.6, image_emb * 0.4])

    return text_emb
```

### 2. Popularity Weighting

```python
def get_similar_with_popularity(self, item_id: str, limit: int = 10):
    """Similar recommendations considering popularity"""
    similar = self.get_similar(item_id, limit=limit * 2)

    # Get popularity
    for item in similar:
        item["popularity"] = get_item_popularity(item["id"])
        item["final_score"] = item["score"] * 0.7 + item["popularity"] * 0.3

    # Sort by combined score
    similar.sort(key=lambda x: x["final_score"], reverse=True)
    return similar[:limit]
```

### 3. Diversity Optimization

```python
def get_diverse_similar(self, item_id: str, limit: int = 10):
    """Diverse recommendations (avoid too similar)"""
    candidates = self.get_similar(item_id, limit=limit * 3)

    selected = []
    for item in candidates:
        # Check similarity with already selected items
        is_diverse = all(
            self.similarity(item, s) < 0.9
            for s in selected
        )
        if is_diverse:
            selected.append(item)
            if len(selected) >= limit:
                break

    return selected
```

## Vertical Applications

See `verticals/` directory for detailed guides:
- `ecommerce.md` - Similar products
- `content.md` - Related articles/videos
- `music.md` - Similar songs

## Related Tools

- Vectorization: `core:embedding`
- Indexing: `core:indexing`
- User recommendations: `scenarios:user-to-item`
