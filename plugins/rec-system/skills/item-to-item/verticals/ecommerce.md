# E-commerce Similar Product Recommendations

## Use Cases

- Product detail page "Similar Products"
- "Also Bought" recommendations
- Out-of-stock alternative recommendations
- Competitive analysis

## Recommended Configuration

| Config | Recommended Value | Description |
|--------|------------------|-------------|
| Embedding Model | `BAAI/bge-large-en-v1.5` | Text features |
| | `openai/clip-vit-base-patch32` | Image features |
| Feature Combination | Text + Image + Attributes | Multi-dimensional similarity |
| Index Type | HNSW | High precision |
| | IVF_PQ | Large data volume |

## Feature Design

### Option 1: Pure Text Features

```python
def create_text_embedding(product: dict) -> list:
    """Create vector based on product text"""
    # Combine key text
    text = f"{product['title']} {product['brand']} {product['category']}"

    # Add attributes
    if product.get("attributes"):
        attrs = " ".join([f"{k}:{v}" for k, v in product["attributes"].items()])
        text += f" {attrs}"

    return model.encode(text).tolist()
```

### Option 2: Multimodal Features

```python
def create_multimodal_embedding(product: dict) -> list:
    """Text + image fusion features"""
    # Text vector (1024 dim)
    text = f"{product['title']} {product['brand']}"
    text_emb = text_model.encode(text)

    # Image vector (512 dim)
    image = load_image(product["image_url"])
    image_emb = clip_model.encode(image)

    # Concatenate or weighted fusion
    # Option A: Concatenate
    combined = np.concatenate([text_emb, image_emb])  # 1536 dim

    # Option B: Weighted average (requires same dimensions)
    # combined = 0.6 * text_emb + 0.4 * image_emb

    return combined.tolist()
```

### Option 3: Structured Attribute Features

```python
def create_attribute_embedding(product: dict) -> list:
    """Vector based on structured attributes"""
    # Normalize numeric attributes
    price_norm = product["price"] / 10000  # Assume max price 10k
    rating_norm = product["rating"] / 5

    # Category one-hot
    category_vec = category_encoder.transform([product["category"]])

    # Brand embedding
    brand_emb = brand_embeddings.get(product["brand"], default_brand_emb)

    # Combine
    return np.concatenate([
        [price_norm, rating_norm],
        category_vec,
        brand_emb
    ]).tolist()
```

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("product_id", DataType.VARCHAR, max_length=64)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

# Filter fields
schema.add_field("category_l1", DataType.VARCHAR, max_length=64)
schema.add_field("category_l2", DataType.VARCHAR, max_length=64)
schema.add_field("brand", DataType.VARCHAR, max_length=128)
schema.add_field("price", DataType.FLOAT)
schema.add_field("price_tier", DataType.VARCHAR, max_length=16)    # low/mid/high
schema.add_field("in_stock", DataType.BOOL)
schema.add_field("shop_id", DataType.VARCHAR, max_length=64)
```

## Implementation

```python
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

class SimilarProductRecommender:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    def get_similar(self, product_id: str, limit: int = 10,
                    same_category: bool = True,
                    same_price_tier: bool = True,
                    exclude_same_shop: bool = True) -> list:
        """Get similar products"""
        # Get current product
        product = self.client.get(
            collection_name="products",
            ids=[product_id],
            output_fields=["embedding", "category_l1", "price_tier", "shop_id"]
        )

        if not product:
            return []

        product = product[0]

        # Build filter conditions
        filters = [
            f'product_id != "{product_id}"',
            'in_stock == true'
        ]

        if same_category:
            filters.append(f'category_l1 == "{product["category_l1"]}"')

        if same_price_tier:
            filters.append(f'price_tier == "{product["price_tier"]}"')

        if exclude_same_shop:
            filters.append(f'shop_id != "{product["shop_id"]}"')

        filter_expr = ' and '.join(filters)

        # Search
        results = self.client.search(
            collection_name="products",
            data=[product["embedding"]],
            filter=filter_expr,
            limit=limit,
            output_fields=["product_id", "title", "price", "brand", "image_url"]
        )

        return [{
            "product_id": r["entity"]["product_id"],
            "title": r["entity"]["title"],
            "price": r["entity"]["price"],
            "brand": r["entity"]["brand"],
            "image_url": r["entity"]["image_url"],
            "similarity": r["distance"]
        } for r in results[0]]

    def get_alternatives(self, product_id: str, limit: int = 5) -> list:
        """Get alternatives (for out-of-stock scenarios)"""
        product = self.client.get(
            collection_name="products",
            ids=[product_id],
            output_fields=["embedding", "category_l2", "price"]
        )

        if not product:
            return []

        product = product[0]
        price = product["price"]

        # Alternatives: same L2 category, similar price (±20%)
        filter_expr = f'''
            product_id != "{product_id}"
            and in_stock == true
            and category_l2 == "{product["category_l2"]}"
            and price >= {price * 0.8}
            and price <= {price * 1.2}
        '''

        results = self.client.search(
            collection_name="products",
            data=[product["embedding"]],
            filter=filter_expr,
            limit=limit,
            output_fields=["product_id", "title", "price"]
        )

        return results[0]

    def get_complementary(self, product_id: str, limit: int = 5) -> list:
        """Get complementary products (frequently bought together)"""
        # This requires purchase history data for training
        # Simplified version: rule-based
        product = self.client.get(
            collection_name="products",
            ids=[product_id],
            output_fields=["category_l2"]
        )

        if not product:
            return []

        category = product[0]["category_l2"]

        # Complementary category mapping
        complementary_map = {
            "phones": ["phone cases", "screen protectors", "chargers"],
            "laptops": ["laptop bags", "mice", "keyboards"],
            "cameras": ["camera bags", "memory cards", "tripods"],
        }

        comp_categories = complementary_map.get(category, [])
        if not comp_categories:
            return []

        # Search complementary products
        results = self.client.query(
            collection_name="products",
            filter=f'category_l2 in {comp_categories} and in_stock == true',
            output_fields=["product_id", "title", "price", "category_l2"],
            limit=limit
        )

        return results
```

## Examples

```python
recommender = SimilarProductRecommender()

# Similar products
similar = recommender.get_similar(
    product_id="iphone15_001",
    limit=10,
    same_category=True,
    exclude_same_shop=True
)

print("Similar products:")
for p in similar:
    print(f"  {p['title']} - ${p['price']} (similarity: {p['similarity']:.2f})")

# Alternatives (out-of-stock scenario)
alternatives = recommender.get_alternatives("iphone15_001")

# Complementary purchases
complementary = recommender.get_complementary("iphone15_001")
```

## Optimization Strategies

1. **Popular product caching**: Pre-compute similar product lists for popular items
2. **Partition search**: Partition by L1 category, reduce search scope
3. **Multi-path recall**: Same brand + same category + same price tier multi-path recall and fusion
4. **A/B testing**: Compare click rates of different similarity algorithms
