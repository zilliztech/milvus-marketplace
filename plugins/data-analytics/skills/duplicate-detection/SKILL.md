---
name: duplicate-detection
description: "Use when user needs to find duplicate or similar content. Triggers on: duplicate, deduplication, plagiarism detection, similar content, near-duplicate, similarity detection."
---

# Duplicate Detection

Batch detection of duplicate or highly similar content.

## Use Cases

- Plagiarism/content spinning detection
- Content deduplication (crawler data cleaning)
- Duplicate question merging (FAQ/tickets)
- Duplicate product detection
- Resume deduplication

## Architecture

```
Content to check → Vectorize → Compare with library → Similarity > threshold → Mark as duplicate
```

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
import hashlib

class DuplicateDetector:
    def __init__(self, uri: str = "./milvus.db", threshold: float = 0.9):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        self.threshold = threshold
        self.collection_name = "duplicate_detection"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("content_hash", DataType.VARCHAR, max_length=64)  # Exact dedup
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 256})
        index_params.add_index(field_name="content_hash", index_type="TRIE")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def _hash_content(self, content: str) -> str:
        """Calculate content hash"""
        return hashlib.md5(content.encode()).hexdigest()

    def check_duplicate(self, content: str, source: str = "") -> dict:
        """Check if single content is duplicate"""
        content_hash = self._hash_content(content)

        # 1. Exact match (hash)
        exact_match = self.client.query(
            collection_name=self.collection_name,
            filter=f'content_hash == "{content_hash}"',
            output_fields=["id", "source"],
            limit=1
        )
        if exact_match:
            return {
                "is_duplicate": True,
                "type": "exact",
                "match_id": exact_match[0]["id"],
                "match_source": exact_match[0]["source"],
                "similarity": 1.0
            }

        # 2. Semantic similarity (vector)
        embedding = self.model.encode(content).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=1,
            output_fields=["id", "content", "source"]
        )

        if results[0] and results[0][0]["distance"] >= self.threshold:
            return {
                "is_duplicate": True,
                "type": "similar",
                "match_id": results[0][0]["entity"]["id"],
                "match_source": results[0][0]["entity"]["source"],
                "similarity": results[0][0]["distance"],
                "match_content": results[0][0]["entity"]["content"][:200] + "..."
            }

        return {"is_duplicate": False, "similarity": results[0][0]["distance"] if results[0] else 0}

    def add_content(self, content_id: str, content: str, source: str = ""):
        """Add content to library (call after checking)"""
        content_hash = self._hash_content(content)
        embedding = self.model.encode(content).tolist()

        self.client.insert(
            collection_name=self.collection_name,
            data=[{
                "id": content_id,
                "content_hash": content_hash,
                "content": content,
                "source": source,
                "embedding": embedding
            }]
        )

    def batch_check(self, contents: list) -> list:
        """Batch check
        contents: [{"id": "...", "content": "...", "source": "..."}]
        """
        results = []
        for item in contents:
            result = self.check_duplicate(item["content"], item.get("source", ""))
            result["id"] = item["id"]
            results.append(result)

            # If not duplicate, add to library
            if not result["is_duplicate"]:
                self.add_content(item["id"], item["content"], item.get("source", ""))

        return results

    def find_all_duplicates(self, top_k: int = 100) -> list:
        """Find all duplicate content in library"""
        # Get all content
        all_items = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["id", "content", "embedding"],
            limit=10000
        )

        duplicate_groups = []
        processed = set()

        for item in all_items:
            if item["id"] in processed:
                continue

            # Search similar content
            results = self.client.search(
                collection_name=self.collection_name,
                data=[item["embedding"]],
                limit=top_k,
                output_fields=["id", "content"]
            )

            # Find similar ones
            group = [item["id"]]
            for hit in results[0]:
                if hit["entity"]["id"] != item["id"] and hit["distance"] >= self.threshold:
                    group.append(hit["entity"]["id"])
                    processed.add(hit["entity"]["id"])

            if len(group) > 1:
                duplicate_groups.append(group)

            processed.add(item["id"])

        return duplicate_groups

# Usage
detector = DuplicateDetector(threshold=0.85)

# Single check
result = detector.check_duplicate("This is an article about Python programming...")
if result["is_duplicate"]:
    print(f"Duplicate detected! Similarity: {result['similarity']:.2f}")
    print(f"Duplicates with {result['match_source']}")
else:
    print("Original content")
    detector.add_content("doc001", "This is an article about Python programming...", "blog.md")

# Batch check
results = detector.batch_check([
    {"id": "1", "content": "Python is a programming language", "source": "a.txt"},
    {"id": "2", "content": "Python is a programming language for coding", "source": "b.txt"},  # Similar
    {"id": "3", "content": "The weather is nice today", "source": "c.txt"},  # Different
])

for r in results:
    print(f"{r['id']}: {'Duplicate' if r['is_duplicate'] else 'Original'} (Similarity: {r['similarity']:.2f})")
```

## Threshold Selection

| Scenario | Recommended Threshold | Description |
|----------|----------------------|-------------|
| Strict dedup | 0.95+ | Nearly identical |
| Plagiarism detection | 0.85-0.90 | Allow rewording |
| Similar content | 0.75-0.85 | Topic related |
| Loose matching | 0.65-0.75 | Roughly related |

## Optimization Strategies

### 1. Chunk Detection (Long Text)

```python
def check_long_document(self, content: str, chunk_size: int = 500):
    """Long document chunk detection"""
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

    duplicate_chunks = []
    for i, chunk in enumerate(chunks):
        result = self.check_duplicate(chunk)
        if result["is_duplicate"]:
            duplicate_chunks.append({
                "chunk_index": i,
                "similarity": result["similarity"],
                "match_source": result["match_source"]
            })

    duplicate_ratio = len(duplicate_chunks) / len(chunks)
    return {
        "duplicate_ratio": duplicate_ratio,
        "is_duplicate": duplicate_ratio > 0.5,
        "duplicate_chunks": duplicate_chunks
    }
```

### 2. SimHash Pre-filtering

```python
from simhash import Simhash

def quick_filter(self, content: str) -> bool:
    """SimHash quick pre-filtering"""
    sh = Simhash(content)
    # Compare with SimHash in library, only do vector detection if distance < 3
    # Greatly reduces vector computation
```

## Vertical Applications

See `verticals/` directory for detailed guides:
- `plagiarism.md` - Plagiarism detection
- `content-dedup.md` - Content deduplication
- `faq-merge.md` - FAQ question merging

## Related Tools

- Vectorization: `core:embedding`
- Document chunking: `core:chunking`
- Clustering: `scenarios:clustering`
