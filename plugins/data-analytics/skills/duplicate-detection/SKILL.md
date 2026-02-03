---
name: duplicate-detection
description: "Use when user needs to find duplicate or similar content. Triggers on: duplicate, deduplication, plagiarism detection, similar content, near-duplicate, similarity detection, content dedup, find copies."
---

# Duplicate Detection

Batch detection of duplicate or highly similar content using vector similarity.

## When to Activate

Activate this skill when:
- User needs to **find duplicates** in a dataset
- User mentions "deduplication", "plagiarism", "similar content"
- User wants to **clean data** by removing near-duplicates
- User needs to **merge similar items** (FAQ, tickets, products)

**Do NOT activate** when:
- User needs general semantic search → use `semantic-search`
- User needs to group by topic → use `clustering`
- User needs recommendation → use `rec-system`

## Interactive Flow

### Step 1: Understand Duplicate Type

"What type of duplicates are you looking for?"

A) **Exact duplicates** (100% identical)
   - Copy-paste content
   - File deduplication
   - Use hash-based detection (fast)

B) **Near-duplicates** (semantically similar)
   - Paraphrased content
   - Rewritten articles
   - Use vector similarity (accurate)

C) **Both**
   - Hash for exact, vector for near
   - Most comprehensive

Which do you need? (A/B/C)

### Step 2: Determine Threshold

"How similar should content be to be considered duplicate?"

| Threshold | Interpretation | Use Case |
|-----------|----------------|----------|
| **0.95+** | Near identical | Strict deduplication |
| **0.85-0.95** | Very similar | Plagiarism detection |
| **0.75-0.85** | Related | FAQ merging |
| **0.65-0.75** | Loosely related | Topic clustering |

What threshold fits your needs?

### Step 3: Confirm Configuration

"Based on your requirements:

- **Method**: Hash + Vector (comprehensive)
- **Similarity threshold**: 0.90
- **Embedding model**: BGE-large

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Finding Twins

Think of duplicate detection as **finding twins in a crowd**:
- **Identical twins** (exact duplicates) → Same fingerprint (hash)
- **Fraternal twins** (near-duplicates) → Similar appearance (vectors)

```
┌─────────────────────────────────────────────────────────┐
│                  Duplicate Detection                     │
│                                                          │
│  New Content: "Machine learning requires lots of data"   │
│                         │                                │
│          ┌──────────────┴──────────────┐                │
│          │                             │                │
│          ▼                             ▼                │
│  ┌───────────────┐           ┌───────────────┐         │
│  │   Hash Check  │           │ Vector Search │         │
│  │   (Exact)     │           │  (Semantic)   │         │
│  └───────┬───────┘           └───────┬───────┘         │
│          │                           │                  │
│          ▼                           ▼                  │
│  "Content hash matches      "Similar to: 'ML needs     │
│   doc_123" → EXACT DUP       big datasets' (0.92)"     │
│                              → NEAR DUPLICATE           │
│                                                          │
│  Result: {"is_duplicate": true, "type": "near",         │
│           "similarity": 0.92, "match": "doc_123"}       │
└─────────────────────────────────────────────────────────┘
```

### Hash vs Vector Detection

| Method | Speed | Finds | Misses |
|--------|-------|-------|--------|
| **Hash** | ★★★★★ | Exact copies | Paraphrases |
| **Vector** | ★★★ | Semantic similarity | Different meanings same words |
| **Both** | ★★★ | Comprehensive | Best coverage |

## Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
import hashlib

class DuplicateDetector:
    def __init__(self, uri: str = "./milvus.db", threshold: float = 0.9):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.threshold = threshold
        self.collection_name = "duplicate_detection"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="content_hash", index_type="TRIE")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def _hash_content(self, content: str) -> str:
        """Calculate content hash (normalized)"""
        normalized = ''.join(content.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def check_duplicate(self, content: str, source: str = "") -> dict:
        """Check if content is duplicate"""
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
                "type": "near",
                "match_id": results[0][0]["entity"]["id"],
                "match_source": results[0][0]["entity"]["source"],
                "similarity": results[0][0]["distance"],
                "match_preview": results[0][0]["entity"]["content"][:200] + "..."
            }

        return {
            "is_duplicate": False,
            "similarity": results[0][0]["distance"] if results[0] else 0
        }

    def add_content(self, content_id: str, content: str, source: str = ""):
        """Add content to library"""
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

    def batch_dedup(self, items: list) -> dict:
        """Batch check and deduplicate
        items: [{"id": "...", "content": "...", "source": "..."}]
        Returns: {"unique": [...], "duplicates": [...]}
        """
        unique = []
        duplicates = []

        for item in items:
            result = self.check_duplicate(item["content"], item.get("source", ""))
            result["id"] = item["id"]

            if result["is_duplicate"]:
                duplicates.append(result)
            else:
                unique.append(item)
                self.add_content(item["id"], item["content"], item.get("source", ""))

        return {
            "unique": unique,
            "duplicates": duplicates,
            "unique_count": len(unique),
            "duplicate_count": len(duplicates),
            "duplicate_ratio": len(duplicates) / len(items) if items else 0
        }

# Usage
detector = DuplicateDetector(threshold=0.85)

# Check single content
result = detector.check_duplicate("Machine learning requires a lot of data")
if result["is_duplicate"]:
    print(f"Duplicate found! Similarity: {result['similarity']:.2f}")
else:
    detector.add_content("doc001", "Machine learning requires a lot of data", "blog.md")

# Batch deduplication
results = detector.batch_dedup([
    {"id": "1", "content": "Python is a programming language", "source": "a.txt"},
    {"id": "2", "content": "Python is a coding language", "source": "b.txt"},  # Near-duplicate
    {"id": "3", "content": "The weather is nice today", "source": "c.txt"},
])
print(f"Unique: {results['unique_count']}, Duplicates: {results['duplicate_count']}")
```

## Threshold Selection Guide

| Use Case | Threshold | Rationale |
|----------|-----------|-----------|
| Exact dedup (data cleaning) | 0.95+ | Only near-identical |
| Plagiarism detection | 0.85-0.90 | Allow rewording |
| FAQ merging | 0.80-0.85 | Same question, different words |
| Similar content grouping | 0.70-0.80 | Topically related |

## Common Pitfalls

### ❌ Pitfall 1: Threshold Too Low

**Problem**: Everything is marked as duplicate

**Why**: Low threshold catches even loosely related content

**Fix**: Start with 0.90 and adjust down if needed

### ❌ Pitfall 2: Hash Only Detection

**Problem**: Paraphrased duplicates not detected

**Why**: Hash only catches exact matches

**Fix**: Always use vector similarity for semantic duplicates

### ❌ Pitfall 3: Not Normalizing Before Hash

**Problem**: Same content with different whitespace not detected

**Why**: Hash is sensitive to exact characters

**Fix**: Normalize content before hashing
```python
normalized = ''.join(content.lower().split())
```

### ❌ Pitfall 4: Processing Order Matters

**Problem**: Different "original" detected depending on order

**Why**: First item becomes the reference

**Fix**: Process by timestamp (oldest first) or have clear policy

## When to Level Up

| Need | Upgrade To |
|------|------------|
| Group similar content | `clustering` |
| Plagiarism with source matching | `verticals/plagiarism.md` |
| Large-scale batch processing | Add `core:ray` |

## References

- Plagiarism detection: `verticals/plagiarism.md`
- Content deduplication: `verticals/content-dedup.md`
- FAQ merging: `verticals/faq-merge.md`
