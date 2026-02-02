# Academic Paper Search

## Data Characteristics

- Clear structure (title, abstract, body, references)
- Dense professional terminology
- Cross-language requirements (Chinese and English papers)
- Citation relationships are important

## Recommended Configuration

| Config Item | Recommended Value | Notes |
|-------------|-------------------|-------|
| Embedding Model | `BAAI/bge-m3` | Multilingual academic |
| | `allenai/specter2` | English papers specialized |
| | `sentence-transformers/allenai-specter` | Lightweight version |
| Chunking Strategy | By section + paragraph | Preserve structure |
| Index Type | HNSW | Accuracy priority |

## Multi-Field Indexing Strategy

```python
# Papers should index multiple fields for separate searching
schema.add_field("title_embedding", DataType.FLOAT_VECTOR, dim=1024)
schema.add_field("abstract_embedding", DataType.FLOAT_VECTOR, dim=1024)
schema.add_field("content_embedding", DataType.FLOAT_VECTOR, dim=1024)

# Metadata
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("authors", DataType.VARCHAR, max_length=1024)
schema.add_field("venue", DataType.VARCHAR, max_length=256)      # Journal/Conference
schema.add_field("year", DataType.INT32)
schema.add_field("citations", DataType.INT32)                    # Citation count
schema.add_field("field", DataType.VARCHAR, max_length=128)      # Research field
schema.add_field("keywords", DataType.VARCHAR, max_length=512)
```

## Search Strategies

### 1. Title-Priority Search

```python
def search_by_relevance(query: str, limit: int = 10):
    """Search title first, then abstract"""
    # Search titles
    title_results = client.search(
        collection_name="papers",
        data=[embed(query)],
        anns_field="title_embedding",
        limit=limit,
        output_fields=["title", "authors", "year"]
    )

    # Search abstracts
    abstract_results = client.search(
        collection_name="papers",
        data=[embed(query)],
        anns_field="abstract_embedding",
        limit=limit,
        output_fields=["title", "authors", "year"]
    )

    # Merge and deduplicate, title matches prioritized
    return merge_results(title_results, abstract_results)
```

### 2. Citation Weighting

```python
def search_with_citation_boost(query: str, limit: int = 10):
    """Boost highly cited papers"""
    results = client.search(
        collection_name="papers",
        data=[embed(query)],
        limit=limit * 2,  # Fetch more for re-ranking
        output_fields=["title", "citations", "year"]
    )

    # Citation weighting (logarithmic decay + time decay)
    import math
    current_year = 2024

    for r in results[0]:
        citation_boost = math.log10(r["entity"]["citations"] + 1) / 5
        recency_boost = (r["entity"]["year"] - 2000) / (current_year - 2000) * 0.1
        r["score"] = r["distance"] + citation_boost + recency_boost

    results[0].sort(key=lambda x: x["score"], reverse=True)
    return results[0][:limit]
```

## Special Features

### Related Paper Recommendations

```python
def find_related_papers(paper_id: str, limit: int = 10):
    """Find related papers based on paper content"""
    # Get paper embedding
    paper = client.get(collection_name="papers", ids=[paper_id])

    # Search similar papers (exclude self)
    results = client.search(
        collection_name="papers",
        data=[paper[0]["abstract_embedding"]],
        filter=f'id != "{paper_id}"',
        limit=limit
    )
    return results
```

## Example Queries

```python
# Search papers in specific field
results = search.search(
    "transformer attention mechanism",
    filter='field == "NLP" and year >= 2020'
)

# Search highly cited papers
results = search.search(
    "large language model",
    filter='citations >= 100'
)
```
