# Legal Document Search

## Data Characteristics

- Long texts (contracts, judgments typically 10,000+ characters)
- Many professional terms
- Need precise matching for law article citations
- Multilingual (Chinese and English legal documents)

## Recommended Configuration

| Config Item | Recommended Value | Notes |
|-------------|-------------------|-------|
| Embedding Model | `BAAI/bge-large-zh-v1.5` | Chinese legal |
| | `BAAI/bge-m3` | Chinese-English mixed |
| Chunk Size | 512-1024 tokens | Legal clauses are typically longer |
| Chunk Overlap | 100-200 tokens | Preserve context |
| Index Type | HNSW | Accuracy priority |
| Distance Metric | COSINE | Semantic similarity |

## Chunking Strategy

```python
# Legal document special chunking: split by clauses
import re

def split_legal_document(text: str) -> list:
    """Split by legal clauses"""
    # Match patterns like "Article X", "第X条", etc.
    pattern = r'(第[一二三四五六七八九十百千\d]+条|Article\s+\d+)'

    parts = re.split(pattern, text)

    chunks = []
    current_chunk = ""

    for i, part in enumerate(parts):
        if re.match(pattern, part):
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = part
        else:
            current_chunk += part

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
```

## Metadata Fields

```python
schema.add_field("doc_type", DataType.VARCHAR, max_length=64)    # contract/judgment/regulation
schema.add_field("court_level", DataType.VARCHAR, max_length=32) # Court level
schema.add_field("case_type", DataType.VARCHAR, max_length=64)   # Civil/Criminal/Administrative
schema.add_field("year", DataType.INT32)                         # Year
schema.add_field("jurisdiction", DataType.VARCHAR, max_length=64) # Jurisdiction
```

## Search Optimization

1. **Law Article Citation Extraction**: Extract cited law articles during preprocessing, index separately
2. **Case Type Classification**: Classify cases to support filtered search
3. **Time Weighting**: Recent judgments may have more reference value

## Example Queries

```python
# Find similar cases
results = search.search(
    "loan contract dispute failure to repay on time penalty calculation",
    filter='doc_type == "judgment" and case_type == "Civil"'
)

# Find related legal provisions
results = search.search(
    "conditions for contract termination",
    filter='doc_type == "regulation"'
)
```
