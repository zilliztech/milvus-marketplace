# Enterprise Knowledge Base QA

## Data Characteristics

- Diverse document formats (PDF, Word, PPT, web pages)
- Many internal terms and abbreviations
- Permission control requirements
- Need to trace sources

## Recommended Configuration

| Config Item | Recommended Value | Notes |
|-------------|-------------------|-------|
| Embedding Model | `BAAI/bge-large-zh-v1.5` | Chinese |
| | `BAAI/bge-m3` | Multilingual |
| Chunk Size | 512 tokens | Balance precision and context |
| Chunk Overlap | 50-100 tokens | Preserve context |
| LLM | GPT-4o / Claude 3 | High-quality answers |
| | Qwen-72B | Private deployment |

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

# Source tracing
schema.add_field("source", DataType.VARCHAR, max_length=512)       # File path
schema.add_field("page", DataType.INT32)                           # Page number
schema.add_field("chunk_index", DataType.INT32)                    # Chunk index

# Permission control
schema.add_field("department", DataType.VARCHAR, max_length=64)    # Department
schema.add_field("access_level", DataType.INT32)                   # Access level 1-5
schema.add_field("tags", DataType.VARCHAR, max_length=256)         # Tags

# Metadata
schema.add_field("doc_type", DataType.VARCHAR, max_length=32)      # pdf/word/ppt
schema.add_field("author", DataType.VARCHAR, max_length=128)
schema.add_field("updated_at", DataType.INT64)                     # Update timestamp
```

## Permission Filtering

```python
def search_with_permission(query: str, user_info: dict, limit: int = 10):
    """Search with permission control"""
    user_dept = user_info.get("department", "")
    user_level = user_info.get("access_level", 1)

    # Build permission filter condition
    filter_expr = f'access_level <= {user_level}'

    # Department restriction (some documents visible only within department)
    if user_dept:
        filter_expr += f' and (department == "" or department == "{user_dept}")'

    return client.search(
        collection_name="enterprise_kb",
        data=[embed(query)],
        filter=filter_expr,
        limit=limit,
        output_fields=["content", "source", "page", "department"]
    )
```

## Source Citations

```python
def generate_answer_with_citations(question: str, user_info: dict) -> dict:
    """Generate answer with citations"""
    # Retrieve
    contexts = search_with_permission(question, user_info, limit=5)

    # Build prompt
    context_text = ""
    sources = []
    for i, ctx in enumerate(contexts[0]):
        context_text += f"[{i+1}] {ctx['entity']['content']}\n\n"
        sources.append({
            "index": i + 1,
            "source": ctx["entity"]["source"],
            "page": ctx["entity"]["page"]
        })

    prompt = f"""Answer the question based on the following reference materials. Use [1], [2], etc. to cite sources in your answer.

Reference Materials:
{context_text}

Question: {question}

Answer:"""

    response = llm.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources
    }
```

## Document Update Strategy

```python
def update_document(file_path: str, department: str, access_level: int):
    """Update a single document"""
    import time

    # Delete old chunks
    client.delete(
        collection_name="enterprise_kb",
        filter=f'source == "{file_path}"'
    )

    # Reprocess and insert
    chunks = process_document(file_path)

    data = []
    for i, chunk in enumerate(chunks):
        data.append({
            "id": f"{file_path}_{i}",
            "content": chunk["text"],
            "embedding": embed(chunk["text"]),
            "source": file_path,
            "page": chunk.get("page", 0),
            "chunk_index": i,
            "department": department,
            "access_level": access_level,
            "doc_type": file_path.split(".")[-1],
            "updated_at": int(time.time())
        })

    client.insert(collection_name="enterprise_kb", data=data)
```

## Example Queries

```python
# Regular employee query
user = {"department": "Sales", "access_level": 2}
result = generate_answer_with_citations("What is the company's reimbursement process?", user)

# Management query
admin = {"department": "Management", "access_level": 5}
result = generate_answer_with_citations("This quarter's financial report analysis", admin)
```

## Best Practices

1. **Regular Updates**: Set scheduled tasks to check for document changes
2. **Version Control**: Keep historical versions, support rollback
3. **Usage Feedback**: Collect user feedback to optimize retrieval quality
4. **Glossary**: Maintain internal company glossary to assist query rewriting
