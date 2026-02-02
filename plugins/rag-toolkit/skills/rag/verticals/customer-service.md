# Customer Service Knowledge Base QA

## Data Characteristics

- FAQ question-answer pairs
- Product documentation
- Historical ticket records
- Requires fast response
- Multi-turn conversation scenarios

## Recommended Configuration

| Config Item | Recommended Value | Notes |
|-------------|-------------------|-------|
| Embedding Model | `text-embedding-3-small` | OpenAI embedding |
| Chunk Size | 256-512 tokens | Short answers preferred |
| LLM | GPT-3.5-turbo | Speed priority |
| | GPT-4o-mini | Cost-effective |
| Retrieval Count | 3-5 | Customer service doesn't need too much context |

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

# Knowledge classification
schema.add_field("kb_type", DataType.VARCHAR, max_length=32)       # faq/doc/ticket
schema.add_field("category", DataType.VARCHAR, max_length=64)      # Product category
schema.add_field("product", DataType.VARCHAR, max_length=128)      # Specific product

# FAQ-specific fields
schema.add_field("question", DataType.VARCHAR, max_length=512)     # Original question
schema.add_field("answer", DataType.VARCHAR, max_length=65535)     # Standard answer

# Ticket-specific fields
schema.add_field("resolution", DataType.VARCHAR, max_length=65535) # Resolution
schema.add_field("solved", DataType.BOOL)                          # Whether solved

# Statistics fields
schema.add_field("hit_count", DataType.INT32)                      # Hit count
schema.add_field("helpful_count", DataType.INT32)                  # Helpful count
```

## Multi-Source Retrieval Strategy

```python
class CustomerServiceRAG:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()

    def _embed(self, text: str) -> list:
        """Generate embedding using OpenAI API"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

    def search_faq(self, query: str, limit: int = 3) -> list:
        """Search FAQ"""
        embedding = self._embed(query)
        return self.client.search(
            collection_name="customer_service_kb",
            data=[embedding],
            filter='kb_type == "faq"',
            limit=limit,
            output_fields=["question", "answer", "hit_count"]
        )

    def search_docs(self, query: str, product: str = "", limit: int = 3) -> list:
        """Search product documentation"""
        embedding = self._embed(query)
        filter_expr = 'kb_type == "doc"'
        if product:
            filter_expr += f' and product == "{product}"'

        return self.client.search(
            collection_name="customer_service_kb",
            data=[embedding],
            filter=filter_expr,
            limit=limit,
            output_fields=["content", "product"]
        )

    def search_similar_tickets(self, query: str, limit: int = 3) -> list:
        """Search similar tickets"""
        embedding = self._embed(query)
        return self.client.search(
            collection_name="customer_service_kb",
            data=[embedding],
            filter='kb_type == "ticket" and solved == true',
            limit=limit,
            output_fields=["content", "resolution"]
        )

    def answer(self, question: str, product: str = "") -> dict:
        """Generate answer"""
        # 1. Search FAQ first (exact match)
        faq_results = self.search_faq(question, limit=3)

        # If FAQ highly matches, return directly
        if faq_results[0] and faq_results[0][0]["distance"] > 0.9:
            top_faq = faq_results[0][0]["entity"]
            return {
                "answer": top_faq["answer"],
                "source": "faq",
                "confidence": "high"
            }

        # 2. Search documentation and tickets
        doc_results = self.search_docs(question, product, limit=3)
        ticket_results = self.search_similar_tickets(question, limit=2)

        # 3. Combine context
        context_parts = []

        for r in faq_results[0]:
            context_parts.append(f"FAQ: {r['entity']['question']}\nAnswer: {r['entity']['answer']}")

        for r in doc_results[0]:
            context_parts.append(f"Documentation: {r['entity']['content']}")

        for r in ticket_results[0]:
            context_parts.append(f"Historical Ticket: {r['entity']['content']}\nResolution: {r['entity']['resolution']}")

        context = "\n\n".join(context_parts)

        # 4. LLM generation
        prompt = f"""You are a professional customer service assistant. Answer user questions based on the following reference materials.

Reference Materials:
{context}

User Question: {question}

Requirements:
1. Keep answers concise and clear
2. If uncertain, indicate transfer to human agent is needed
3. Provide next step recommendations

Answer:"""

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return {
            "answer": response.choices[0].message.content,
            "source": "rag",
            "confidence": "medium"
        }
```

## Feedback Learning

```python
def record_feedback(kb_id: str, helpful: bool):
    """Record user feedback"""
    # Update hit count
    client.upsert(
        collection_name="customer_service_kb",
        data=[{
            "id": kb_id,
            "hit_count": {"$inc": 1},
            "helpful_count": {"$inc": 1 if helpful else 0}
        }]
    )

def get_low_quality_entries(threshold: float = 0.3, min_hits: int = 10):
    """Get low-quality entries (for manual review)"""
    results = client.query(
        collection_name="customer_service_kb",
        filter=f"hit_count >= {min_hits}",
        output_fields=["id", "question", "answer", "hit_count", "helpful_count"]
    )

    low_quality = []
    for r in results:
        helpful_rate = r["helpful_count"] / r["hit_count"]
        if helpful_rate < threshold:
            low_quality.append({**r, "helpful_rate": helpful_rate})

    return sorted(low_quality, key=lambda x: x["helpful_rate"])
```

## Examples

```python
rag = CustomerServiceRAG()

# Simple question (FAQ direct answer)
result = rag.answer("How to reset password?")

# Product-related question
result = rag.answer("How to fix printer paper jam?", product="HP LaserJet Pro")

# Complex question (needs multi-source information)
result = rag.answer("My order shows shipped but hasn't arrived after a week, what should I do?")
```
