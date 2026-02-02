# Legal Consultation RAG + Rerank

## Why Rerank is Needed

Legal consultation scenarios require extremely high accuracy:
- **One word difference, different meaning**: Legal terminology requires precision
- **Context dependent**: Need to understand query and answer correlation
- **Recall ≠ relevance**: Vector similarity doesn't equal legal relevance

## Recommended Configuration

| Config | Recommended Value | Description |
|--------|------------------|-------------|
| Embedding | `text-embedding-3-small` | OpenAI embedding |
| Reranker | `BAAI/bge-reranker-large` | Precision ranking |
| | `Cohere rerank-multilingual-v3.0` | Multilingual |
| Initial Recall | 30-50 | Recall more for reranking |
| After Rerank | 5-10 | Select for generation |
| LLM | GPT-4 / Claude 3 | Legal needs high quality |

## Data Preparation

```python
# Legal knowledge base structure
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

# Legal specific fields
schema.add_field("doc_type", DataType.VARCHAR, max_length=32)       # law/regulation/case/opinion
schema.add_field("law_category", DataType.VARCHAR, max_length=64)   # civil/criminal/labor/...
schema.add_field("effectiveness", DataType.VARCHAR, max_length=32)  # valid/invalid/amended
schema.add_field("publish_date", DataType.INT64)                    # Publish date
schema.add_field("source", DataType.VARCHAR, max_length=256)        # Source (law name/case number)
```

## Implementation

```python
from sentence_transformers import CrossEncoder
from pymilvus import MilvusClient
from openai import OpenAI

class LegalRAG:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()
        self.reranker = CrossEncoder('BAAI/bge-reranker-large')

    def _embed(self, text: str) -> list:
        """Generate embedding using OpenAI API"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

    def retrieve(self, query: str, law_category: str = None, limit: int = 30) -> list:
        """Initial recall"""
        embedding = self._embed(query)

        filter_expr = 'effectiveness == "valid"'  # Only search valid laws
        if law_category:
            filter_expr += f' and law_category == "{law_category}"'

        results = self.client.search(
            collection_name="legal_kb",
            data=[embedding],
            filter=filter_expr,
            limit=limit,
            output_fields=["content", "doc_type", "source", "law_category"]
        )

        return results[0]

    def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
        """Precision reranking"""
        # Build query-passage pairs
        pairs = [[query, c["entity"]["content"]] for c in candidates]

        # CrossEncoder scoring
        scores = self.reranker.predict(pairs)

        # Sort
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]

    def generate(self, query: str, contexts: list) -> dict:
        """Generate answer"""
        # Organize context by document type
        laws = [c for c in contexts if c["entity"]["doc_type"] == "law"]
        cases = [c for c in contexts if c["entity"]["doc_type"] == "case"]
        opinions = [c for c in contexts if c["entity"]["doc_type"] == "opinion"]

        context_text = ""

        if laws:
            context_text += "[Relevant Law Articles]\n"
            for l in laws:
                context_text += f"- {l['entity']['source']}: {l['entity']['content']}\n"

        if cases:
            context_text += "\n[Relevant Cases]\n"
            for c in cases:
                context_text += f"- {c['entity']['source']}: {c['entity']['content'][:500]}...\n"

        if opinions:
            context_text += "\n[Legal Interpretations/Opinions]\n"
            for o in opinions:
                context_text += f"- {o['entity']['source']}: {o['entity']['content']}\n"

        prompt = f"""You are a professional legal consultant. Please answer the user's legal question based on the following legal materials.

{context_text}

User question: {query}

Requirements:
1. When citing specific law articles, indicate the law name and article number
2. If there are relevant cases, briefly explain the case key points
3. Distinguish between definitive opinions and advisory suggestions
4. If the issue involves complex situations, recommend consulting a professional lawyer

Answer:"""

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2  # Legal answers need low temperature for accuracy
        )

        # Extract cited legal sources
        sources = list(set(c["entity"]["source"] for c in contexts))

        return {
            "answer": response.choices[0].message.content,
            "sources": sources,
            "contexts": [{"source": c["entity"]["source"],
                         "content": c["entity"]["content"][:200] + "...",
                         "type": c["entity"]["doc_type"],
                         "rerank_score": c["rerank_score"]}
                        for c in contexts]
        }

    def query(self, question: str, law_category: str = None) -> dict:
        """Complete QA pipeline"""
        # 1. Recall
        candidates = self.retrieve(question, law_category, limit=30)

        # 2. Rerank
        reranked = self.rerank(question, candidates, top_k=8)

        # 3. Generate
        return self.generate(question, reranked)
```

## Example Queries

```python
rag = LegalRAG()

# Labor law related
result = rag.query(
    "Can I claim double wages if the company didn't sign a labor contract?",
    law_category="labor"
)

# Contract disputes
result = rag.query(
    "If the other party breaches the contract, can I terminate it and claim compensation?",
    law_category="contract"
)

# General query
result = rag.query("What can I do if the landlord won't return my deposit after lease ends?")

print(result["answer"])
print("\nReference sources:")
for src in result["sources"]:
    print(f"  - {src}")
```

## Important Notes

1. **Timeliness**: Laws get amended, ensure database is updated promptly
2. **Jurisdiction**: Different regions may have different regulations
3. **Disclaimer**: AI answers are for reference only, important matters should consult lawyers
4. **Privacy Protection**: Don't log user sensitive case information
