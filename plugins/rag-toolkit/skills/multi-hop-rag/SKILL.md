---
name: multi-hop-rag
description: "Use when user needs multi-step reasoning with retrieval. Triggers on: multi-hop, multi-step RAG, complex questions, chain of retrieval, iterative retrieval, complex reasoning, cross-document reasoning."
---

# Multi-Hop RAG

For complex questions, perform multiple rounds of retrieval, using previous round results to guide the next retrieval.

## Use Cases

- Complex research questions (need to synthesize multiple sources)
- Cross-document fact-checking
- Multi-step troubleshooting
- Q&A requiring reasoning chains

## Architecture

```
Question → Decompose sub-questions → Retrieve sub-question 1 → Retrieve sub-question 2 (based on previous results) → ... → Synthesize answer
```

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MultiHopRAG:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        self.collection_name = "multi_hop_rag"
        self._init_collection()

    def _embed(self, texts: list) -> list:
        """Generate embeddings using OpenAI API"""
        if isinstance(texts, str):
            texts = [texts]
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 256})

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add_document(self, text: str, source: str = ""):
        chunks = self.splitter.split_text(text)
        embeddings = self._embed(chunks)
        data = [{"text": c, "source": source, "embedding": e} for c, e in zip(chunks, embeddings)]
        self.client.insert(collection_name=self.collection_name, data=data)

    def retrieve(self, query: str, top_k: int = 5):
        """Single retrieval"""
        embedding = self._embed(query)[0]
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=top_k,
            output_fields=["text", "source"]
        )
        return [{"text": hit["entity"]["text"], "source": hit["entity"]["source"]}
                for hit in results[0]]

    def decompose_question(self, question: str):
        """Decompose complex question into sub-questions"""
        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""Decompose the following complex question into 2-4 simple sub-questions for step-by-step retrieval.
Output format: one sub-question per line, no numbering.

Question: {question}

Sub-questions:"""
            }],
            temperature=0
        )
        sub_questions = response.choices[0].message.content.strip().split("\n")
        return [q.strip() for q in sub_questions if q.strip()]

    def generate_followup_query(self, original_question: str, context: str, step: int):
        """Generate follow-up retrieval query based on existing context"""
        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""Based on the original question and retrieved information, generate the next retrieval query.

Original question: {original_question}

Existing information:
{context}

Please generate a retrieval query to obtain missing information needed to answer the original question:"""
            }],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def multi_hop_retrieve(self, question: str, max_hops: int = 3):
        """Multi-hop retrieval"""
        all_contexts = []
        queries = [question]

        for hop in range(max_hops):
            # Current query
            current_query = queries[-1]

            # Retrieve
            results = self.retrieve(current_query, top_k=5)
            all_contexts.extend(results)

            # Check if we need to continue
            if hop < max_hops - 1:
                context_text = "\n".join([c["text"] for c in all_contexts])

                # Check if we have enough information
                check_response = self.openai.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{
                        "role": "user",
                        "content": f"""Determine if the existing information is sufficient to answer the question.

Question: {question}

Existing information:
{context_text}

Answer "sufficient" or "insufficient":"""
                    }],
                    temperature=0
                )

                if "sufficient" in check_response.choices[0].message.content.lower():
                    break

                # Generate follow-up query
                followup = self.generate_followup_query(question, context_text, hop + 1)
                queries.append(followup)

        return all_contexts, queries

    def generate_answer(self, question: str, contexts: list):
        """Generate final answer"""
        # Deduplicate
        seen = set()
        unique_contexts = []
        for c in contexts:
            if c["text"] not in seen:
                seen.add(c["text"])
                unique_contexts.append(c)

        context_text = "\n\n".join([
            f"[Source: {c['source']}]\n{c['text']}" for c in unique_contexts
        ])

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""Answer the question based on the following references. Please synthesize information from multiple sources for a complete answer.

References:
{context_text}

Question: {question}

Answer:"""
            }],
            temperature=0.3
        )
        return response.choices[0].message.content

    def query(self, question: str, max_hops: int = 3):
        """Complete multi-hop Q&A"""
        # Multi-hop retrieval
        contexts, queries = self.multi_hop_retrieve(question, max_hops)

        # Generate answer
        answer = self.generate_answer(question, contexts)

        return {
            "answer": answer,
            "hops": len(queries),
            "queries": queries,
            "sources": list(set(c["source"] for c in contexts))
        }

# Usage
rag = MultiHopRAG()

# Add documents
rag.add_document("John is the CEO of ABC Company, founded in 2020...", source="company_intro.md")
rag.add_document("ABC Company achieved $1 billion revenue in 2023...", source="financial_report.md")
rag.add_document("John graduated from MIT Computer Science department...", source="ceo_bio.md")

# Complex question (requires multi-hop reasoning)
result = rag.query("What is the educational background of ABC Company's CEO, and how has the company performed since he founded it?")
print(f"Answer: {result['answer']}")
print(f"Retrieval rounds: {result['hops']}")
print(f"Retrieval queries: {result['queries']}")
```

## Strategy Selection

| Strategy | Use Case | Implementation Complexity |
|----------|----------|--------------------------|
| Question decomposition | Clearly splittable questions | Low |
| Iterative retrieval | Questions requiring dynamic discovery | Medium |
| Hybrid strategy | Complex research questions | High |

## Parameter Recommendations

| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| max_hops | 2-4 | Maximum retrieval rounds |
| top_k_per_hop | 3-5 | Results per retrieval round |

## Vertical Applications

See `verticals/` directory for detailed guides:
- `research.md` - Complex research questions
- `fact-checking.md` - Fact verification
- `troubleshooting.md` - Troubleshooting

## Related Tools

- Basic RAG: `scenarios:rag`
- Reranking: `core:rerank`
- Document chunking: `core:chunking`
