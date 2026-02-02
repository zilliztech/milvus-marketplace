---
name: multimodal-rag
description: "Use when user needs RAG on documents with images and text. Triggers on: multimodal RAG, image-text mixed, document with images, PDF with charts, visual RAG, visual Q&A."
---

# Multimodal RAG

Handle Q&A on mixed documents containing images, tables, and charts.

## Use Cases

- Product manual Q&A (image-text mixed)
- Medical report analysis (images + text)
- Financial report analysis (charts + data)
- Technical documentation (with flowcharts/architecture diagrams)

## Architecture

```
Image-text document → Text extraction + Image description (VLM) → Unified vectorization → Store
Query → Retrieve text and images → Combine context → VLM/LLM generates answer
```

## Data Processing

Batch processing of image-text documents is recommended using Ray orchestration (see `core:ray`).

**Key Steps**:

1. **PDF parsing**: PyMuPDF extracts text + images
2. **Image description**: VLM (GPT-4o/Qwen-VL) generates descriptions
3. **Vectorization**: BGE encodes text and descriptions uniformly
4. **Write to Milvus**: Batch insert

**Tool Selection**:

| Step | Tool |
|------|------|
| PDF parsing | PyMuPDF (fitz) |
| Word/PPT | python-docx, python-pptx |
| Image description | GPT-4o / Qwen-VL / LLaVA |
| Vectorization | BGE-large-zh |

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("content_type", DataType.VARCHAR, max_length=16)  # text/image
schema.add_field("content", DataType.VARCHAR, max_length=65535)    # Text or image description
schema.add_field("image_path", DataType.VARCHAR, max_length=512)   # Image path
schema.add_field("source", DataType.VARCHAR, max_length=512)
schema.add_field("page", DataType.INT32)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

index_params.add_index("embedding", index_type="HNSW", metric_type="COSINE",
                       params={"M": 16, "efConstruction": 256})
```

## Q&A Implementation

```python
from pymilvus import MilvusClient
from openai import OpenAI
import base64
import os

class MultimodalRAG:
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

    def retrieve(self, query: str, limit: int = 10) -> list:
        """Retrieve (text + images)"""
        embedding = self._embed(query)

        results = self.client.search(
            collection_name="multimodal_rag",
            data=[embedding],
            limit=limit,
            output_fields=["content_type", "content", "image_path", "source", "page"]
        )

        return [{
            "type": hit["entity"]["content_type"],
            "content": hit["entity"]["content"],
            "image_path": hit["entity"]["image_path"],
            "source": hit["entity"]["source"],
            "page": hit["entity"]["page"],
            "score": hit["distance"]
        } for hit in results[0]]

    def query(self, question: str, use_vision: bool = True) -> dict:
        """Q&A"""
        contexts = self.retrieve(question, limit=10)

        text_contexts = [c for c in contexts if c["type"] == "text"]
        image_contexts = [c for c in contexts if c["type"] == "image"]

        # Build messages
        messages = [{"role": "user", "content": []}]

        context_text = "\n\n".join([
            f"[{c['source']} Page {c['page']}]\n{c['content']}"
            for c in text_contexts[:5]
        ])
        messages[0]["content"].append({"type": "text", "text": f"Reference text:\n{context_text}\n"})

        # Add images
        if use_vision and image_contexts:
            for img in image_contexts[:3]:
                if os.path.exists(img["image_path"]):
                    with open(img["image_path"], "rb") as f:
                        b64 = base64.standard_b64encode(f.read()).decode()
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })

        messages[0]["content"].append({"type": "text", "text": f"\nQuestion: {question}\nAnswer:"})

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            temperature=0.3
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": list(set(c["source"] for c in contexts))
        }

# Usage
rag = MultimodalRAG()
result = rag.query("What are the product installation steps?")
print(f"Answer: {result['answer']}")
```

## VLM Selection

| Model | Features | Use Case |
|-------|----------|----------|
| GPT-4o | Best, expensive | Complex charts |
| Claude 3 | Good quality | General |
| Qwen-VL | Good for Chinese | Chinese documents |
| LLaVA | Open-source | Local deployment |

## Vertical Applications

See `verticals/` directory for detailed guides:
- `product-manual.md` - Product manuals
- `medical-report.md` - Medical reports
- `financial-report.md` - Financial report analysis

## Related Tools

- Data processing orchestration: `core:ray`
- Vectorization: `core:embedding`
- Basic RAG: `scenarios:rag`
