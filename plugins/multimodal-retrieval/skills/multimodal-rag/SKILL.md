---
name: multimodal-rag
description: "Use when user needs RAG on documents with images and text. Triggers on: multimodal RAG, image-text mixed, document with images, PDF with charts, visual RAG, visual Q&A, documents with figures."
---

# Multimodal RAG

Handle Q&A on documents containing images, tables, charts, and text — answer questions that require understanding both visual and textual content.

## When to Activate

Activate this skill when:
- User has **documents with images** (PDFs with charts, manuals with diagrams)
- User needs to **answer questions** about visual content in documents
- User mentions "PDF with images", "document with charts", "visual Q&A"
- User's documents have **tables, flowcharts, or screenshots**

**Do NOT activate** when:
- User only needs image similarity search → use `image-search`
- User only needs text-based RAG → use `rag-toolkit:rag`
- User needs video search → use `video-search`

## Interactive Flow

### Step 1: Understand Document Type

"What type of documents are you processing?"

A) **Technical manuals** (product docs, installation guides)
   - Mix of text and diagrams
   - Questions like "How do I install component X?"

B) **Reports with charts** (financial, research, analytics)
   - Data visualizations, tables
   - Questions like "What was Q3 revenue?"

C) **Mixed content** (presentations, marketing materials)
   - Varied image types
   - Diverse question types

Which describes your documents? (A/B/C)

### Step 2: Choose Processing Strategy

"How should we handle images?"

| Strategy | When to Use |
|----------|-------------|
| **VLM Description** | Charts, diagrams, complex visuals |
| **OCR Extraction** | Screenshots with text, tables |
| **Caption Only** | Photos, simple images |

For most cases, **VLM Description** is recommended.

### Step 3: Confirm Configuration

"Based on your requirements:

- **Text extraction**: PyMuPDF
- **Image processing**: GPT-4o for descriptions
- **Embedding**: text-embedding-3-small (1536 dim)
- **Answer model**: GPT-4o with vision

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Illustrated Encyclopedia

Think of multimodal RAG as building a **searchable illustrated encyclopedia**:

1. **Extract all content**: Text paragraphs AND image descriptions
2. **Index everything**: Both become searchable vectors
3. **Retrieve mixed results**: May get text AND images for a query
4. **Generate answer**: VLM combines all context to answer

```
┌─────────────────────────────────────────────────────────────┐
│                    Multimodal RAG Pipeline                   │
│                                                              │
│  Document (PDF with images)                                  │
│       │                                                      │
│       ├──────────────────────────────────────┐              │
│       │                                      │              │
│       ▼                                      ▼              │
│  ┌──────────────┐                    ┌──────────────┐      │
│  │ Text Chunks  │                    │   Images     │      │
│  │              │                    │              │      │
│  │ "Section 1   │                    │ [diagram.png]│      │
│  │  describes..." │                  │ [chart.png]  │      │
│  └──────┬───────┘                    └──────┬───────┘      │
│         │                                   │               │
│         │                                   ▼               │
│         │                           ┌──────────────┐       │
│         │                           │  VLM Caption │       │
│         │                           │  "This chart │       │
│         │                           │   shows..."  │       │
│         │                           └──────┬───────┘       │
│         │                                  │               │
│         ▼                                  ▼               │
│  ┌──────────────────────────────────────────────┐        │
│  │            Unified Text Embeddings            │        │
│  │     (both text chunks and image captions)     │        │
│  └──────────────────────┬───────────────────────┘        │
│                         │                                 │
│                         ▼                                 │
│                  ┌──────────────┐                        │
│                  │    Milvus    │                        │
│                  └──────┬───────┘                        │
│                         │                                 │
│                         ▼                                 │
│  Query: "What does the chart show?"                       │
│                         │                                 │
│                         ▼                                 │
│  Retrieved: [text chunk] + [image caption + image]        │
│                         │                                 │
│                         ▼                                 │
│  ┌──────────────────────────────────────────────┐        │
│  │    VLM Answer Generation (with images)        │        │
│  │    "Based on the chart, revenue increased..." │        │
│  └──────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Why Multimodal Matters

| Question Type | Text-Only RAG | Multimodal RAG |
|---------------|---------------|----------------|
| "What does paragraph 3 say?" | ✅ Works | ✅ Works |
| "What does the chart show?" | ❌ Can't see | ✅ Understands |
| "Summarize the diagram" | ❌ Can't see | ✅ Describes |
| "What's the value in the table?" | ⚠️ If OCR'd | ✅ Reads directly |

## Implementation

```python
from pymilvus import MilvusClient, DataType
from openai import OpenAI
import fitz  # PyMuPDF
import base64
import os
import uuid

class MultimodalRAG:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()
        self.collection_name = "multimodal_rag"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("content_type", DataType.VARCHAR, max_length=16)  # text/image
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("image_path", DataType.VARCHAR, max_length=512)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("page", DataType.INT32)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

        index_params = self.client.prepare_index_params()
        index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def _embed(self, text: str) -> list:
        """Generate embedding using OpenAI API."""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

    def _describe_image(self, image_path: str) -> str:
        """Generate description of image using VLM."""
        with open(image_path, "rb") as f:
            b64_image = base64.standard_b64encode(f.read()).decode()

        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail. If it's a chart or diagram, explain what it shows. If there's text, include it."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }],
            max_tokens=1000
        )
        return response.choices[0].message.content

    def ingest_pdf(self, pdf_path: str, image_output_dir: str = "./images"):
        """Process PDF and index text + images."""
        os.makedirs(image_output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        source = os.path.basename(pdf_path)
        data = []

        for page_num, page in enumerate(doc):
            # Extract text
            text = page.get_text()
            if text.strip():
                chunks = self._chunk_text(text)
                for chunk in chunks:
                    data.append({
                        "id": str(uuid.uuid4()),
                        "content_type": "text",
                        "content": chunk,
                        "image_path": "",
                        "source": source,
                        "page": page_num + 1,
                        "embedding": self._embed(chunk)
                    })

            # Extract images
            images = page.get_images()
            for img_idx, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                image_path = f"{image_output_dir}/{source}_p{page_num+1}_img{img_idx}.png"
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # Generate description
                description = self._describe_image(image_path)

                data.append({
                    "id": str(uuid.uuid4()),
                    "content_type": "image",
                    "content": description,
                    "image_path": image_path,
                    "source": source,
                    "page": page_num + 1,
                    "embedding": self._embed(description)
                })

        self.client.insert(collection_name=self.collection_name, data=data)
        return len(data)

    def _chunk_text(self, text: str, chunk_size: int = 500) -> list:
        """Split text into chunks."""
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def retrieve(self, query: str, limit: int = 10) -> list:
        """Retrieve relevant text and images."""
        embedding = self._embed(query)

        results = self.client.search(
            collection_name=self.collection_name,
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

    def query(self, question: str, use_images: bool = True) -> dict:
        """Answer question using retrieved context."""
        contexts = self.retrieve(question, limit=10)

        text_contexts = [c for c in contexts if c["type"] == "text"]
        image_contexts = [c for c in contexts if c["type"] == "image"]

        # Build message with text and images
        messages = [{"role": "user", "content": []}]

        # Add text context
        context_text = "\n\n".join([
            f"[{c['source']} Page {c['page']}]\n{c['content']}"
            for c in text_contexts[:5]
        ])
        messages[0]["content"].append({
            "type": "text",
            "text": f"Context:\n{context_text}\n"
        })

        # Add images
        if use_images and image_contexts:
            for img in image_contexts[:3]:
                if os.path.exists(img["image_path"]):
                    with open(img["image_path"], "rb") as f:
                        b64 = base64.standard_b64encode(f.read()).decode()
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })

        # Add question
        messages[0]["content"].append({
            "type": "text",
            "text": f"\nQuestion: {question}\nAnswer based on the provided context and images:"
        })

        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": list(set(c["source"] for c in contexts)),
            "pages": list(set(c["page"] for c in contexts))
        }

# Usage
rag = MultimodalRAG()

# Ingest PDF with images
rag.ingest_pdf("product_manual.pdf")

# Ask questions
result = rag.query("What are the installation steps shown in the diagram?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

## Processing Strategy by Document Type

| Document Type | Text Extraction | Image Processing | VLM Prompt |
|---------------|-----------------|------------------|------------|
| **Technical manual** | Full text | Diagram description | "Explain what this diagram shows step by step" |
| **Financial report** | Full text | Chart data extraction | "Extract all data points from this chart" |
| **Medical report** | Full text | Image analysis | "Describe any medical findings visible" |
| **Presentation** | Slide text | Screenshot description | "Summarize what this slide conveys" |

## Common Pitfalls

### ❌ Pitfall 1: Images Too Small

**Problem**: VLM can't read chart text

**Why**: PDF images extracted at low resolution

**Fix**: Extract at higher resolution
```python
mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
pix = page.get_pixmap(matrix=mat)
```

### ❌ Pitfall 2: Ignoring Image-Text Connection

**Problem**: Image description doesn't mention surrounding context

**Why**: Image processed in isolation

**Fix**: Include nearby text in prompt
```python
prompt = f"This image appears near the text: '{nearby_text}'. Describe the image and how it relates to this text."
```

### ❌ Pitfall 3: Too Many API Calls

**Problem**: Processing costs explode

**Why**: Calling VLM for every image

**Fix**: Batch processing, caching, or use cheaper models
```python
# Use gpt-4o-mini for initial pass
# Only use gpt-4o for complex charts
```

### ❌ Pitfall 4: Not Returning Images in Answer

**Problem**: User asks about chart but can't see it

**Why**: Only returning text answer

**Fix**: Include image references in response
```python
return {
    "answer": answer,
    "referenced_images": [c["image_path"] for c in image_contexts[:3]]
}
```

## When to Level Up

| Need | Upgrade To |
|------|------------|
| Better chunking | Add `core:chunking` |
| Higher precision | Add `core:rerank` |
| Video content | `video-search` |
| Pure text documents | `rag-toolkit:rag` |

## References

- VLM models: GPT-4o, Claude 3, Qwen-VL, LLaVA
- PDF processing: PyMuPDF, pdf2image
- Batch processing: `core:ray`
