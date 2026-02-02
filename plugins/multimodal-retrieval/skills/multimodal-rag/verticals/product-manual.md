# Product Manual Multimodal QA

## Use Cases

- Electronics user manual QA
- Equipment operation manual queries
- Automotive user manuals
- Appliance usage guides

## Data Characteristics

- Mixed text and images (operation steps with illustrations)
- Structured content (chapters, steps)
- Technical parameter tables
- Troubleshooting flowcharts

## Recommended Configuration

| Config | Recommended Value | Description |
|--------|------------------|-------------|
| Text Embedding | `BAAI/bge-large-en-v1.5` | English |
| VLM | GPT-4o | Best for charts |
| | Qwen-VL-Max | Chinese charts |
| PDF Parsing | PyMuPDF + pdfplumber | Table extraction |
| LLM | GPT-4o | Supports multimodal input |

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("content", DataType.VARCHAR, max_length=65535)      # Text or image description
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

# Content type
schema.add_field("content_type", DataType.VARCHAR, max_length=16)   # text/image/table
schema.add_field("image_path", DataType.VARCHAR, max_length=512)    # Image path

# Product information
schema.add_field("product", DataType.VARCHAR, max_length=128)       # Product name
schema.add_field("model", DataType.VARCHAR, max_length=64)          # Model
schema.add_field("manual_version", DataType.VARCHAR, max_length=32) # Manual version

# Structure information
schema.add_field("chapter", DataType.VARCHAR, max_length=128)       # Chapter
schema.add_field("section", DataType.VARCHAR, max_length=128)       # Section
schema.add_field("page", DataType.INT32)
```

## Implementation

```python
from pymilvus import MilvusClient, DataType
from openai import OpenAI
import fitz  # PyMuPDF
import pdfplumber
import os
import base64

class ProductManualRAG:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()
        self._init_collection()

    def _embed(self, text: str) -> list:
        """Generate embedding using OpenAI API"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

    def _describe_image(self, image_path: str, context: str = "") -> str:
        """Describe image with VLM"""
        with open(image_path, "rb") as f:
            base64_image = base64.standard_b64encode(f.read()).decode()

        prompt = "Please describe this image in detail."
        if context:
            prompt += f" This is an image from a product manual, context: {context}"
        prompt += " If it's an operation step diagram, please describe the steps. If it's a parts diagram, please describe each part."

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }],
            max_tokens=500
        )
        return response.choices[0].message.content

    def _extract_tables(self, pdf_path: str, page_num: int) -> list:
        """Extract tables"""
        tables = []
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < len(pdf.pages):
                page = pdf.pages[page_num]
                for table in page.extract_tables():
                    # Convert to markdown format
                    if table and table[0]:
                        headers = table[0]
                        rows = table[1:]
                        md = "| " + " | ".join(str(h) for h in headers) + " |\n"
                        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                        for row in rows:
                            md += "| " + " | ".join(str(c) for c in row) + " |\n"
                        tables.append(md)
        return tables

    def add_manual(self, pdf_path: str, product: str, model: str,
                   output_dir: str = "./manual_data"):
        """Process product manual"""
        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        data = []

        current_chapter = ""
        current_section = ""

        for page_num, page in enumerate(doc):
            # 1. Extract text
            text = page.get_text()

            # Simple chapter detection (adjust based on actual format)
            lines = text.split('\n')
            for line in lines:
                if line.startswith('Chapter') or 'CHAPTER' in line.upper():
                    current_chapter = line.strip()
                elif line.startswith('Section') or 'SECTION' in line.upper():
                    current_section = line.strip()

            # Store text in chunks
            if text.strip():
                chunks = self._split_text(text, chunk_size=500)
                for chunk in chunks:
                    data.append({
                        "id": f"{product}_{model}_{page_num}_text_{len(data)}",
                        "content": chunk,
                        "embedding": self._embed(chunk).tolist(),
                        "content_type": "text",
                        "image_path": "",
                        "product": product,
                        "model": model,
                        "manual_version": "1.0",
                        "chapter": current_chapter,
                        "section": current_section,
                        "page": page_num + 1
                    })

            # 2. Extract tables
            tables = self._extract_tables(pdf_path, page_num)
            for i, table in enumerate(tables):
                data.append({
                    "id": f"{product}_{model}_{page_num}_table_{i}",
                    "content": table,
                    "embedding": self._embed(table).tolist(),
                    "content_type": "table",
                    "image_path": "",
                    "product": product,
                    "model": model,
                    "manual_version": "1.0",
                    "chapter": current_chapter,
                    "section": current_section,
                    "page": page_num + 1
                })

            # 3. Extract images
            images = page.get_images()
            for img_idx, img in enumerate(images):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                # Save image
                img_path = os.path.join(output_dir, f"{product}_{model}_p{page_num+1}_img{img_idx+1}.png")
                if pix.n < 5:
                    pix.save(img_path)
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(img_path)

                # Get surrounding text as context
                context = text[:200] if text else ""

                # Describe image
                try:
                    description = self._describe_image(img_path, context)
                    data.append({
                        "id": f"{product}_{model}_{page_num}_img_{img_idx}",
                        "content": description,
                        "embedding": self._embed(description).tolist(),
                        "content_type": "image",
                        "image_path": img_path,
                        "product": product,
                        "model": model,
                        "manual_version": "1.0",
                        "chapter": current_chapter,
                        "section": current_section,
                        "page": page_num + 1
                    })
                except Exception as e:
                    print(f"Image processing failed: {img_path}, {e}")

        # Batch insert
        if data:
            self.client.insert(collection_name="product_manuals", data=data)

        return {
            "product": product,
            "model": model,
            "pages": len(doc),
            "text_chunks": len([d for d in data if d["content_type"] == "text"]),
            "tables": len([d for d in data if d["content_type"] == "table"]),
            "images": len([d for d in data if d["content_type"] == "image"])
        }

    def query(self, question: str, product: str = None, model: str = None) -> dict:
        """QA"""
        embedding = self._embed(question).tolist()

        # Build filter conditions
        filters = []
        if product:
            filters.append(f'product == "{product}"')
        if model:
            filters.append(f'model == "{model}"')

        filter_expr = ' and '.join(filters) if filters else ""

        # Retrieve
        results = self.client.search(
            collection_name="product_manuals",
            data=[embedding],
            filter=filter_expr,
            limit=10,
            output_fields=["content", "content_type", "image_path", "chapter", "section", "page"]
        )

        # Organize context
        text_contexts = []
        image_contexts = []
        table_contexts = []

        for r in results[0]:
            ctx = {
                "content": r["entity"]["content"],
                "chapter": r["entity"]["chapter"],
                "page": r["entity"]["page"],
                "score": r["distance"]
            }

            if r["entity"]["content_type"] == "text":
                text_contexts.append(ctx)
            elif r["entity"]["content_type"] == "image":
                ctx["image_path"] = r["entity"]["image_path"]
                image_contexts.append(ctx)
            elif r["entity"]["content_type"] == "table":
                table_contexts.append(ctx)

        # Generate answer
        return self._generate_answer(question, text_contexts, image_contexts, table_contexts)

    def _generate_answer(self, question: str, texts: list, images: list, tables: list) -> dict:
        """Generate answer"""
        messages = [{"role": "user", "content": []}]

        # Text context
        if texts:
            text_content = "\n\n".join([
                f"[{t['chapter']} Page {t['page']}]\n{t['content']}"
                for t in texts[:5]
            ])
            messages[0]["content"].append({
                "type": "text",
                "text": f"Reference text:\n{text_content}\n"
            })

        # Table context
        if tables:
            table_content = "\n\n".join([
                f"[Page {t['page']} Table]\n{t['content']}"
                for t in tables[:3]
            ])
            messages[0]["content"].append({
                "type": "text",
                "text": f"Reference tables:\n{table_content}\n"
            })

        # Image context
        if images:
            messages[0]["content"].append({
                "type": "text",
                "text": "Reference images:"
            })
            for img in images[:3]:
                if os.path.exists(img["image_path"]):
                    with open(img["image_path"], "rb") as f:
                        base64_img = base64.standard_b64encode(f.read()).decode()
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                    })
                    messages[0]["content"].append({
                        "type": "text",
                        "text": f"(Page {img['page']}, Image description: {img['content'][:100]}...)"
                    })

        messages[0]["content"].append({
            "type": "text",
            "text": f"\nQuestion: {question}\n\nPlease answer based on the references above. If it involves operation steps, list detailed steps."
        })

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            temperature=0.3
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": {
                "text_refs": [{"page": t["page"], "chapter": t["chapter"]} for t in texts],
                "image_refs": [{"page": i["page"], "path": i["image_path"]} for i in images],
                "table_refs": [{"page": t["page"]} for t in tables]
            }
        }
```

## Examples

```python
rag = ProductManualRAG()

# Import product manual
stats = rag.add_manual(
    pdf_path="iphone_user_guide.pdf",
    product="iPhone",
    model="iPhone 15 Pro"
)
print(f"Import complete: {stats}")

# QA
result = rag.query(
    "How do I set up Face ID?",
    product="iPhone",
    model="iPhone 15 Pro"
)

print(f"Answer: {result['answer']}")
print(f"\nReference sources:")
for ref in result['sources']['text_refs']:
    print(f"  - {ref['chapter']} Page {ref['page']}")
```

## Special Processing

### Operation Step Recognition

```python
def extract_steps(text: str) -> list:
    """Extract operation steps"""
    import re

    # Match step patterns
    patterns = [
        r'(\d+)\.\s*(.+?)(?=\d+\.|$)',           # 1. Step
        r'Step\s*(\d+)[：:]\s*(.+?)(?=Step|$)'   # Step 1: Step
    ]

    steps = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            steps = [{"step": int(m[0]), "content": m[1].strip()} for m in matches]
            break

    return steps
```
