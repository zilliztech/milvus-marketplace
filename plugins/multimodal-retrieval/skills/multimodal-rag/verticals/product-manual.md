# Product Manual Multimodal RAG

> Answer questions about product manuals using text, images, and tables.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Manual Language

<ask_user>
What language are your product manuals in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** | English models + GPT-4o |
| **Chinese** | Chinese models + Qwen-VL |
| **Multilingual** | Multilingual models |
</ask_user>

### 2. Text Embedding

<ask_user>
Choose text embedding:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download |

Local options:
- `BAAI/bge-base-en-v1.5` (768d, English)
- `BAAI/bge-base-zh-v1.5` (768d, Chinese)
</ask_user>

### 3. Vision Language Model

<ask_user>
Choose VLM for image understanding:

| Model | Notes |
|-------|-------|
| **GPT-4o** | Best quality for charts/diagrams |
| **GPT-4o-mini** | Cost-effective |
| **Qwen-VL-Max** | Good for Chinese content |
</ask_user>

### 4. Data Scale

<ask_user>
How many manuals do you have?

- Each manual ≈ 50-200 chunks (text + images + tables)
- Example: 100 manuals ≈ 10K-20K vectors

| Vector Count | Recommended Milvus |
|--------------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 5. Project Setup

<ask_user>
| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production |
| **pip** | Quick prototype |
</ask_user>

---

## Dependencies

```bash
uv init product-manual-rag
cd product-manual-rag
uv add pymilvus openai pymupdf pdfplumber Pillow
```

---

## End-to-End Implementation

### Step 1: Configure Models

```python
from openai import OpenAI
import base64

client = OpenAI()

def embed_text(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

def describe_image(image_path: str, context: str = "") -> str:
    """Describe image using VLM."""
    with open(image_path, "rb") as f:
        base64_image = base64.standard_b64encode(f.read()).decode()

    prompt = "Describe this image in detail."
    if context:
        prompt += f" Context: {context}"
    prompt += " If it's an operation diagram, describe the steps."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
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

def generate_answer(question: str, contexts: list[dict]) -> str:
    """Generate answer from multimodal contexts."""
    messages = [{"role": "user", "content": []}]

    # Add text contexts
    text_parts = [c for c in contexts if c["type"] == "text"]
    if text_parts:
        text_content = "\n\n".join([c["content"][:500] for c in text_parts[:5]])
        messages[0]["content"].append({
            "type": "text",
            "text": f"Reference text:\n{text_content}\n"
        })

    # Add image contexts
    image_parts = [c for c in contexts if c["type"] == "image"]
    for img in image_parts[:3]:
        if img.get("image_path"):
            with open(img["image_path"], "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode()
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    messages[0]["content"].append({
        "type": "text",
        "text": f"\nQuestion: {question}\n\nAnswer based on the references above."
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("product_manuals.db")

schema = milvus.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("content_type", DataType.VARCHAR, max_length=16)  # text/image/table
schema.add_field("image_path", DataType.VARCHAR, max_length=512)
schema.add_field("product", DataType.VARCHAR, max_length=128)
schema.add_field("model", DataType.VARCHAR, max_length=64)
schema.add_field("chapter", DataType.VARCHAR, max_length=128)
schema.add_field("page", DataType.INT32)

index_params = milvus.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

milvus.create_collection("product_manuals", schema=schema, index_params=index_params)
```

### Step 3: Process & Index Manual

```python
import fitz  # PyMuPDF
import pdfplumber
import os

def process_manual(pdf_path: str, product: str, model_name: str, output_dir: str = "./images"):
    """Process and index a product manual."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    data = []

    current_chapter = ""

    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text()

        # Detect chapter
        for line in text.split('\n'):
            if 'chapter' in line.lower() or 'section' in line.lower():
                current_chapter = line.strip()[:100]
                break

        # Chunk and index text
        if text.strip():
            chunks = [text[i:i+500] for i in range(0, len(text), 450)]
            for chunk in chunks:
                if len(chunk) > 50:
                    embedding = embed_text([chunk])[0]
                    data.append({
                        "embedding": embedding,
                        "content": chunk,
                        "content_type": "text",
                        "image_path": "",
                        "product": product,
                        "model": model_name,
                        "chapter": current_chapter,
                        "page": page_num + 1
                    })

        # Extract and index images
        images = page.get_images()
        for img_idx, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            img_path = os.path.join(output_dir, f"{product}_{page_num+1}_{img_idx+1}.png")
            if pix.n < 5:
                pix.save(img_path)
            else:
                pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(img_path)

            # Describe image
            context = text[:200] if text else ""
            try:
                description = describe_image(img_path, context)
                embedding = embed_text([description])[0]
                data.append({
                    "embedding": embedding,
                    "content": description,
                    "content_type": "image",
                    "image_path": img_path,
                    "product": product,
                    "model": model_name,
                    "chapter": current_chapter,
                    "page": page_num + 1
                })
            except Exception as e:
                print(f"Image processing failed: {e}")

    # Extract tables
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            for table in page.extract_tables():
                if table and table[0]:
                    # Convert to markdown
                    headers = table[0]
                    md = "| " + " | ".join(str(h) for h in headers) + " |\n"
                    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    for row in table[1:]:
                        md += "| " + " | ".join(str(c) for c in row) + " |\n"

                    embedding = embed_text([md])[0]
                    data.append({
                        "embedding": embedding,
                        "content": md,
                        "content_type": "table",
                        "image_path": "",
                        "product": product,
                        "model": model_name,
                        "chapter": "",
                        "page": page_num + 1
                    })

    if data:
        milvus.insert(collection_name="product_manuals", data=data)

    return {
        "text_chunks": len([d for d in data if d["content_type"] == "text"]),
        "images": len([d for d in data if d["content_type"] == "image"]),
        "tables": len([d for d in data if d["content_type"] == "table"])
    }
```

### Step 4: Query

```python
def query_manual(question: str, product: str = None, model_name: str = None):
    """Answer question using multimodal RAG."""
    embedding = embed_text([question])[0]

    filters = []
    if product:
        filters.append(f'product == "{product}"')
    if model_name:
        filters.append(f'model == "{model_name}"')

    filter_expr = ' and '.join(filters) if filters else None

    results = milvus.search(
        collection_name="product_manuals",
        data=[embedding],
        filter=filter_expr,
        limit=10,
        output_fields=["content", "content_type", "image_path", "chapter", "page"]
    )

    contexts = [{
        "type": r["entity"]["content_type"],
        "content": r["entity"]["content"],
        "image_path": r["entity"]["image_path"],
        "page": r["entity"]["page"]
    } for r in results[0]]

    answer = generate_answer(question, contexts)

    return {
        "answer": answer,
        "sources": [{"page": c["page"], "type": c["type"]} for c in contexts]
    }
```

---

## Run Example

```python
# Process manual
stats = process_manual(
    pdf_path="iphone_user_guide.pdf",
    product="iPhone",
    model_name="iPhone 15 Pro"
)
print(f"Indexed: {stats}")

# Query
result = query_manual(
    "How do I set up Face ID?",
    product="iPhone",
    model_name="iPhone 15 Pro"
)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```
