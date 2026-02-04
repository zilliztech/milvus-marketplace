# SEC EDGAR Financial Report Search

> Search SEC filings (10-K, 10-Q, 8-K) by company, topic, or financial concept.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Filing Types

<ask_user>
Which SEC filings do you want to search?

| Filing Type | Description | Length |
|-------------|-------------|--------|
| **10-K** | Annual report | Very long (~200-500 pages) |
| **10-Q** | Quarterly report | Long (~50-100 pages) |
| **8-K** | Current events | Short (~5-20 pages) |
| **All types** | Mixed filings | Variable |
</ask_user>

### 2. Filing Format

<ask_user>
Which format do you prefer to process?

| Format | Pros | Cons |
|--------|------|------|
| **HTML** (recommended) | Structured, easier to parse | Requires BeautifulSoup |
| **PDF** | Original layout | Harder to extract clean text |
</ask_user>

### 3. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality, fast | Requires API key, costs money |
| **Local Model** | Free, offline | Model download required |

Do you have an OpenAI API Key?
</ask_user>

### 4. Local Model Selection (if local)

<ask_user>
Choose an embedding model (English financial text):

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast, good for prototyping |
| `BAAI/bge-base-en-v1.5` (recommended) | 768 | 440MB | Higher quality |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3GB | Best quality, GPU recommended |
</ask_user>

### 5. Data Scale

<ask_user>
How many filings do you plan to index?

- 10-K filings: ~200-500 chunks each
- 10-Q filings: ~50-100 chunks each
- Example: 50 companies × 5 years × 4 quarters = ~1000 filings ≈ 50K-100K vectors

| Vector Count | Recommended Milvus |
|--------------|-------------------|
| < 100K | **Milvus Lite** |
| 100K - 10M | **Milvus Standalone** |
| > 10M | **Zilliz Cloud** |
</ask_user>

### 6. Project Setup

<ask_user>
Choose project management:

| Method | Best For |
|--------|----------|
| **uv + pyproject.toml** (recommended) | Production projects |
| **pip** | Quick prototypes |
</ask_user>

---

## Dependencies

### OpenAI + uv
```bash
uv init sec-search
cd sec-search
uv add pymilvus openai sec-edgar-downloader beautifulsoup4 lxml
```

### Local Model + uv
```bash
uv init sec-search
cd sec-search
uv add pymilvus sentence-transformers sec-edgar-downloader beautifulsoup4 lxml
```

---

## End-to-End Implementation

### Step 1: Configure Embedding

```python
# === Choose ONE ===

# Option A: OpenAI API
from openai import OpenAI
client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

# Option B: Local Model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()

DIMENSION = 768
```

### Step 2: Download SEC Filings

```python
from sec_edgar_downloader import Downloader
import os

def download_filings(
    tickers: list[str],
    filing_type: str = "10-K",
    num_filings: int = 5,
    output_dir: str = "sec_filings"
):
    """Download SEC filings for given tickers."""
    dl = Downloader("MyCompany", "email@example.com", output_dir)

    for ticker in tickers:
        try:
            dl.get(filing_type, ticker, limit=num_filings)
            print(f"Downloaded {filing_type} for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    return output_dir

# Download 10-K filings for major tech companies
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
download_filings(tickers, filing_type="10-K", num_filings=3)
```

### Step 3: Parse HTML Filings

```python
from bs4 import BeautifulSoup
from pathlib import Path
import re

def parse_filing_html(file_path: str) -> dict:
    """Parse SEC filing HTML and extract text."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    # Remove scripts and styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Extract text
    text = soup.get_text(separator="\n")

    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    # Try to extract metadata from path
    path_parts = Path(file_path).parts
    ticker = path_parts[-4] if len(path_parts) >= 4 else "unknown"
    filing_type = path_parts[-3] if len(path_parts) >= 3 else "unknown"

    return {
        "ticker": ticker,
        "filing_type": filing_type,
        "text": text.strip(),
        "path": file_path
    }

def load_all_filings(base_dir: str) -> list[dict]:
    """Load all downloaded filings."""
    filings = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".html") or file.endswith(".htm"):
                file_path = os.path.join(root, file)
                try:
                    filing = parse_filing_html(file_path)
                    if len(filing["text"]) > 1000:  # Skip very short files
                        filings.append(filing)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

    return filings

filings = load_all_filings("sec_filings")
print(f"Loaded {len(filings)} filings")
```

### Step 4: Chunk Filings

```python
def chunk_filing(filing: dict, chunk_size: int = 1500, overlap: int = 200) -> list[dict]:
    """Split filing into chunks."""
    text = filing["text"]
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Try to break at paragraph boundary
        if end < len(text):
            last_para = chunk_text.rfind("\n\n")
            if last_para > chunk_size // 2:
                chunk_text = chunk_text[:last_para]
                end = start + last_para

        chunks.append({
            "ticker": filing["ticker"],
            "filing_type": filing["filing_type"],
            "chunk_index": len(chunks),
            "text": chunk_text.strip(),
            "path": filing["path"]
        })

        start = end - overlap

    return chunks

def process_all_filings(filings: list[dict]) -> list[dict]:
    """Process all filings into chunks."""
    all_chunks = []
    for filing in filings:
        chunks = chunk_filing(filing)
        all_chunks.extend(chunks)
    return all_chunks

chunks = process_all_filings(filings)
print(f"Created {len(chunks)} chunks")
```

### Step 5: Index into Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient("sec_filings.db")  # Milvus Lite
# client = MilvusClient(uri="http://localhost:19530")  # Standalone

client.create_collection(
    collection_name="sec_filings",
    dimension=DIMENSION,
    auto_id=True
)

def index_chunks(chunks: list[dict], batch_size: int = 50):
    """Embed and index filing chunks."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        vectors = embed(texts)

        data = [
            {
                "vector": vec,
                "ticker": c["ticker"],
                "filing_type": c["filing_type"],
                "chunk_index": c["chunk_index"],
                "text": c["text"][:1000]
            }
            for vec, c in zip(vectors, batch)
        ]
        client.insert(collection_name="sec_filings", data=data)
        print(f"Indexed {i + len(batch)}/{len(chunks)}")

index_chunks(chunks)
```

### Step 6: Search

```python
def search_filings(query: str, top_k: int = 5, ticker: str = None):
    """Search SEC filings."""
    query_vector = embed([query])[0]

    filter_expr = f'ticker == "{ticker}"' if ticker else None

    results = client.search(
        collection_name="sec_filings",
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=["ticker", "filing_type", "text"]
    )
    return results[0]

def print_results(results):
    for i, hit in enumerate(results, 1):
        entity = hit["entity"]
        print(f"\n{'='*60}")
        print(f"#{i} [{entity['ticker']}] {entity['filing_type']} (score: {hit['distance']:.3f})")
        print(f"\n{entity['text'][:400]}...")
```

---

## Run Example

```python
# Download and index
tickers = ["AAPL", "GOOGL", "MSFT"]
download_filings(tickers, "10-K", num_filings=2)
filings = load_all_filings("sec_filings")
chunks = process_all_filings(filings)
index_chunks(chunks)

# Search examples
print_results(search_filings("revenue growth from cloud services"))
print_results(search_filings("risk factors related to AI"))
print_results(search_filings("executive compensation", ticker="AAPL"))
```

---

## Advanced: Section-Aware Chunking

```python
# Common 10-K sections
SECTIONS = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 7": "MD&A",
    "Item 8": "Financial Statements"
}

def extract_sections(text: str) -> dict[str, str]:
    """Extract major sections from 10-K filing."""
    sections = {}
    for item, name in SECTIONS.items():
        pattern = rf'{item}[\.\s]+{name}(.*?)(?=Item \d|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[item] = match.group(1).strip()
    return sections
```
