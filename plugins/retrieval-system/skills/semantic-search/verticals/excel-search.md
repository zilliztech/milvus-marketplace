# Excel / Spreadsheet Search

> Search Excel rows and tables by natural language queries.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Data Language

<ask_user>
What language is your spreadsheet data in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** | English models |
| **Chinese** | Chinese models |
| **Mixed** | Multilingual models |
</ask_user>

### 2. Search Granularity

<ask_user>
How do you want to search?

| Granularity | Description |
|-------------|-------------|
| **Row-level** (recommended) | Each row becomes a searchable unit |
| **Cell-level** | Individual cells are searchable |
| **Sheet-level** | Entire sheets summarized |
</ask_user>

### 3. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality, fast | Requires API key |
| **Local Model** | Free, offline | Model download needed |
</ask_user>

### 4. Local Model (if local)

<ask_user>
Choose embedding model:

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | English |
| `BAAI/bge-base-zh-v1.5` | 768 | 400MB | Chinese |
</ask_user>

### 5. Data Scale

<ask_user>
How much data do you have?

- Each row = 1 vector
- Example: 100 Excel files × 1000 rows = 100K vectors

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
uv init excel-search
cd excel-search
uv add pymilvus openai openpyxl pandas
```

### Local Model + uv
```bash
uv init excel-search
cd excel-search
uv add pymilvus sentence-transformers openpyxl pandas
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
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()

DIMENSION = 384
```

### Step 2: Load Excel Files

```python
import pandas as pd
import os
from pathlib import Path

def load_excel(file_path: str) -> list[dict]:
    """Load all sheets from an Excel file."""
    sheets = []

    xl = pd.ExcelFile(file_path)
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet_name)

        # Skip empty sheets
        if df.empty:
            continue

        sheets.append({
            "file": os.path.basename(file_path),
            "sheet": sheet_name,
            "headers": list(df.columns),
            "dataframe": df
        })

    return sheets

def load_all_excels(folder: str) -> list[dict]:
    """Load all Excel files from a folder."""
    all_sheets = []

    for file in os.listdir(folder):
        if file.endswith((".xlsx", ".xls")):
            file_path = os.path.join(folder, file)
            try:
                sheets = load_excel(file_path)
                all_sheets.extend(sheets)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return all_sheets

sheets = load_all_excels("./excel_files")
print(f"Loaded {len(sheets)} sheets")
```

### Step 3: Convert Rows to Text

```python
def row_to_text(headers: list, row: pd.Series) -> str:
    """Convert a row to natural language text."""
    parts = []
    for header, value in zip(headers, row):
        if pd.notna(value) and str(value).strip():
            parts.append(f"{header}: {value}")
    return "; ".join(parts)

def process_sheets(sheets: list[dict]) -> list[dict]:
    """Convert all sheets to searchable rows."""
    all_rows = []

    for sheet in sheets:
        df = sheet["dataframe"]
        headers = sheet["headers"]

        for idx, row in df.iterrows():
            text = row_to_text(headers, row)
            if text.strip():
                all_rows.append({
                    "file": sheet["file"],
                    "sheet": sheet["sheet"],
                    "row_index": idx,
                    "text": text,
                    "raw_data": row.to_dict()
                })

    return all_rows

rows = process_sheets(sheets)
print(f"Processed {len(rows)} rows")
```

### Step 4: Index into Milvus

```python
from pymilvus import MilvusClient
import json

client = MilvusClient("excel.db")  # Milvus Lite

client.create_collection(
    collection_name="excel_rows",
    dimension=DIMENSION,
    auto_id=True
)

def index_rows(rows: list[dict], batch_size: int = 100):
    """Embed and index Excel rows."""
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        texts = [r["text"] for r in batch]
        vectors = embed(texts)

        data = [
            {
                "vector": vec,
                "file": r["file"],
                "sheet": r["sheet"],
                "row_index": r["row_index"],
                "text": r["text"][:500],
                "raw_data": json.dumps(r["raw_data"], default=str)[:1000]
            }
            for vec, r in zip(vectors, batch)
        ]
        client.insert(collection_name="excel_rows", data=data)
        print(f"Indexed {i + len(batch)}/{len(rows)}")

index_rows(rows)
```

### Step 5: Search

```python
import json

def search_excel(query: str, top_k: int = 10, file: str = None, sheet: str = None):
    """Search Excel rows by natural language query."""
    query_vector = embed([query])[0]

    # Build filter
    filters = []
    if file:
        filters.append(f'file == "{file}"')
    if sheet:
        filters.append(f'sheet == "{sheet}"')
    filter_expr = " and ".join(filters) if filters else None

    results = client.search(
        collection_name="excel_rows",
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=["file", "sheet", "row_index", "text", "raw_data"]
    )
    return results[0]

def print_results(results):
    for i, hit in enumerate(results, 1):
        e = hit["entity"]
        print(f"\n{'='*60}")
        print(f"#{i} [{e['file']}] Sheet: {e['sheet']}, Row: {e['row_index']}")
        print(f"Score: {hit['distance']:.3f}")
        print(f"\n{e['text']}")

        # Parse and display raw data
        try:
            raw = json.loads(e["raw_data"])
            print("\nRaw data:")
            for k, v in raw.items():
                if v and str(v) != "nan":
                    print(f"  {k}: {v}")
        except:
            pass
```

---

## Run Example

```python
# Load and index
sheets = load_all_excels("./excel_files")
rows = process_sheets(sheets)
index_rows(rows)

# Search examples
print_results(search_excel("sales in Q4 2024"))
print_results(search_excel("customer from California"))
print_results(search_excel("order over $10000"))
print_results(search_excel("pending invoices"))
```

---

## Advanced: Aggregate Search

```python
def search_and_aggregate(query: str, agg_column: str, top_k: int = 50):
    """Search and aggregate results by a column."""
    results = search_excel(query, top_k=top_k)

    aggregated = {}
    for hit in results:
        raw = json.loads(hit["entity"]["raw_data"])
        key = raw.get(agg_column, "unknown")
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(hit)

    return aggregated

# Example: Search orders and group by customer
groups = search_and_aggregate("large orders", agg_column="customer_name")
for customer, hits in groups.items():
    print(f"\n{customer}: {len(hits)} matches")
```

---

## Advanced: CSV Support

```python
def load_csv(file_path: str) -> dict:
    """Load a CSV file."""
    df = pd.read_csv(file_path)
    return {
        "file": os.path.basename(file_path),
        "sheet": "main",
        "headers": list(df.columns),
        "dataframe": df
    }

def load_all_files(folder: str) -> list[dict]:
    """Load Excel and CSV files."""
    all_sheets = []

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if file.endswith((".xlsx", ".xls")):
                all_sheets.extend(load_excel(file_path))
            elif file.endswith(".csv"):
                all_sheets.append(load_csv(file_path))
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return all_sheets
```
