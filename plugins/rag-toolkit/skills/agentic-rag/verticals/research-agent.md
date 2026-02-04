# Research Assistant Agent

> Build an autonomous agent that searches, analyzes, and synthesizes research.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Research Domain Language

<ask_user>
What language are your research materials in?

| Option | Model Recommendation |
|--------|---------------------|
| **English** | English models |
| **Chinese** | Chinese models |
| **Multilingual** | Multilingual models |
</ask_user>

### 2. Embedding Method

<ask_user>
Choose your embedding approach:

| Method | Pros | Cons |
|--------|------|------|
| **OpenAI API** | High quality | Requires API key |
| **Local Model** | Free, offline | Model download |
</ask_user>

### 3. Agent LLM

<ask_user>
Choose LLM for agent reasoning:

| Model | Notes |
|-------|-------|
| **GPT-4o** | Best reasoning, tool use |
| **GPT-4o-mini** | Cost-effective |
| **Claude 3.5 Sonnet** | Good for research |
</ask_user>

### 4. Data Scale

<ask_user>
How many papers/documents do you have?

| Document Count | Recommended Milvus |
|----------------|-------------------|
| < 10K | **Milvus Lite** |
| 10K - 1M | **Milvus Standalone** |
| > 1M | **Zilliz Cloud** |
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
uv init research-agent
cd research-agent
uv add pymilvus openai
```

---

## Agent Capabilities

A research agent needs to autonomously:
1. Understand research questions
2. Plan retrieval strategies
3. Multi-round retrieval and filtering
4. Comprehensive analysis and summarization

---

## End-to-End Implementation

### Step 1: Configure Models

```python
from openai import OpenAI
import json

client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536
```

### Step 2: Define Tools

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search academic papers by query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "year_from": {"type": "integer", "description": "Start year"},
                    "year_to": {"type": "integer", "description": "End year"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_details",
            "description": "Get full paper details including abstract and references",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"}
                },
                "required": ["paper_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_finding",
            "description": "Save a research finding to notes",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["background", "method", "result", "insight"]},
                    "content": {"type": "string"},
                    "source": {"type": "string"}
                },
                "required": ["category", "content", "source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate research report from saved findings",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["summary", "detailed"]}
                },
                "required": ["format"]
            }
        }
    }
]
```

### Step 3: Create Collection & Tool Functions

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("research.db")

# Create papers collection
schema = milvus.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("paper_id", DataType.VARCHAR, max_length=64)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("abstract", DataType.VARCHAR, max_length=65535)
schema.add_field("authors", DataType.VARCHAR, max_length=512)
schema.add_field("year", DataType.INT32)
schema.add_field("venue", DataType.VARCHAR, max_length=128)
schema.add_field("citations", DataType.INT32)

index_params = milvus.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

milvus.create_collection("papers", schema=schema, index_params=index_params)

# Tool implementations
findings = []

def search_papers(query: str, year_from: int = None, year_to: int = None, limit: int = 10):
    embedding = embed([query])[0]

    filters = []
    if year_from:
        filters.append(f'year >= {year_from}')
    if year_to:
        filters.append(f'year <= {year_to}')

    filter_expr = ' and '.join(filters) if filters else None

    results = milvus.search(
        collection_name="papers",
        data=[embedding],
        filter=filter_expr,
        limit=limit,
        output_fields=["paper_id", "title", "abstract", "authors", "year", "citations"]
    )

    return [{
        "paper_id": r["entity"]["paper_id"],
        "title": r["entity"]["title"],
        "abstract": r["entity"]["abstract"][:300] + "...",
        "authors": r["entity"]["authors"],
        "year": r["entity"]["year"],
        "citations": r["entity"]["citations"]
    } for r in results[0]]

def get_paper_details(paper_id: str):
    results = milvus.query(
        collection_name="papers",
        filter=f'paper_id == "{paper_id}"',
        output_fields=["title", "abstract", "authors", "year"]
    )
    return results[0] if results else None

def save_finding(category: str, content: str, source: str):
    findings.append({"category": category, "content": content, "source": source})
    return {"status": "saved", "total_findings": len(findings)}

def generate_report(format: str = "summary"):
    if not findings:
        return "No findings yet."

    by_category = {}
    for f in findings:
        cat = f["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(f)

    findings_text = ""
    for cat, items in by_category.items():
        findings_text += f"\n## {cat.upper()}\n"
        for item in items:
            findings_text += f"- {item['content']} (Source: {item['source']})\n"

    prompt = f"""Generate a {'brief' if format == 'summary' else 'detailed'} research report.

Findings:
{findings_text}

Requirements:
1. Clear structure: background, main findings, conclusions
2. {'Under 500 words' if format == 'summary' else 'Detailed with citations'}
3. Point out limitations and future directions

Report:"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

def execute_tool(name: str, args: dict) -> str:
    if name == "search_papers":
        return json.dumps(search_papers(**args), ensure_ascii=False)
    elif name == "get_paper_details":
        result = get_paper_details(**args)
        return json.dumps(result, ensure_ascii=False) if result else "Not found"
    elif name == "save_finding":
        return json.dumps(save_finding(**args))
    elif name == "generate_report":
        return generate_report(**args)
    return "Unknown tool"
```

### Step 4: Agent Loop

```python
def research(topic: str, max_iterations: int = 10) -> str:
    """Run research agent."""
    global findings
    findings = []  # Reset findings

    messages = [{
        "role": "system",
        "content": """You are a professional research assistant. Your task is to:
1. Understand the research question
2. Search relevant papers
3. Analyze and filter important findings
4. Use save_finding to record key discoveries
5. Finally use generate_report to produce the research report

Work systematically through the research task."""
    }, {
        "role": "user",
        "content": f"Please research: {topic}"
    }]

    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"[Agent] Calling: {tool_name}({tool_args})")

                result = execute_tool(tool_name, tool_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

                if tool_name == "generate_report":
                    return result
        else:
            return message.content

    return "Max iterations reached."
```

---

## Run Example

```python
# Index some papers first
milvus.insert(collection_name="papers", data=[{
    "paper_id": "paper_001",
    "embedding": embed(["Large language models for code generation"])[0],
    "title": "CodeGen: A Conversational Paradigm for Program Synthesis",
    "abstract": "We introduce CodeGen, a family of large language models...",
    "authors": "Erik Nijkamp et al.",
    "year": 2023,
    "venue": "ICLR",
    "citations": 500
}])

# Run research agent
report = research("Latest advances in LLMs for code generation")
print(report)
```

Output:
```
[Agent] Calling: search_papers({"query": "large language model code generation", "year_from": 2023})
[Agent] Calling: get_paper_details({"paper_id": "paper_001"})
[Agent] Calling: save_finding({"category": "method", "content": "CodeGen uses conversational paradigm", "source": "Nijkamp 2023"})
[Agent] Calling: generate_report({"format": "detailed"})
```
