# Troubleshooting Multi-Hop RAG

> Build a fault diagnosis system with multi-step retrieval chains.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Documentation Language

<ask_user>
What language are your troubleshooting docs in?

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

### 3. LLM for Diagnosis

<ask_user>
Choose LLM for reasoning:

| Model | Notes |
|-------|-------|
| **GPT-4o** | Best reasoning |
| **GPT-4o-mini** | Cost-effective |
</ask_user>

### 4. Data Scale

<ask_user>
How many fault patterns do you have?

- Each fault = symptom + causes + solutions
- Example: 500 faults × 3 tables ≈ 5K-10K vectors

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
uv init troubleshooting-rag
cd troubleshooting-rag
uv add pymilvus openai
```

---

## Why Multi-Hop for Troubleshooting

Troubleshooting requires chained retrieval:
1. **Symptom Recognition**: User description → Match fault patterns
2. **Root Cause Analysis**: Symptom → Possible causes → Root cause
3. **Solution Retrieval**: Cause → Resolution steps
4. **Verification**: After resolution → Verify if fixed

A single retrieval cannot complete this chain.

---

## End-to-End Implementation

### Step 1: Configure Models

```python
from openai import OpenAI

client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536

def generate(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content
```

### Step 2: Create Collections (Multi-Table)

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("troubleshooting.db")

# Faults collection
fault_schema = milvus.create_schema(auto_id=True)
fault_schema.add_field("id", DataType.INT64, is_primary=True)
fault_schema.add_field("fault_id", DataType.VARCHAR, max_length=64)
fault_schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
fault_schema.add_field("symptom", DataType.VARCHAR, max_length=65535)
fault_schema.add_field("fault_code", DataType.VARCHAR, max_length=32)
fault_schema.add_field("fault_name", DataType.VARCHAR, max_length=256)
fault_schema.add_field("severity", DataType.VARCHAR, max_length=16)
fault_schema.add_field("system", DataType.VARCHAR, max_length=64)

fault_index = milvus.prepare_index_params()
fault_index.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
milvus.create_collection("faults", schema=fault_schema, index_params=fault_index)

# Causes collection
cause_schema = milvus.create_schema(auto_id=True)
cause_schema.add_field("id", DataType.INT64, is_primary=True)
cause_schema.add_field("cause_id", DataType.VARCHAR, max_length=64)
cause_schema.add_field("fault_id", DataType.VARCHAR, max_length=64)
cause_schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
cause_schema.add_field("description", DataType.VARCHAR, max_length=65535)
cause_schema.add_field("probability", DataType.FLOAT)
cause_schema.add_field("is_root_cause", DataType.BOOL)

cause_index = milvus.prepare_index_params()
cause_index.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
milvus.create_collection("causes", schema=cause_schema, index_params=cause_index)

# Solutions collection
sol_schema = milvus.create_schema(auto_id=True)
sol_schema.add_field("id", DataType.INT64, is_primary=True)
sol_schema.add_field("solution_id", DataType.VARCHAR, max_length=64)
sol_schema.add_field("cause_id", DataType.VARCHAR, max_length=64)
sol_schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
sol_schema.add_field("steps", DataType.VARCHAR, max_length=65535)
sol_schema.add_field("difficulty", DataType.VARCHAR, max_length=16)
sol_schema.add_field("estimated_time", DataType.INT32)

sol_index = milvus.prepare_index_params()
sol_index.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
milvus.create_collection("solutions", schema=sol_schema, index_params=sol_index)
```

### Step 3: Multi-Hop Diagnosis

```python
def diagnose(symptom_description: str, system: str = None) -> dict:
    """Complete multi-hop diagnosis."""
    result = {
        "symptom": symptom_description,
        "matched_faults": [],
        "possible_causes": [],
        "recommended_solutions": [],
        "diagnosis_path": []
    }

    # Hop 1: Symptom → Fault patterns
    symptom_embedding = embed([symptom_description])[0]
    filter_expr = f'system == "{system}"' if system else None

    faults = milvus.search(
        collection_name="faults",
        data=[symptom_embedding],
        filter=filter_expr,
        limit=5,
        output_fields=["fault_id", "fault_code", "fault_name", "severity", "symptom"]
    )

    result["matched_faults"] = [{
        "fault_id": f["entity"]["fault_id"],
        "fault_code": f["entity"]["fault_code"],
        "fault_name": f["entity"]["fault_name"],
        "severity": f["entity"]["severity"],
        "match_score": f["distance"]
    } for f in faults[0]]

    result["diagnosis_path"].append({
        "step": 1, "action": "Symptom matching", "found": len(faults[0])
    })

    if not faults[0]:
        return result

    # Hop 2: Fault → Causes
    all_causes = []
    for fault in result["matched_faults"][:3]:
        causes = milvus.query(
            collection_name="causes",
            filter=f'fault_id == "{fault["fault_id"]}"',
            output_fields=["cause_id", "description", "probability", "is_root_cause"],
            limit=10
        )
        for c in causes:
            c["fault_name"] = fault["fault_name"]
        all_causes.extend(causes)

    all_causes.sort(key=lambda x: x.get("probability", 0), reverse=True)
    result["possible_causes"] = all_causes

    result["diagnosis_path"].append({
        "step": 2, "action": "Cause analysis", "found": len(all_causes)
    })

    # Hop 3: Cause → Solutions
    for cause in all_causes[:5]:
        solutions = milvus.query(
            collection_name="solutions",
            filter=f'cause_id == "{cause["cause_id"]}"',
            output_fields=["steps", "difficulty", "estimated_time"],
            limit=5
        )
        for s in solutions:
            s["cause_description"] = cause["description"]
        result["recommended_solutions"].extend(solutions)

    # Sort by difficulty
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    result["recommended_solutions"].sort(
        key=lambda x: (difficulty_order.get(x.get("difficulty", "medium"), 1),
                      x.get("estimated_time", 60))
    )

    result["diagnosis_path"].append({
        "step": 3, "action": "Solution retrieval", "found": len(result["recommended_solutions"])
    })

    # Generate summary
    result["summary"] = generate_summary(result)

    return result

def generate_summary(diagnosis: dict) -> str:
    """Generate diagnosis summary with LLM."""
    prompt = f"""Based on the diagnosis results, generate a fault diagnosis report.

Symptom: {diagnosis["symptom"]}

Matched faults:
{chr(10).join([f"- {f['fault_name']} ({f['severity']})" for f in diagnosis["matched_faults"][:3]])}

Possible causes (by probability):
{chr(10).join([f"- {c['description']} ({c.get('probability', 'N/A')})" for c in diagnosis["possible_causes"][:5]])}

Solutions:
{chr(10).join([f"- {s['steps'][:100]}... ({s.get('difficulty', 'N/A')}, {s.get('estimated_time', 'N/A')} min)" for s in diagnosis["recommended_solutions"][:3]])}

Generate:
1. Most likely fault assessment
2. Recommended troubleshooting order
3. Important notes

Report:"""

    return generate(prompt)
```

---

## Run Example

```python
# Index fault data
import uuid

fault_id = str(uuid.uuid4())
cause_id = str(uuid.uuid4())

milvus.insert(collection_name="faults", data=[{
    "fault_id": fault_id,
    "embedding": embed(["Server slow, CPU high"])[0],
    "symptom": "Server response slow, CPU usage above 90%",
    "fault_code": "SRV-001",
    "fault_name": "CPU Overload",
    "severity": "high",
    "system": "Linux Server"
}])

milvus.insert(collection_name="causes", data=[{
    "cause_id": cause_id,
    "fault_id": fault_id,
    "embedding": embed(["Runaway process consuming CPU"])[0],
    "description": "Runaway process consuming CPU resources",
    "probability": 0.7,
    "is_root_cause": True
}])

milvus.insert(collection_name="solutions", data=[{
    "solution_id": str(uuid.uuid4()),
    "cause_id": cause_id,
    "embedding": embed(["Kill runaway process"])[0],
    "steps": "1. Run 'top' to identify process. 2. Kill with 'kill -9 PID'. 3. Monitor CPU.",
    "difficulty": "easy",
    "estimated_time": 5
}])

# Diagnose
result = diagnose("Server response is slow, CPU usage consistently above 90%", system="Linux Server")

print("=== Diagnosis Report ===")
print(f"Matched Faults: {[f['fault_name'] for f in result['matched_faults']]}")
print(f"Possible Causes: {[c['description'] for c in result['possible_causes'][:3]]}")
print(f"\nSummary:\n{result['summary']}")
```
