# Personal Assistant Memory System

> Build a chatbot with long-term memory across sessions.

---

## Before You Start

Answer these questions to configure the optimal setup:

### 1. Conversation Language

<ask_user>
What language will conversations be in?

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

### 3. LLM for Assistant

<ask_user>
Choose LLM for conversation:

| Model | Notes |
|-------|-------|
| **GPT-4o** | Best quality |
| **GPT-4o-mini** | Cost-effective |
| **Local LLM** | Privacy, offline |
</ask_user>

### 4. Data Scale

<ask_user>
How many users and memories?

- Each user ≈ 100-1000 memories over time

| Memory Count | Recommended Milvus |
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
uv init personal-assistant
cd personal-assistant
uv add pymilvus openai
```

---

## Memory Types

| Type | Description | Examples |
|------|-------------|----------|
| Fact | User information | "I live in New York" |
| Preference | User preferences | "I like concise answers" |
| Task | Pending tasks | "Remind me tomorrow" |

---

## End-to-End Implementation

### Step 1: Configure Models

```python
from openai import OpenAI
import time
import uuid

client = OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

DIMENSION = 1536
```

### Step 2: Create Collection

```python
from pymilvus import MilvusClient, DataType

milvus = MilvusClient("assistant_memory.db")

schema = milvus.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("user_id", DataType.VARCHAR, max_length=64)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("memory_type", DataType.VARCHAR, max_length=32)  # fact/preference/task
schema.add_field("category", DataType.VARCHAR, max_length=64)
schema.add_field("created_at", DataType.INT64)
schema.add_field("last_accessed", DataType.INT64)
schema.add_field("access_count", DataType.INT32)
schema.add_field("importance", DataType.FLOAT)

index_params = milvus.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")

milvus.create_collection("assistant_memory", schema=schema, index_params=index_params)
```

### Step 3: Memory Extraction & Storage

```python
def extract_memories(user_message: str, assistant_reply: str) -> list:
    """Extract memorable information from conversation."""
    prompt = f"""Analyze this conversation and extract information worth remembering.

User: {user_message}
Assistant: {assistant_reply}

Return JSON array of memories, each with:
- content: The memory content
- type: fact/preference/task
- category: personal/work/health/hobby/other
- importance: 0-1

If nothing worth remembering, return []

Example:
[{{"content": "User lives in New York", "type": "fact", "category": "personal", "importance": 0.8}}]

Output:"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        import json
        return json.loads(resp.choices[0].message.content)
    except:
        return []

def save_memories(user_id: str, memories: list):
    """Save memories, deduplicating similar ones."""
    for mem in memories:
        # Check for similar existing memory
        existing = search_memories(user_id, mem["content"], limit=1)
        if existing and existing[0]["distance"] > 0.9:
            # Update existing
            milvus.upsert(
                collection_name="assistant_memory",
                data=[{
                    "id": existing[0]["id"],
                    "last_accessed": int(time.time()),
                    "access_count": existing[0]["entity"]["access_count"] + 1
                }]
            )
            continue

        # Insert new
        milvus.insert(
            collection_name="assistant_memory",
            data=[{
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "embedding": embed([mem["content"]])[0],
                "content": mem["content"],
                "memory_type": mem["type"],
                "category": mem.get("category", "other"),
                "created_at": int(time.time()),
                "last_accessed": int(time.time()),
                "access_count": 1,
                "importance": mem.get("importance", 0.5)
            }]
        )

def search_memories(user_id: str, query: str, limit: int = 5, memory_type: str = None):
    """Search relevant memories."""
    embedding = embed([query])[0]

    filter_expr = f'user_id == "{user_id}"'
    if memory_type:
        filter_expr += f' and memory_type == "{memory_type}"'

    results = milvus.search(
        collection_name="assistant_memory",
        data=[embedding],
        filter=filter_expr,
        limit=limit,
        output_fields=["id", "content", "memory_type", "category", "importance", "access_count"]
    )

    return results[0]
```

### Step 4: Memory-Enhanced Response

```python
def get_user_profile(user_id: str) -> dict:
    """Get user profile from memories."""
    facts = milvus.query(
        collection_name="assistant_memory",
        filter=f'user_id == "{user_id}" and memory_type == "fact"',
        output_fields=["content", "category"],
        limit=50
    )

    preferences = milvus.query(
        collection_name="assistant_memory",
        filter=f'user_id == "{user_id}" and memory_type == "preference"',
        output_fields=["content"],
        limit=50
    )

    profile = {"facts": {}, "preferences": []}

    for f in facts:
        cat = f["category"]
        if cat not in profile["facts"]:
            profile["facts"][cat] = []
        profile["facts"][cat].append(f["content"])

    profile["preferences"] = [p["content"] for p in preferences]

    return profile

def generate_response(user_id: str, message: str, history: list = None) -> str:
    """Generate response with memory context."""
    # Search relevant memories
    relevant = search_memories(user_id, message, limit=5)

    # Get user profile
    profile = get_user_profile(user_id)

    # Build system prompt with memory
    system_prompt = "You are a personal assistant.\n\n"

    if profile["facts"]:
        system_prompt += "User information:\n"
        for cat, facts in profile["facts"].items():
            system_prompt += f"- {cat}: {', '.join(facts)}\n"

    if profile["preferences"]:
        system_prompt += f"\nPreferences: {', '.join(profile['preferences'])}\n"

    if relevant:
        system_prompt += "\nRelevant memories:\n"
        for m in relevant:
            system_prompt += f"- {m['entity']['content']}\n"

    # Generate response
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-10:])
    messages.append({"role": "user", "content": message})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )

    reply = resp.choices[0].message.content

    # Extract and save new memories
    new_memories = extract_memories(message, reply)
    if new_memories:
        save_memories(user_id, new_memories)

    return reply
```

---

## Run Example

```python
user_id = "user_001"

# First conversation - establishes memory
reply = generate_response(user_id, "Hi, I'm Mike and I live in New York")
print(f"Assistant: {reply}")
# Memories saved: name=Mike, location=New York

# Later conversation - uses memory
reply = generate_response(user_id, "How's the weather in my city?")
print(f"Assistant: {reply}")
# Assistant knows user is in New York

# Preference learning
reply = generate_response(user_id, "Keep answers brief, don't ramble")
print(f"Assistant: {reply}")
# Future answers will be more concise

# Check profile
profile = get_user_profile(user_id)
print(f"User Profile: {profile}")
```

---

## Memory Decay (Optional)

```python
def decay_old_memories(user_id: str, decay_rate: float = 0.1):
    """Reduce importance of old, unused memories."""
    one_month_ago = int(time.time()) - 30 * 24 * 3600

    old_memories = milvus.query(
        collection_name="assistant_memory",
        filter=f'user_id == "{user_id}" and last_accessed < {one_month_ago}',
        output_fields=["id", "importance"]
    )

    for mem in old_memories:
        new_importance = max(0.1, mem["importance"] * (1 - decay_rate))
        milvus.upsert(
            collection_name="assistant_memory",
            data=[{"id": mem["id"], "importance": new_importance}]
        )
```
