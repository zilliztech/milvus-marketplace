# Personal Assistant Memory System

## Use Cases

- Long-term personal assistant (cross-session memory)
- Remember user preferences and habits
- Contextually continuous conversation experience

## Memory Types

| Type | Description | Examples |
|------|-------------|----------|
| Fact Memory | Personal information provided by user | "I live in New York", "I'm allergic to peanuts" |
| Preference Memory | User preferences | "I like concise answers", "I don't like emojis" |
| Conversation Memory | Historical conversation content | Previously discussed topics |
| Task Memory | Uncompleted tasks | "Remind me of the meeting tomorrow" |

## Schema Design

```python
schema = client.create_schema()
schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("user_id", DataType.VARCHAR, max_length=64)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

# Memory classification
schema.add_field("memory_type", DataType.VARCHAR, max_length=32)   # fact/preference/conversation/task
schema.add_field("category", DataType.VARCHAR, max_length=64)      # personal/work/health/hobby

# Time and importance
schema.add_field("created_at", DataType.INT64)
schema.add_field("last_accessed", DataType.INT64)
schema.add_field("access_count", DataType.INT32)
schema.add_field("importance", DataType.FLOAT)                     # Importance 0-1

# Task specific
schema.add_field("due_time", DataType.INT64)                       # Due time
schema.add_field("is_completed", DataType.BOOL)
```

## Implementation

```python
from pymilvus import MilvusClient, DataType
from openai import OpenAI
import time
import uuid

class PersonalAssistantMemory:
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

    def _init_collection(self):
        # ... collection initialization code ...
        pass

    def extract_memory(self, message: str, response: str) -> list:
        """Extract memorable information from conversation"""
        prompt = f"""Analyze the following conversation and extract information worth remembering.

User message: {message}
Assistant reply: {response}

Return JSON format for information to remember, each containing:
- content: Memory content
- type: fact/preference/task
- category: personal/work/health/hobby/other
- importance: Importance 0-1

If no information worth remembering, return empty array []

Example output:
[
  {{"content": "User lives in New York", "type": "fact", "category": "personal", "importance": 0.8}},
  {{"content": "User doesn't like spicy food", "type": "preference", "category": "personal", "importance": 0.6}}
]

Output:"""

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            import json
            return json.loads(response.choices[0].message.content)
        except:
            return []

    def save_memory(self, user_id: str, memories: list):
        """Save memories"""
        data = []
        for mem in memories:
            # Check if similar memory exists
            existing = self.search_memory(user_id, mem["content"], limit=1)
            if existing and existing[0]["distance"] > 0.9:
                # Update existing memory
                self.client.upsert(
                    collection_name="assistant_memory",
                    data=[{
                        "id": existing[0]["entity"]["id"],
                        "last_accessed": int(time.time()),
                        "access_count": existing[0]["entity"]["access_count"] + 1
                    }]
                )
                continue

            data.append({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "content": mem["content"],
                "embedding": self._embed(mem["content"]).tolist(),
                "memory_type": mem["type"],
                "category": mem.get("category", "other"),
                "created_at": int(time.time()),
                "last_accessed": int(time.time()),
                "access_count": 1,
                "importance": mem.get("importance", 0.5),
                "due_time": 0,
                "is_completed": False
            })

        if data:
            self.client.insert(collection_name="assistant_memory", data=data)

    def search_memory(self, user_id: str, query: str, limit: int = 5,
                      memory_type: str = None) -> list:
        """Search relevant memories"""
        embedding = self._embed(query).tolist()

        filter_expr = f'user_id == "{user_id}"'
        if memory_type:
            filter_expr += f' and memory_type == "{memory_type}"'

        results = self.client.search(
            collection_name="assistant_memory",
            data=[embedding],
            filter=filter_expr,
            limit=limit,
            output_fields=["id", "content", "memory_type", "category",
                          "importance", "access_count", "created_at"]
        )

        # Update access time
        for r in results[0]:
            self.client.upsert(
                collection_name="assistant_memory",
                data=[{
                    "id": r["entity"]["id"],
                    "last_accessed": int(time.time()),
                    "access_count": r["entity"]["access_count"] + 1
                }]
            )

        return results[0]

    def get_user_profile(self, user_id: str) -> dict:
        """Get user profile"""
        # Get all fact memories
        facts = self.client.query(
            collection_name="assistant_memory",
            filter=f'user_id == "{user_id}" and memory_type == "fact"',
            output_fields=["content", "category", "importance"],
            limit=50
        )

        # Get all preferences
        preferences = self.client.query(
            collection_name="assistant_memory",
            filter=f'user_id == "{user_id}" and memory_type == "preference"',
            output_fields=["content", "importance"],
            limit=50
        )

        # Organize by category
        profile = {
            "facts": {},
            "preferences": []
        }

        for f in facts:
            cat = f["category"]
            if cat not in profile["facts"]:
                profile["facts"][cat] = []
            profile["facts"][cat].append(f["content"])

        profile["preferences"] = [p["content"] for p in preferences]

        return profile

    def generate_response(self, user_id: str, message: str,
                          conversation_history: list = None) -> str:
        """Generate response with memory"""
        # 1. Search relevant memories
        relevant_memories = self.search_memory(user_id, message, limit=5)

        # 2. Get user profile
        profile = self.get_user_profile(user_id)

        # 3. Build system prompt
        system_prompt = "You are the user's personal assistant.\n\n"

        if profile["facts"]:
            system_prompt += "Information about the user:\n"
            for cat, facts in profile["facts"].items():
                system_prompt += f"- {cat}: {', '.join(facts)}\n"

        if profile["preferences"]:
            system_prompt += f"\nUser preferences: {', '.join(profile['preferences'])}\n"

        if relevant_memories:
            system_prompt += "\nRelevant memories:\n"
            for m in relevant_memories:
                system_prompt += f"- {m['entity']['content']}\n"

        # 4. Generate response
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history[-10:])  # Last 10 turns

        messages.append({"role": "user", "content": message})

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            temperature=0.7
        )

        assistant_reply = response.choices[0].message.content

        # 5. Extract new memories
        new_memories = self.extract_memory(message, assistant_reply)
        if new_memories:
            self.save_memory(user_id, new_memories)

        return assistant_reply
```

## Examples

```python
assistant = PersonalAssistantMemory()
user_id = "user_001"

# First conversation
reply = assistant.generate_response(user_id, "Hi, I'm Mike, I live in New York")
# Assistant remembers: User is named Mike, lives in New York

# Later conversation
reply = assistant.generate_response(user_id, "How's the weather in my city lately?")
# Assistant knows user is in New York, can give more accurate answer

# Remember preferences
reply = assistant.generate_response(user_id, "Keep your answers brief, don't ramble")
# Assistant remembers preference, future answers will be more concise

# View user profile
profile = assistant.get_user_profile(user_id)
print(profile)
```

## Memory Decay

```python
def decay_memories(self, user_id: str, decay_rate: float = 0.1):
    """Memory decay: reduce importance of long-unaccessed memories"""
    current_time = int(time.time())
    one_month = 30 * 24 * 3600

    old_memories = self.client.query(
        collection_name="assistant_memory",
        filter=f'user_id == "{user_id}" and last_accessed < {current_time - one_month}',
        output_fields=["id", "importance"]
    )

    for mem in old_memories:
        new_importance = max(0.1, mem["importance"] * (1 - decay_rate))
        self.client.upsert(
            collection_name="assistant_memory",
            data=[{"id": mem["id"], "importance": new_importance}]
        )
```
