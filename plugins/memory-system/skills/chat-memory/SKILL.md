---
name: chat-memory
description: "Use when user needs long-term memory for chatbots. Triggers on: chat memory, conversation history, long-term memory, chatbot memory, memory retrieval, persistent memory."
---

# Chat Memory

Implement long-term memory for chatbots, retrieving relevant conversation history across sessions.

## Use Cases

- Long-term conversation assistants (remember user preferences)
- Customer service systems (remember history issues)
- Game NPCs (remember player interactions)
- Personal AI butler

## Architecture

```
User message → Retrieve relevant history → Combine context → LLM generates → Store new conversation
```

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType
from openai import OpenAI
import time

class ChatMemory:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()
        self.collection_name = "chat_memory"
        self._init_collection()

    def _embed(self, text: str) -> list:
        """Generate embedding using OpenAI API"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("user_id", DataType.VARCHAR, max_length=64)
        schema.add_field("role", DataType.VARCHAR, max_length=16)  # user/assistant
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("timestamp", DataType.INT64)
        schema.add_field("session_id", DataType.VARCHAR, max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="user_id", index_type="TRIE")
        index_params.add_index(field_name="timestamp", index_type="STL_SORT")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def _generate_id(self) -> str:
        import uuid
        return str(uuid.uuid4())

    def store_message(self, user_id: str, role: str, content: str, session_id: str = ""):
        """Store message"""
        embedding = self._embed(content)
        self.client.insert(
            collection_name=self.collection_name,
            data=[{
                "id": self._generate_id(),
                "user_id": user_id,
                "role": role,
                "content": content,
                "timestamp": int(time.time()),
                "session_id": session_id,
                "embedding": embedding
            }]
        )

    def retrieve_relevant_history(self, user_id: str, query: str, limit: int = 10,
                                   time_decay: bool = True) -> list:
        """Retrieve relevant history"""
        embedding = self._embed(query)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            filter=f'user_id == "{user_id}"',
            limit=limit * 2,  # Get more for time decay
            output_fields=["role", "content", "timestamp", "session_id"]
        )

        memories = []
        for hit in results[0]:
            memory = {
                "role": hit["entity"]["role"],
                "content": hit["entity"]["content"],
                "timestamp": hit["entity"]["timestamp"],
                "session_id": hit["entity"]["session_id"],
                "similarity": hit["distance"]
            }

            # Time decay
            if time_decay:
                days_ago = (time.time() - memory["timestamp"]) / 86400
                decay = 0.95 ** days_ago  # 5% decay per day
                memory["final_score"] = memory["similarity"] * decay
            else:
                memory["final_score"] = memory["similarity"]

            memories.append(memory)

        # Sort by final score
        memories.sort(key=lambda x: x["final_score"], reverse=True)
        return memories[:limit]

    def get_recent_messages(self, user_id: str, session_id: str = "", limit: int = 10) -> list:
        """Get recent messages"""
        filter_expr = f'user_id == "{user_id}"'
        if session_id:
            filter_expr += f' and session_id == "{session_id}"'

        results = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["role", "content", "timestamp"],
            limit=limit
        )

        # Sort by time
        results.sort(key=lambda x: x["timestamp"])
        return results

    def chat(self, user_id: str, message: str, session_id: str = "", use_memory: bool = True) -> str:
        """Chat (with memory)"""
        messages = [{"role": "system", "content": """You are an assistant with long-term memory.
You can remember previous conversations with the user and reference these memories when appropriate.
If the user mentions a previously discussed topic, try to connect to the earlier conversation."""}]

        # 1. Retrieve relevant history
        if use_memory:
            relevant_history = self.retrieve_relevant_history(user_id, message, limit=5)
            if relevant_history:
                memory_text = "\n".join([
                    f"[{m['role']}]: {m['content']}" for m in relevant_history
                ])
                messages.append({
                    "role": "system",
                    "content": f"Here are relevant previous conversations with the user:\n{memory_text}"
                })

        # 2. Get recent messages from current session
        recent = self.get_recent_messages(user_id, session_id, limit=5)
        for msg in recent:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # 3. Add current message
        messages.append({"role": "user", "content": message})

        # 4. Generate response
        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            temperature=0.7
        )
        assistant_message = response.choices[0].message.content

        # 5. Store conversation
        self.store_message(user_id, "user", message, session_id)
        self.store_message(user_id, "assistant", assistant_message, session_id)

        return assistant_message

    def summarize_user_profile(self, user_id: str) -> str:
        """Summarize user profile"""
        # Get recent 100 conversations
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'user_id == "{user_id}"',
            output_fields=["role", "content"],
            limit=100
        )

        if not results:
            return "Not enough conversation data yet"

        user_messages = [r["content"] for r in results if r["role"] == "user"]
        sample_text = "\n".join(user_messages[-20:])  # Last 20

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""Based on the following user conversation history, summarize the user's characteristics, interests, and preferences:

{sample_text}

User profile:"""
            }],
            temperature=0
        )
        return response.choices[0].message.content

# Usage
memory = ChatMemory()

# Chat
user_id = "user001"
session_id = "session001"

# Day 1
response = memory.chat(user_id, "I'm learning Python lately", session_id)
print(f"Assistant: {response}")

response = memory.chat(user_id, "Any good learning resources to recommend?", session_id)
print(f"Assistant: {response}")

# Few days later... new session
session_id = "session002"
response = memory.chat(user_id, "I want to learn something new")  # Will retrieve earlier Python learning memory
print(f"Assistant: {response}")

# View user profile
profile = memory.summarize_user_profile(user_id)
print(f"User profile: {profile}")
```

## Memory Strategies

### 1. Sliding Window + Retrieval

```python
# Short-term: Recent N messages (sliding window)
# Long-term: Relevance retrieval
recent = self.get_recent_messages(limit=5)      # Short-term
relevant = self.retrieve_relevant_history(limit=5)  # Long-term
```

### 2. Memory Compression

```python
def compress_memories(self, user_id: str):
    """Periodically compress old memories"""
    old_messages = self.get_messages_before(user_id, days_ago=30)

    # Summarize with LLM
    summary = self.llm.summarize(old_messages)

    # Store summary, delete originals
    self.store_summary(user_id, summary)
    self.delete_old_messages(user_id, days_ago=30)
```

### 3. Importance Marking

```python
# Mark important memories (no decay)
def mark_important(self, memory_id: str):
    self.client.upsert(
        collection_name=self.collection_name,
        data=[{"id": memory_id, "important": True}]
    )
```

## Vertical Applications

See `verticals/` directory for detailed guides:
- `personal-assistant.md` - Personal assistant
- `customer-service.md` - Customer service memory
- `game-npc.md` - Game NPC

## Related Tools

- RAG: `rag-toolkit:rag`
- Vectorization: `core:embedding`
- Indexing: `core:indexing`
