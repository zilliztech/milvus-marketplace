---
name: chat-memory
description: "Use when user needs long-term memory for chatbots. Triggers on: chat memory, conversation history, long-term memory, chatbot memory, memory retrieval, persistent memory, remember conversations."
---

# Chat Memory

Implement long-term memory for chatbots — remember and retrieve relevant conversation history across sessions.

## When to Activate

Activate this skill when:
- User needs **persistent conversation memory** for a chatbot
- User mentions "remember", "long-term memory", "conversation history"
- User wants a chatbot that **recalls past interactions**
- User needs to **personalize responses** based on history

**Do NOT activate** when:
- User only needs in-session context → use standard context window
- User needs document search → use `rag-toolkit:rag`
- User needs user recommendations → use `rec-system`

## Interactive Flow

### Step 1: Understand Memory Scope

"What should the chatbot remember?"

A) **User preferences** (favorite topics, communication style)
   - Long retention, low decay
   - e.g., "User prefers technical explanations"

B) **Conversation context** (discussed topics, mentioned names)
   - Medium retention
   - e.g., "User mentioned they're working on project X"

C) **Factual information** (user-provided facts)
   - Variable retention
   - e.g., "User's dog is named Max"

D) **All of the above**

Which types matter most?

### Step 2: Retention Strategy

"How long should memories last?"

| Type | Retention | Decay |
|------|-----------|-------|
| **Preferences** | Permanent | None |
| **Recent context** | Days-weeks | 5% per day |
| **Old context** | Compressed | Summarized |

### Step 3: Confirm Configuration

"Based on your requirements:

- **Memory types**: All (preferences, context, facts)
- **Retrieval**: Semantic similarity + time decay
- **Compression**: Auto-summarize after 30 days

Proceed? (yes / adjust [what])"

## Core Concepts

### Mental Model: Human Memory

Think of chat memory like **human memory**:
- **Working memory**: Current conversation (context window)
- **Long-term memory**: Past conversations (vector database)
- **Recall**: Retrieve relevant memories when needed

```
┌─────────────────────────────────────────────────────────┐
│                   Chat Memory System                     │
│                                                          │
│  User Message: "How's my Python project going?"          │
│                         │                                │
│       ┌─────────────────┼─────────────────┐             │
│       │                 │                 │             │
│       ▼                 ▼                 ▼             │
│  ┌─────────┐    ┌─────────────┐   ┌─────────────┐      │
│  │ Recent  │    │   Memory    │   │   Memory    │      │
│  │ Context │    │  Retrieval  │   │   Retrieval │      │
│  │(session)│    │ (semantic)  │   │   (keyword) │      │
│  └────┬────┘    └──────┬──────┘   └──────┬──────┘      │
│       │                │                 │              │
│       ▼                ▼                 ▼              │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Combined Context                     │  │
│  │                                                   │  │
│  │  Recent: "We discussed Python yesterday"         │  │
│  │  Memory: "User started Python project 2 weeks    │  │
│  │          ago, learning Flask for web app"        │  │
│  │  Memory: "User mentioned deadline is next month" │  │
│  └──────────────────────┬───────────────────────────┘  │
│                         │                               │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │                   LLM Response                    │  │
│  │  "Based on our previous conversations, I know    │  │
│  │   you're building a Flask web app. Since your    │  │
│  │   deadline is next month, let me help you..."    │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  [Store: "User asked about Python project progress"]    │
└─────────────────────────────────────────────────────────┘
```

### Memory vs RAG

| Aspect | Chat Memory | RAG |
|--------|-------------|-----|
| **Source** | Past conversations | Documents |
| **Updates** | Every conversation | Batch indexing |
| **Personal** | Per-user | Shared |
| **Decay** | Time-based | Usually none |

## Implementation

```python
from pymilvus import MilvusClient, DataType
from openai import OpenAI
import time
import uuid

class ChatMemory:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()
        self.collection_name = "chat_memory"
        self._init_collection()

    def _embed(self, text: str) -> list:
        """Generate embedding"""
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
        schema.add_field("importance", DataType.FLOAT)  # 0-1, for prioritization
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

    def store_message(self, user_id: str, role: str, content: str,
                      session_id: str = "", importance: float = 0.5):
        """Store a message in memory"""
        embedding = self._embed(content)

        self.client.insert(
            collection_name=self.collection_name,
            data=[{
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "role": role,
                "content": content,
                "timestamp": int(time.time()),
                "session_id": session_id,
                "importance": importance,
                "embedding": embedding
            }]
        )

    def retrieve_relevant(self, user_id: str, query: str, limit: int = 10,
                          apply_decay: bool = True) -> list:
        """Retrieve relevant memories for a query"""
        embedding = self._embed(query)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            filter=f'user_id == "{user_id}"',
            limit=limit * 2,  # Get extra for decay filtering
            output_fields=["role", "content", "timestamp", "importance"]
        )

        memories = []
        for hit in results[0]:
            memory = {
                "role": hit["entity"]["role"],
                "content": hit["entity"]["content"],
                "timestamp": hit["entity"]["timestamp"],
                "importance": hit["entity"]["importance"],
                "similarity": hit["distance"]
            }

            # Apply time decay
            if apply_decay:
                days_ago = (time.time() - memory["timestamp"]) / 86400
                decay = 0.95 ** days_ago  # 5% decay per day
                memory["final_score"] = memory["similarity"] * decay * (0.5 + memory["importance"] * 0.5)
            else:
                memory["final_score"] = memory["similarity"]

            memories.append(memory)

        # Sort by final score
        memories.sort(key=lambda x: x["final_score"], reverse=True)
        return memories[:limit]

    def get_recent(self, user_id: str, session_id: str = "", limit: int = 10) -> list:
        """Get recent messages (for current session context)"""
        filter_expr = f'user_id == "{user_id}"'
        if session_id:
            filter_expr += f' and session_id == "{session_id}"'

        results = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["role", "content", "timestamp"],
            limit=limit
        )

        results.sort(key=lambda x: x["timestamp"])
        return results

    def chat(self, user_id: str, message: str, session_id: str = "") -> str:
        """Chat with memory-enhanced response"""
        # Build context
        messages = [{
            "role": "system",
            "content": """You are a helpful assistant with long-term memory.
You can recall previous conversations and use them to provide personalized responses.
When relevant, reference past discussions naturally."""
        }]

        # Retrieve relevant memories
        memories = self.retrieve_relevant(user_id, message, limit=5)
        if memories:
            memory_text = "\n".join([
                f"[Past - {m['role']}]: {m['content']}"
                for m in memories
            ])
            messages.append({
                "role": "system",
                "content": f"Relevant past conversations:\n{memory_text}"
            })

        # Add recent session context
        recent = self.get_recent(user_id, session_id, limit=5)
        for msg in recent:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current message
        messages.append({"role": "user", "content": message})

        # Generate response
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
        )
        assistant_message = response.choices[0].message.content

        # Store conversation
        self.store_message(user_id, "user", message, session_id)
        self.store_message(user_id, "assistant", assistant_message, session_id)

        return assistant_message

    def mark_important(self, user_id: str, content_snippet: str):
        """Mark a memory as important (no decay)"""
        # Find the memory
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'user_id == "{user_id}"',
            output_fields=["id", "content", "importance"],
            limit=100
        )

        for r in results:
            if content_snippet in r["content"]:
                self.client.upsert(
                    collection_name=self.collection_name,
                    data=[{"id": r["id"], "importance": 1.0}]
                )
                return True
        return False

# Usage
memory = ChatMemory()

user_id = "user001"
session_id = "session_" + str(int(time.time()))

# Day 1 conversation
response = memory.chat(user_id, "I'm learning Python for a web project", session_id)
print(f"Bot: {response}")

response = memory.chat(user_id, "Any Flask tutorials you recommend?", session_id)
print(f"Bot: {response}")

# Days later, new session
new_session = "session_" + str(int(time.time()))
response = memory.chat(user_id, "How should I continue my learning?", new_session)
# Bot will recall the Python/Flask context
print(f"Bot: {response}")
```

## Memory Strategies

### 1. Sliding Window + Retrieval

```python
# Combine recent context (window) with relevant memories (retrieval)
recent_messages = get_recent(limit=5)  # Last 5 messages
relevant_memories = retrieve_relevant(query, limit=5)  # Top 5 relevant

context = recent_messages + relevant_memories
```

### 2. Memory Compression

```python
def compress_old_memories(self, user_id: str, days_threshold: int = 30):
    """Summarize and compress old memories"""
    cutoff = int(time.time()) - (days_threshold * 86400)

    old_memories = self.client.query(
        collection_name=self.collection_name,
        filter=f'user_id == "{user_id}" and timestamp < {cutoff}',
        output_fields=["content"],
        limit=100
    )

    if len(old_memories) > 10:
        # Summarize with LLM
        content = "\n".join([m["content"] for m in old_memories])
        summary = self.llm.summarize(content)

        # Store summary, delete originals
        self.store_message(user_id, "summary", summary, importance=0.8)
        self.delete_old_memories(user_id, cutoff)
```

### 3. Importance Detection

```python
def detect_importance(self, content: str) -> float:
    """Automatically detect if content is important"""
    # Keywords indicating important info
    important_keywords = ["always", "never", "prefer", "hate", "love",
                          "my name", "birthday", "deadline", "important"]

    content_lower = content.lower()
    matches = sum(1 for kw in important_keywords if kw in content_lower)

    return min(0.5 + (matches * 0.1), 1.0)
```

## Common Pitfalls

### ❌ Pitfall 1: Retrieving Irrelevant Memories

**Problem**: Bot mentions unrelated past conversations

**Fix**: Increase similarity threshold
```python
memories = [m for m in memories if m["similarity"] > 0.7]
```

### ❌ Pitfall 2: Memory Overload

**Problem**: Too many memories in context, confuses LLM

**Fix**: Limit memories, summarize if needed
```python
memories = retrieve_relevant(query, limit=3)  # Only top 3
```

### ❌ Pitfall 3: Privacy Leaks

**Problem**: Memory from one user leaks to another

**Fix**: Always filter by user_id
```python
filter=f'user_id == "{user_id}"'  # ALWAYS include
```

### ❌ Pitfall 4: Stale Context

**Problem**: Bot keeps mentioning outdated information

**Fix**: Apply time decay
```python
decay = 0.95 ** days_ago
final_score = similarity * decay
```

## When to Level Up

| Need | Upgrade To |
|------|------------|
| Search documents | `rag-toolkit:rag` |
| Multi-user shared knowledge | Combine with RAG |
| Real-time streaming | Add message queue |
| Complex memory graphs | Consider Neo4j |

## References

- RAG for documents: `rag-toolkit:rag`
- Embedding models: `core:embedding`
- Vertical guides: `verticals/`
