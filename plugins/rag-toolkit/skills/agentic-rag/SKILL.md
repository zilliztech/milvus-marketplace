---
name: agentic-rag
description: "Use when user needs an autonomous RAG agent that decides when and what to retrieve dynamically. Triggers on: agentic RAG, agent, autonomous retrieval, tool use, function calling, research agent, conversational RAG, dynamic retrieval, self-directed search, RAG with tools, intelligent assistant, adaptive retrieval."
---

# Agentic RAG

Build an autonomous agent that decides when to retrieve, what to search for, and whether it needs more information — enabling dynamic, multi-step reasoning over your knowledge base.

## When to Activate

This skill should be activated when the user:
- Needs an AI that autonomously decides when to search
- Wants conversational Q&A that retrieves on-demand
- Asks about "agentic", "autonomous", or "tool-use" RAG
- Needs multi-step research that explores a knowledge base
- Wants the LLM to evaluate retrieval quality and retry if needed

## Interactive Flow

Agentic RAG is more complex than basic RAG. Validate the need and design tools carefully.

### Step 1: Validate Agentic Need

```
"Agentic RAG lets the AI decide when to search. But it's more complex and costly.

Does your use case need:
A) Every query needs document search (→ basic RAG is simpler)
B) Some queries need search, some don't (→ agentic fits)
C) Multi-turn conversation with context (→ agentic fits)
D) Research tasks exploring multiple topics (→ agentic fits)

Which describes your case? (A/B/C/D)"
```

| Answer | Recommendation |
|--------|----------------|
| A | "Basic RAG is simpler and faster. Use `rag` skill instead?" |
| B, C, D | Proceed with agentic design |

### Step 2: Design the Tools

```
"What knowledge sources should the agent access?

List them (e.g., 'product docs', 'FAQ', 'support tickets'):
1. ___
2. ___
3. ___

I'll create a search tool for each, or combine into one.
Should they be separate tools or one unified search? (separate / unified)"
```

| Choice | When to Use |
|--------|-------------|
| Separate tools | Different sources have different schemas, agent needs to choose |
| Unified search | Sources are similar, simpler for agent |

**Tool design confirmation**:
```
"Proposed tools:

1. **search_knowledge_base**
   - Searches: [sources]
   - When agent should use: "When user asks factual questions"

2. **search_by_source**
   - Searches within specific document
   - When agent should use: "When user mentions a specific doc"

Add/remove/modify any tools? (looks good / add [tool] / remove [tool])"
```

### Step 3: Agent Behavior

```
"How should the agent behave?

A) Conservative - only search when clearly needed, prefer direct answers
B) Thorough - always verify with search, multiple searches OK
C) Balanced - search for factual questions, skip for clarifications

Which style? (A/B/C)"
```

This determines the system prompt tone.

### Step 4: Iteration Limits

```
"How many search attempts before the agent must answer?

A) 3 (quick responses, may be incomplete)
B) 5 (balanced, good for most cases)
C) 10 (thorough research, slower)

Recommend B for most cases. Which? (A/B/C)"
```

### Step 5: Confirm Before Implementation

```
"Agent configuration:

- **Tools**: [list from Step 2]
- **Behavior**: [from Step 3]
- **Max iterations**: [from Step 4]
- **Base retrieval**: Top-5 per search

Ready to implement? (yes / adjust [what])"
```

### Checkpoints During Conversation

When running the agent, pause at key moments:

| Situation | Checkpoint |
|-----------|------------|
| Agent searches 3+ times | "Agent is searching extensively. Continue or answer now?" |
| No relevant results | "Search returned nothing relevant. Should I try different terms or answer with caveats?" |
| Agent wants to answer | "Agent ready to answer. Want to see the sources first?" |

### Red Flags - When Agentic is Overkill

```
"I notice your queries are straightforward factual questions.
Basic RAG would be:
- Faster (no agent reasoning overhead)
- Cheaper (fewer LLM calls)
- Simpler to debug

Switch to basic RAG? (yes / keep agentic)"
```

## Core Concepts

### The Agency Problem

Traditional RAG is **reactive**: every query triggers retrieval. But not every question needs retrieval:

| Question Type | Needs Retrieval? | Example |
|---------------|------------------|---------|
| Factual about corpus | Yes | "What's our refund policy?" |
| Clarification | No | "Can you explain that simpler?" |
| General knowledge | No | "What is machine learning?" |
| Follow-up | Maybe | "What about for enterprise customers?" |

**Key insight**: The LLM should decide when retrieval adds value, not retrieve blindly.

### ReAct Pattern

Agentic RAG follows the **ReAct** (Reasoning + Acting) pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ReAct Loop                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Question                                                  │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                     │
│  │  THINK  │───▶│   ACT   │───▶│ OBSERVE │──┐                  │
│  │ (Reason)│    │ (Tool)  │    │ (Result)│  │                  │
│  └─────────┘    └─────────┘    └─────────┘  │                  │
│       ▲                                      │                  │
│       └──────────────────────────────────────┘                  │
│                    Loop until done                              │
│                          │                                      │
│                          ▼                                      │
│                    Final Answer                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Example trace**:
```
User: "Compare our pricing with competitor X"

Think: I need to find our pricing first.
Act: search_knowledge_base("pricing plans features")
Observe: [Our pricing: Basic $10, Pro $25, Enterprise custom...]

Think: Now I need competitor X's pricing. Let me search.
Act: search_knowledge_base("competitor X pricing comparison")
Observe: [No relevant results found]

Think: Knowledge base doesn't have competitor info. I should answer with what I have.
Answer: "Our pricing is... I don't have competitor X's pricing in the knowledge base."
```

### Tools as Capabilities

The agent's capabilities are defined by its tools:

```python
tools = [
    {
        "name": "search_knowledge_base",
        "description": "Search for information in the knowledge base",
        "when_to_use": "When you need specific facts to answer the question"
    },
    {
        "name": "search_by_source",
        "description": "Search within a specific document",
        "when_to_use": "When you know which document contains the answer"
    },
    {
        "name": "list_sources",
        "description": "List all available documents",
        "when_to_use": "When you need to know what information is available"
    }
]
```

## Why Agentic RAG Over Alternatives

| Approach | Retrieval Control | Multi-Step | Conversation | Complexity |
|----------|-------------------|------------|--------------|------------|
| Basic RAG | Always retrieves | No | Limited | Low |
| **Agentic RAG** | Agent decides | Yes | Native | Medium |
| Multi-Hop RAG | Structured hops | Yes (fixed) | Limited | Medium |
| Full Agent | Full autonomy | Yes | Yes | High |

**Choose Agentic RAG when**:
- Queries vary widely (some need retrieval, some don't)
- Conversation context matters
- You want the LLM to evaluate and retry searches
- Research tasks require exploring the knowledge base

**Skip Agentic RAG when**:
- Every query needs retrieval (use basic RAG)
- Latency budget is very tight (<500ms)
- Simple Q&A without conversation

## Implementation

```python
from pymilvus import MilvusClient, DataType
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

class AgenticRAG:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()
        self.collection_name = "agentic_rag"
        self._init_collection()

        # Define tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search the knowledge base for relevant information. Use when you need to find specific facts, policies, or documentation to answer the user's question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query - be specific and use keywords"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_by_source",
                    "description": "Search within a specific document. Use when you know or suspect the answer is in a particular document.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "source": {"type": "string", "description": "Document name/path"}
                        },
                        "required": ["query", "source"]
                    }
                }
            }
        ]

    def _embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        response = self.openai.embeddings.create(model="text-embedding-3-small", input=texts)
        return [item.embedding for item in response.data]

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return
        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

        index_params = self.client.prepare_index_params()
        index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(self.collection_name, schema=schema, index_params=index_params)

    def add_document(self, text: str, source: str = ""):
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_text(text)
        embeddings = self._embed(chunks)
        data = [{"text": c, "source": source, "embedding": e} for c, e in zip(chunks, embeddings)]
        self.client.insert(self.collection_name, data)

    def search_knowledge_base(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = self._embed(query)[0]
        results = self.client.search(self.collection_name, [embedding], limit=top_k,
                                     output_fields=["text", "source"])
        return [{"text": h["entity"]["text"], "source": h["entity"]["source"]} for h in results[0]]

    def search_by_source(self, query: str, source: str) -> list[dict]:
        embedding = self._embed(query)[0]
        results = self.client.search(self.collection_name, [embedding],
                                     filter=f'source == "{source}"', limit=5,
                                     output_fields=["text", "source"])
        return [{"text": h["entity"]["text"], "source": h["entity"]["source"]} for h in results[0]]

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "search_knowledge_base":
            results = self.search_knowledge_base(**args)
        elif name == "search_by_source":
            results = self.search_by_source(**args)
        else:
            results = {"error": f"Unknown tool: {name}"}
        return json.dumps(results, ensure_ascii=False)

    def chat(self, user_message: str, history: list = None, max_iterations: int = 5) -> dict:
        messages = [
            {"role": "system", "content": """You are a helpful assistant with access to a knowledge base.

Guidelines:
1. Use search_knowledge_base when you need specific information to answer the question
2. Use search_by_source when you know which document contains the answer
3. You can search multiple times to gather complete information
4. If search results aren't relevant, try rephrasing your query
5. If the knowledge base doesn't have the answer, say so clearly
6. Don't search for general knowledge questions you can answer directly"""}
        ]

        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        for iteration in range(max_iterations):
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                # No tool call = final answer
                return {
                    "answer": assistant_message.content,
                    "iterations": iteration,
                    "messages": messages + [{"role": "assistant", "content": assistant_message.content}]
                }

            # Process tool calls
            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                result = self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        return {"answer": "Reached max iterations.", "iterations": max_iterations, "messages": messages}
```

**Usage**:
```python
agent = AgenticRAG()
agent.add_document(open("docs/policies.md").read(), source="policies.md")

# Single query
result = agent.chat("What's our refund policy?")
print(result["answer"])

# Multi-turn conversation
history = result["messages"]
result2 = agent.chat("Does that apply to enterprise customers?", history=history)
```

## Configuration Guide

### Tool Design Principles

| Principle | Good | Bad |
|-----------|------|-----|
| **Specific description** | "Search company policies and procedures" | "Search stuff" |
| **Clear when-to-use** | "Use when you need HR policy information" | (no guidance) |
| **Appropriate granularity** | Separate search vs filter tools | One mega-tool |

### Max Iterations

| Use Case | max_iterations | Rationale |
|----------|----------------|-----------|
| Simple Q&A | 3 | Quick answers |
| Research | 5-7 | Explore deeply |
| Complex analysis | 10 | Thorough investigation |

### System Prompt Tuning

```python
# Conservative agent (prefers not to search)
system_prompt = """Only search if you cannot answer from conversation context.
If unsure, ask clarifying questions before searching."""

# Thorough agent (searches more)
system_prompt = """Always verify your answers with a search.
Search multiple times if needed to ensure accuracy."""

# Balanced agent (default)
system_prompt = """Search when you need specific facts.
Don't search for general knowledge or clarifications."""
```

## Common Pitfalls

### 1. Agent Searches Too Often
**Symptom**: Searches for every question, even greetings
**Fix**: Improve system prompt, add "don't search for" examples

### 2. Agent Never Searches
**Symptom**: Makes up answers instead of searching
**Fix**: Add explicit instruction to search for factual questions

### 3. Poor Search Queries
**Symptom**: Agent uses full sentences instead of keywords
**Fix**: Add query formatting guidance in tool description

```python
{
    "description": "Search knowledge base. Query should be keywords, not full sentences. Example: 'refund policy enterprise' not 'What is the refund policy for enterprise customers?'"
}
```

### 4. Infinite Loops
**Symptom**: Agent keeps searching without answering
**Fix**: Lower max_iterations, add "answer even with incomplete info" instruction

### 5. Lost Context in Conversation
**Symptom**: Agent forgets earlier conversation
**Fix**: Ensure history is passed correctly, consider summarizing long histories

## Advanced Patterns

### Self-Critique and Retry

```python
def chat_with_critique(self, user_message: str, history: list = None) -> dict:
    """Agent that evaluates its own search results"""
    result = self.chat(user_message, history)

    # Self-critique
    critique_response = self.openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Evaluate this answer:

Question: {user_message}
Answer: {result['answer']}

Is this answer complete and accurate based on the knowledge base search?
If not, what additional search would help?

Respond with either:
- "GOOD" if the answer is satisfactory
- "SEARCH: <query>" if more searching would help"""
        }]
    )

    critique = critique_response.choices[0].message.content
    if critique.startswith("SEARCH:"):
        # Retry with suggested query
        new_query = critique.replace("SEARCH:", "").strip()
        return self.chat(f"{user_message}\n\n[Hint: search for '{new_query}']", history)

    return result
```

### Multiple Knowledge Bases

```python
# Define tools for different knowledge bases
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_product_docs",
            "description": "Search product documentation and user guides"
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policies",
            "description": "Search company policies, HR, and compliance docs"
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_support_tickets",
            "description": "Search past support tickets and solutions"
        }
    }
]
```

## When to Level Up

| Symptom | Solution | Recommendation |
|---------|----------|----------------|
| Need higher precision | Add reranking | Use `rag-toolkit:rag-with-rerank` skill |
| Complex multi-step reasoning | Structured decomposition | Use `rag-toolkit:multi-hop-rag` skill |
| Need external tools (web, APIs) | Full agent framework | Consider LangChain/LlamaIndex agents |

## References

**Internal**:
- [references/agent-patterns.md](references/agent-patterns.md) - ReAct, Tool Use, Self-RAG patterns
- [references/tool-design.md](references/tool-design.md) - Tool design best practices

**Related skills**:
- `rag-toolkit:rag` - Basic RAG (simpler, faster)
- `rag-toolkit:multi-hop-rag` - Structured multi-step reasoning

**Verticals**:
- [verticals/research-agent.md](verticals/research-agent.md) - Research assistant
