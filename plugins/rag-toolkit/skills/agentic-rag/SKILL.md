---
name: agentic-rag
description: "Use when user needs autonomous RAG agent that decides when and what to retrieve. Triggers on: agentic RAG, agent, autonomous retrieval, tool use, function calling, research agent."
---

# Agentic RAG

Agent autonomously decides when to retrieve, what to retrieve, and whether more information is needed.

## Use Cases

- Intelligent assistants (dynamic retrieval based on conversation)
- Research agents (autonomous knowledge base exploration)
- Complex task planning (step-by-step retrieval and execution)
- Open-ended Q&A (uncertain what information is needed)

## Architecture

```
User input → Agent thinks → Decide whether to retrieve → Call retrieval tool → Evaluate results → Continue/Answer
               ↑                                                                    │
               └────────────────────────────────────────────────────────────────────┘
```

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType
from openai import OpenAI
import json

class AgenticRAG:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()
        self.collection_name = "agentic_rag"
        self._init_collection()

    def _embed(self, texts: list) -> list:
        """Generate embeddings using OpenAI API"""
        if isinstance(texts, str):
            texts = [texts]
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

        # Define tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search knowledge base for relevant information. Use when you need to find specific information to answer questions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query, should be a specific question or keywords"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return, default 5",
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
                    "description": "Search within a specific source document. Use when you know information might be in a particular document.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "source": {"type": "string", "description": "Document source name"}
                        },
                        "required": ["query", "source"]
                    }
                }
            }
        ]

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 256})

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def add_document(self, text: str, source: str = ""):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_text(text)
        embeddings = self._embed(chunks)
        data = [{"text": c, "source": source, "embedding": e} for c, e in zip(chunks, embeddings)]
        self.client.insert(collection_name=self.collection_name, data=data)

    def search_knowledge_base(self, query: str, top_k: int = 5):
        """Tool: Search knowledge base"""
        embedding = self._embed(query)[0]
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=top_k,
            output_fields=["text", "source"]
        )
        return [{"text": hit["entity"]["text"], "source": hit["entity"]["source"]}
                for hit in results[0]]

    def search_by_source(self, query: str, source: str):
        """Tool: Search within specific source"""
        embedding = self._embed(query)[0]
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            filter=f'source == "{source}"',
            limit=5,
            output_fields=["text", "source"]
        )
        return [{"text": hit["entity"]["text"], "source": hit["entity"]["source"]}
                for hit in results[0]]

    def execute_tool(self, tool_name: str, arguments: dict):
        """Execute tool call"""
        if tool_name == "search_knowledge_base":
            return self.search_knowledge_base(**arguments)
        elif tool_name == "search_by_source":
            return self.search_by_source(**arguments)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def chat(self, user_message: str, history: list = None, max_iterations: int = 5):
        """Chat with agent"""
        if history is None:
            history = []

        messages = [
            {"role": "system", "content": """You are an intelligent assistant with access to a knowledge base for answering questions.

Rules:
1. If the question requires finding specific information, use the search_knowledge_base tool
2. If you know information is in a specific document, use the search_by_source tool
3. You can retrieve multiple times to get complete information
4. If retrieval results are not relevant, try different query terms
5. Synthesize all retrieved information to provide an answer
6. If knowledge base has no relevant information, clearly inform the user"""}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        iteration = 0
        while iteration < max_iterations:
            response = self.openai.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message

            # Check if tool call needed
            if assistant_message.tool_calls:
                messages.append(assistant_message)

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    # Execute tool
                    result = self.execute_tool(tool_name, arguments)

                    # Add tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })

                iteration += 1
            else:
                # No tool call, return final answer
                return {
                    "answer": assistant_message.content,
                    "iterations": iteration,
                    "messages": messages
                }

        return {
            "answer": "Reached maximum iterations, please try simplifying your question.",
            "iterations": iteration,
            "messages": messages
        }

# Usage
agent = AgenticRAG()

# Add knowledge base
agent.add_document("Milvus is an open-source vector database supporting trillion-scale vectors...", source="milvus_intro.md")
agent.add_document("Milvus 2.0 architecture includes Proxy, Coord, Worker...", source="milvus_arch.md")
agent.add_document("Creating a Collection requires defining a Schema...", source="milvus_tutorial.md")

# Agent autonomous retrieval
result = agent.chat("What is Milvus's architecture? What scale does it support?")
print(f"Answer: {result['answer']}")
print(f"Retrieval count: {result['iterations']}")

# Multi-turn conversation
history = result["messages"]
result2 = agent.chat("How do I create a Collection?", history=history)
```

## Agent Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| ReAct | Think-Act-Observe loop | Complex reasoning |
| Tool Use | Direct tool calling | Clear tasks |
| Self-RAG | Self-evaluate retrieval quality | High quality requirements |

## Tool Design Recommendations

```python
# Good tool design
{
    "name": "search_knowledge_base",
    "description": "Search knowledge base for relevant information. Use when you need to find specific information to answer questions.",
    # Clear description helps Agent decide when to use
}

# Additional tools to consider
- search_by_date: Search by time range
- search_similar: Find similar content
- get_document_summary: Get document summary
- list_sources: List all available documents
```

## Vertical Applications

See `verticals/` directory for detailed guides:
- `assistant.md` - Intelligent assistant
- `research-agent.md` - Research agent
- `task-agent.md` - Task planning agent

## Related Tools

- Basic RAG: `rag-toolkit:rag`
- Multi-hop RAG: `rag-toolkit:multi-hop-rag`
- Reranking: `core:rerank`
