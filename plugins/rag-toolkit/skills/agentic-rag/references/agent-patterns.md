# Agent Patterns for RAG

This reference covers agent architectures and patterns for building intelligent retrieval systems.

## Pattern Overview

| Pattern | Description | Best For | Complexity |
|---------|-------------|----------|------------|
| ReAct | Think → Act → Observe loop | General reasoning | Medium |
| Tool Use | Direct function calling | Clear tasks | Low |
| Self-RAG | Self-evaluate retrieval quality | High precision | High |
| Plan-and-Execute | Plan first, then execute | Complex tasks | High |

## ReAct Pattern

### Core Concept

ReAct interleaves reasoning (thinking) with actions (tool calls):

```
Question: "What's the revenue of our top customer?"

Thought 1: I need to find who our top customer is first.
Action 1: search_knowledge_base("top customer largest account")
Observation 1: "Acme Corp is our largest customer with $5M ARR..."

Thought 2: I found Acme Corp is the top customer with $5M ARR. That answers the question.
Answer: Our top customer is Acme Corp with $5M annual revenue.
```

### Implementation

```python
REACT_SYSTEM_PROMPT = """You are an assistant that thinks step-by-step.

For each question:
1. Think about what information you need
2. Use tools to gather that information
3. Observe the results
4. Decide if you need more information
5. Provide your final answer

Always explain your reasoning before taking actions."""

class ReActAgent:
    def __init__(self, rag_system, llm_client):
        self.rag = rag_system
        self.llm = llm_client
        self.tools = self._define_tools()

    def _define_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "reasoning": {"type": "string", "description": "Why you're searching for this"}
                        },
                        "required": ["query", "reasoning"]
                    }
                }
            }
        ]

    def run(self, question: str, max_steps: int = 5) -> dict:
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        trace = []

        for step in range(max_steps):
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            assistant_msg = response.choices[0].message

            if not assistant_msg.tool_calls:
                # Final answer
                return {
                    "answer": assistant_msg.content,
                    "trace": trace,
                    "steps": step + 1
                }

            # Process tool calls
            messages.append(assistant_msg)
            for tool_call in assistant_msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                trace.append({
                    "step": step + 1,
                    "action": tool_call.function.name,
                    "reasoning": args.get("reasoning", ""),
                    "query": args.get("query", "")
                })

                # Execute search
                results = self.rag.search(args["query"])
                observation = json.dumps(results, ensure_ascii=False)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": observation
                })
                trace[-1]["observation"] = results

        return {"answer": "Max steps reached", "trace": trace, "steps": max_steps}
```

## Tool Use Pattern

### Core Concept

Direct tool calling without explicit reasoning steps. Simpler and faster, but less transparent.

```python
TOOL_USE_SYSTEM_PROMPT = """You have access to a knowledge base through the search tool.
Use it when you need specific information to answer questions.
Answer directly when you have enough information."""

class ToolUseAgent:
    def run(self, question: str) -> dict:
        messages = [
            {"role": "system", "content": TOOL_USE_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        # Single pass - either tool call or direct answer
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        if response.choices[0].message.tool_calls:
            # Execute tools and get final answer
            # ... (similar to ReAct but without explicit reasoning)
            pass

        return {"answer": response.choices[0].message.content}
```

### When to Use

- Simple Q&A over documents
- Low latency requirements
- When reasoning trace isn't needed

## Self-RAG Pattern

### Core Concept

Agent evaluates its own retrieval quality and decides whether to:
1. Use the retrieved information
2. Search again with different query
3. Answer without retrieval

```
┌─────────────────────────────────────────────────────────────────┐
│                      Self-RAG Flow                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Question ──▶ Retrieve ──▶ [Relevance Check] ──▶ Generate      │
│                              │      │                           │
│                              │      ▼                           │
│                              │   Relevant?                      │
│                              │   Yes ──▶ Use for answer         │
│                              │   No  ──▶ Retry or skip          │
│                              │                                  │
│                              ▼                                  │
│                         [Support Check]                         │
│                         Is answer supported?                    │
│                         Yes ──▶ Return                          │
│                         No  ──▶ Revise                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class SelfRAGAgent:
    def __init__(self, rag_system, llm_client):
        self.rag = rag_system
        self.llm = llm_client

    def _check_relevance(self, query: str, documents: list[str]) -> list[bool]:
        """Check if each document is relevant to the query"""
        relevance = []
        for doc in documents:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"""Is this document relevant to the query?

Query: {query}
Document: {doc}

Answer only "relevant" or "irrelevant":"""
                }],
                temperature=0
            )
            relevance.append("relevant" in response.choices[0].message.content.lower())
        return relevance

    def _check_support(self, answer: str, documents: list[str]) -> bool:
        """Check if the answer is supported by the documents"""
        doc_text = "\n\n".join(documents)
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Is this answer fully supported by the documents?

Documents:
{doc_text}

Answer: {answer}

Respond "supported" or "unsupported":"""
            }],
            temperature=0
        )
        return "supported" in response.choices[0].message.content.lower()

    def run(self, question: str, max_retries: int = 2) -> dict:
        for attempt in range(max_retries + 1):
            # Retrieve
            results = self.rag.search(question, top_k=5)
            documents = [r["text"] for r in results]

            # Check relevance
            relevance = self._check_relevance(question, documents)
            relevant_docs = [d for d, r in zip(documents, relevance) if r]

            if not relevant_docs:
                # No relevant docs - try rephrasing
                question = self._rephrase_query(question)
                continue

            # Generate answer
            answer = self._generate(question, relevant_docs)

            # Check support
            if self._check_support(answer, relevant_docs):
                return {
                    "answer": answer,
                    "supported": True,
                    "attempts": attempt + 1,
                    "sources": [r["source"] for r in results if r["text"] in relevant_docs]
                }

        # Fallback
        return {
            "answer": self._generate(question, documents),
            "supported": False,
            "attempts": max_retries + 1
        }

    def _rephrase_query(self, query: str) -> str:
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Rephrase this search query using different keywords:\n{query}"
            }]
        )
        return response.choices[0].message.content

    def _generate(self, question: str, documents: list[str]) -> str:
        context = "\n\n".join(documents)
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Answer based on these documents:

{context}

Question: {question}"""
            }]
        )
        return response.choices[0].message.content
```

### When to Use

- High-stakes domains (legal, medical)
- When answer accuracy is more important than speed
- When you need confidence scores

## Plan-and-Execute Pattern

### Core Concept

Decompose complex tasks into a plan, then execute each step:

```
Question: "Compare our Q1 and Q2 revenue and explain the difference"

Plan:
1. Find Q1 revenue figures
2. Find Q2 revenue figures
3. Calculate the difference
4. Search for explanations of any changes
5. Synthesize the comparison

Execute: [Execute each step in order]

Answer: [Synthesized response]
```

### Implementation

```python
class PlanAndExecuteAgent:
    def run(self, question: str) -> dict:
        # Step 1: Create plan
        plan = self._create_plan(question)

        # Step 2: Execute each step
        results = []
        for step in plan:
            step_result = self._execute_step(step, results)
            results.append({"step": step, "result": step_result})

        # Step 3: Synthesize
        answer = self._synthesize(question, results)

        return {
            "answer": answer,
            "plan": plan,
            "execution": results
        }

    def _create_plan(self, question: str) -> list[str]:
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Create a step-by-step plan to answer this question.
Each step should be a specific action (search, calculate, compare, etc.)

Question: {question}

Plan (one step per line):"""
            }]
        )
        plan = response.choices[0].message.content.strip().split("\n")
        return [step.strip() for step in plan if step.strip()]

    def _execute_step(self, step: str, previous_results: list) -> str:
        context = "\n".join([f"Step: {r['step']}\nResult: {r['result']}" for r in previous_results])

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Execute this step using the search tool if needed.

Previous results:
{context}

Current step: {step}"""
            }],
            tools=self.tools,
            tool_choice="auto"
        )
        # ... execute and return result

    def _synthesize(self, question: str, results: list) -> str:
        execution_log = "\n\n".join([
            f"Step: {r['step']}\nResult: {r['result']}"
            for r in results
        ])

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Based on this execution, answer the original question.

Original question: {question}

Execution log:
{execution_log}

Final answer:"""
            }]
        )
        return response.choices[0].message.content
```

### When to Use

- Complex multi-step research
- Tasks with clear sequential dependencies
- When you want to show the reasoning process

## Pattern Comparison

| Pattern | Latency | Accuracy | Transparency | Use Case |
|---------|---------|----------|--------------|----------|
| Tool Use | Low | Medium | Low | Simple Q&A |
| ReAct | Medium | High | High | General reasoning |
| Self-RAG | High | Very High | Medium | High-stakes |
| Plan-Execute | High | High | Very High | Complex research |

## Combining Patterns

You can combine patterns for specific needs:

```python
class HybridAgent:
    """Combines ReAct reasoning with Self-RAG validation"""

    def run(self, question: str) -> dict:
        # Use ReAct for reasoning
        react_result = self.react_agent.run(question)

        # Use Self-RAG to validate
        if not self._validate_answer(react_result["answer"], react_result["trace"]):
            # Retry with Self-RAG
            return self.selfrag_agent.run(question)

        return react_result
```

## Best Practices

### 1. Start Simple
Begin with Tool Use, add complexity only when needed.

### 2. Log Everything
Track reasoning traces for debugging and improvement.

### 3. Set Iteration Limits
Always have max_steps/max_retries to prevent infinite loops.

### 4. Graceful Degradation
When agent can't find answer, admit it rather than hallucinate.

### 5. Test on Edge Cases
- Questions with no answer in corpus
- Ambiguous questions
- Questions requiring multiple searches
