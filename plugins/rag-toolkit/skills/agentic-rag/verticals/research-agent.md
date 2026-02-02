# Research Assistant Agent

## Use Cases

- Literature review assistant
- Market research agent
- Technology research assistant
- Competitive analysis agent

## Core Capabilities

Research agents need to autonomously complete:
1. Understand research questions
2. Plan retrieval strategies
3. Multi-round retrieval and filtering
4. Comprehensive analysis and summarization

## Tool Definitions

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search academic papers, returns title, abstract, authors, year",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "year_from": {"type": "integer", "description": "Start year"},
                    "year_to": {"type": "integer", "description": "End year"},
                    "limit": {"type": "integer", "description": "Number of results", "default": 10}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_details",
            "description": "Get paper details including full text, citations, cited by",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper ID"}
                },
                "required": ["paper_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_citations",
            "description": "Search citations of a paper",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper ID"},
                    "direction": {"type": "string", "enum": ["cited_by", "references"], "description": "Citation direction"}
                },
                "required": ["paper_id", "direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search news reports and industry information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "source": {"type": "string", "description": "Source type: tech/finance/general"},
                    "days": {"type": "integer", "description": "Last N days", "default": 30}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_finding",
            "description": "Save research finding to notes",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Category: background/method/result/insight"},
                    "content": {"type": "string", "description": "Finding content"},
                    "source": {"type": "string", "description": "Source"}
                },
                "required": ["category", "content", "source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate research report based on saved findings",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["summary", "detailed", "slides"], "description": "Report format"}
                },
                "required": ["format"]
            }
        }
    }
]
```

## Implementation

```python
from pymilvus import MilvusClient
from openai import OpenAI
import json

class ResearchAgent:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()
        self.findings = []  # Research findings notes

    def _embed(self, text: str) -> list:
        """Generate embedding using OpenAI API"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

    def search_papers(self, query: str, year_from: int = None,
                      year_to: int = None, limit: int = 10) -> list:
        """Search papers"""
        embedding = self._embed(query).tolist()

        filters = []
        if year_from:
            filters.append(f'year >= {year_from}')
        if year_to:
            filters.append(f'year <= {year_to}')

        filter_expr = ' and '.join(filters) if filters else ""

        results = self.client.search(
            collection_name="papers",
            data=[embedding],
            filter=filter_expr,
            limit=limit,
            output_fields=["title", "abstract", "authors", "year", "venue", "citations"]
        )

        return [{
            "id": r["id"],
            "title": r["entity"]["title"],
            "abstract": r["entity"]["abstract"][:300] + "...",
            "authors": r["entity"]["authors"],
            "year": r["entity"]["year"],
            "venue": r["entity"]["venue"],
            "citations": r["entity"]["citations"]
        } for r in results[0]]

    def get_paper_details(self, paper_id: str) -> dict:
        """Get paper details"""
        paper = self.client.get(
            collection_name="papers",
            ids=[paper_id],
            output_fields=["title", "abstract", "content", "authors", "year", "references"]
        )
        if paper:
            return paper[0]
        return None

    def search_citations(self, paper_id: str, direction: str) -> list:
        """Search citation relationships"""
        if direction == "references":
            # Papers cited by this paper
            paper = self.client.get(
                collection_name="papers",
                ids=[paper_id],
                output_fields=["references"]
            )
            if paper and paper[0].get("references"):
                ref_ids = paper[0]["references"].split(",")
                return self.client.get(
                    collection_name="papers",
                    ids=ref_ids[:20],
                    output_fields=["title", "authors", "year"]
                )
        else:  # cited_by
            # Papers citing this paper
            results = self.client.query(
                collection_name="papers",
                filter=f'references like "%{paper_id}%"',
                output_fields=["title", "authors", "year"],
                limit=20
            )
            return results
        return []

    def search_news(self, query: str, source: str = None, days: int = 30) -> list:
        """Search news"""
        import time
        embedding = self._embed(query).tolist()

        cutoff = int(time.time()) - days * 24 * 3600

        filter_expr = f'publish_time >= {cutoff}'
        if source:
            filter_expr += f' and source_type == "{source}"'

        results = self.client.search(
            collection_name="news",
            data=[embedding],
            filter=filter_expr,
            limit=10,
            output_fields=["title", "summary", "source", "publish_time"]
        )

        return results[0]

    def save_finding(self, category: str, content: str, source: str):
        """Save research finding"""
        self.findings.append({
            "category": category,
            "content": content,
            "source": source
        })
        return {"status": "saved", "total_findings": len(self.findings)}

    def generate_report(self, format: str = "summary") -> str:
        """Generate research report"""
        if not self.findings:
            return "No research findings yet, please conduct research first."

        # Organize findings by category
        by_category = {}
        for f in self.findings:
            cat = f["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(f)

        # Generate report with LLM
        findings_text = ""
        for cat, items in by_category.items():
            findings_text += f"\n## {cat.upper()}\n"
            for item in items:
                findings_text += f"- {item['content']} (Source: {item['source']})\n"

        prompt = f"""Based on the following research findings, generate a {'brief' if format == 'summary' else 'detailed'} research report.

Research Findings:
{findings_text}

Requirements:
1. Clear structure including background, main findings, conclusions
2. {'Concise and clear, under 500 words' if format == 'summary' else 'Detailed discussion with data and citations'}
3. Point out limitations and future directions

Report:"""

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute tool"""
        if tool_name == "search_papers":
            results = self.search_papers(**args)
            return json.dumps(results, ensure_ascii=False)
        elif tool_name == "get_paper_details":
            result = self.get_paper_details(**args)
            return json.dumps(result, ensure_ascii=False) if result else "Paper not found"
        elif tool_name == "search_citations":
            results = self.search_citations(**args)
            return json.dumps(results, ensure_ascii=False)
        elif tool_name == "search_news":
            results = self.search_news(**args)
            return json.dumps([{"title": r["entity"]["title"], "summary": r["entity"]["summary"]}
                              for r in results], ensure_ascii=False)
        elif tool_name == "save_finding":
            result = self.save_finding(**args)
            return json.dumps(result)
        elif tool_name == "generate_report":
            return self.generate_report(**args)
        return "Unknown tool"

    def research(self, topic: str, max_iterations: int = 10) -> str:
        """Execute research task"""
        messages = [{
            "role": "system",
            "content": """You are a professional research assistant. Your task is to:
1. Understand the user's research question
2. Develop retrieval strategies
3. Use tools to search relevant papers and news
4. Analyze and filter important findings
5. Use save_finding to save key findings
6. Finally use generate_report to generate the research report

Please systematically complete the research task."""
        }, {
            "role": "user",
            "content": f"Please help me research: {topic}"
        }]

        for i in range(max_iterations):
            response = self.openai.chat.completions.create(
                model="gpt-5-mini",
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

                    print(f"[Agent] Calling tool: {tool_name}({tool_args})")

                    result = self._execute_tool(tool_name, tool_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                    # If report generated, end
                    if tool_name == "generate_report":
                        return result
            else:
                # No tool calls, agent complete
                return message.content

        return "Max iterations reached, research incomplete."
```

## Example

```python
agent = ResearchAgent()

# Execute research task
report = agent.research(
    "Latest advances in large language models for code generation"
)

print(report)
```

## Agent Behavior Log Example

```
[Agent] Calling tool: search_papers({"query": "large language model code generation", "year_from": 2023})
[Agent] Calling tool: get_paper_details({"paper_id": "paper_001"})
[Agent] Calling tool: save_finding({"category": "method", "content": "GPT-4 achieves 67% pass rate on HumanEval", "source": "OpenAI 2023"})
[Agent] Calling tool: search_news({"query": "AI code generation GitHub Copilot", "days": 90})
[Agent] Calling tool: save_finding({"category": "insight", "content": "GitHub Copilot user code acceptance rate is about 30%", "source": "GitHub Blog"})
[Agent] Calling tool: generate_report({"format": "detailed"})
```
