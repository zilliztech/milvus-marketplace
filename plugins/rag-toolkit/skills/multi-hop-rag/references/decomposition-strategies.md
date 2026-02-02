# Question Decomposition Strategies

This reference covers techniques for breaking complex questions into retrievable sub-questions.

## Why Decomposition Matters

Complex questions often fail with single retrieval because:

| Question Type | Problem | Solution |
|---------------|---------|----------|
| Multi-entity | "Compare A and B" | Retrieve A, then B |
| Causal chain | "How did X lead to Y?" | Trace each step |
| Aggregation | "What are all the..." | Multiple targeted searches |
| Temporal | "How has X changed?" | Search different time periods |

## Decomposition Patterns

### 1. Entity-Based Decomposition

Split questions by entities mentioned.

```
Original: "Compare the revenue of Company A and Company B in 2023"

Sub-questions:
1. "Company A revenue 2023"
2. "Company B revenue 2023"
```

**Implementation**:
```python
def entity_decompose(self, question: str) -> list[str]:
    response = self.llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Identify the main entities in this question and create a sub-question for each.

Question: {question}

Format:
Entity 1: <entity>
Sub-question 1: <question about entity 1>

Entity 2: <entity>
Sub-question 2: <question about entity 2>
..."""
        }]
    )
    # Parse response to extract sub-questions
    return self._parse_entity_response(response.choices[0].message.content)
```

### 2. Causal Chain Decomposition

Break cause-effect questions into steps.

```
Original: "How did the merger affect employee satisfaction?"

Sub-questions:
1. "What changes occurred during the merger?"
2. "What was employee satisfaction before the merger?"
3. "What was employee satisfaction after the merger?"
4. "What factors affected employee satisfaction during merger?"
```

**Implementation**:
```python
def causal_decompose(self, question: str) -> list[str]:
    response = self.llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""This question asks about cause and effect. Break it into steps:
1. What was the initial state?
2. What was the event/change?
3. What was the result?
4. What explains the connection?

Question: {question}

Sub-questions (one per line):"""
        }]
    )
    return self._parse_lines(response.choices[0].message.content)
```

### 3. Temporal Decomposition

Split questions across time periods.

```
Original: "How has our pricing strategy evolved over the past 3 years?"

Sub-questions:
1. "Pricing strategy 2022"
2. "Pricing strategy 2023"
3. "Pricing strategy 2024"
4. "Changes in pricing approach over time"
```

**Implementation**:
```python
def temporal_decompose(self, question: str, periods: list[str] = None) -> list[str]:
    if periods is None:
        # Let LLM identify time periods
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Identify the time periods relevant to this question and create a sub-question for each.

Question: {question}

Sub-questions (one per line):"""
            }]
        )
        return self._parse_lines(response.choices[0].message.content)
    else:
        # Use provided periods
        base_topic = self._extract_topic(question)
        return [f"{base_topic} {period}" for period in periods]
```

### 4. Aspect-Based Decomposition

Split questions by different aspects of a topic.

```
Original: "Evaluate our product launch strategy"

Sub-questions:
1. "Product launch marketing approach"
2. "Product launch timing and schedule"
3. "Product launch target audience"
4. "Product launch results and metrics"
```

**Implementation**:
```python
def aspect_decompose(self, question: str) -> list[str]:
    response = self.llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Identify 3-5 aspects/dimensions that should be explored to fully answer this question.

Question: {question}

Aspects and sub-questions:"""
        }]
    )
    return self._parse_lines(response.choices[0].message.content)
```

### 5. Prerequisite Decomposition

Identify what needs to be known first.

```
Original: "Is our top customer at risk of churning?"

Sub-questions (ordered):
1. "Who is our top customer?" (prerequisite)
2. "What are signs of churn risk?" (prerequisite)
3. "Recent activity of [top customer]" (depends on #1)
4. "Customer satisfaction indicators for [top customer]"
```

**Implementation**:
```python
def prerequisite_decompose(self, question: str) -> list[dict]:
    response = self.llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Break this question into sub-questions, identifying which must be answered first (prerequisites).

Question: {question}

Format each as:
Q: <sub-question>
Requires: <list of prerequisite question numbers, or "none">

Sub-questions:"""
        }]
    )

    # Returns structured list with dependencies
    return self._parse_with_dependencies(response.choices[0].message.content)
```

## Decomposition Quality

### Good Decomposition

```
Original: "What caused the decline in Q3 sales?"

Good decomposition:
1. "Q3 sales figures and trends"
2. "Q2 sales figures for comparison"
3. "Market conditions Q3"
4. "Product issues or changes Q3"
5. "Competitor activity Q3"

Why it's good:
- Each sub-question is specific and searchable
- Covers multiple possible causes
- Enables comparison (Q2 vs Q3)
```

### Bad Decomposition

```
Original: "What caused the decline in Q3 sales?"

Bad decomposition:
1. "Why did sales decline?" (too vague, same as original)
2. "Q3" (not a question)
3. "Sales problems" (too vague)

Why it's bad:
- Sub-questions aren't more specific than original
- Not formulated as searchable queries
- Missing key aspects
```

## Decomposition Strategies by Question Type

| Question Type | Strategy | Example |
|---------------|----------|---------|
| "Compare X and Y" | Entity-based | Retrieve X, retrieve Y |
| "How did X cause Y?" | Causal chain | State before, event, state after |
| "How has X changed?" | Temporal | Different time periods |
| "Evaluate X" | Aspect-based | Multiple dimensions |
| "What is X of Y?" | Prerequisite | Find Y first, then X |
| "What are all the X?" | Aggregation | Multiple targeted searches |

## Implementation: Universal Decomposer

```python
class QuestionDecomposer:
    def __init__(self, llm_client):
        self.llm = llm_client

    def decompose(self, question: str) -> list[str]:
        """Universal decomposition with strategy selection"""

        # First, identify the best strategy
        strategy = self._identify_strategy(question)

        # Then decompose using that strategy
        if strategy == "entity":
            return self._entity_decompose(question)
        elif strategy == "causal":
            return self._causal_decompose(question)
        elif strategy == "temporal":
            return self._temporal_decompose(question)
        elif strategy == "aspect":
            return self._aspect_decompose(question)
        elif strategy == "prerequisite":
            return self._prerequisite_decompose(question)
        else:
            return self._general_decompose(question)

    def _identify_strategy(self, question: str) -> str:
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Classify this question's decomposition strategy:

Question: {question}

Strategies:
- entity: Question compares or asks about multiple entities
- causal: Question asks about cause and effect
- temporal: Question asks about change over time
- aspect: Question requires evaluating multiple dimensions
- prerequisite: Question has information dependencies
- simple: Question doesn't need decomposition

Strategy (one word):"""
            }],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()

    def _general_decompose(self, question: str) -> list[str]:
        """Fallback general decomposition"""
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Break this question into 2-4 simpler sub-questions.
Each should be specific enough to retrieve relevant information.

Question: {question}

Sub-questions (one per line):"""
            }],
            temperature=0
        )
        lines = response.choices[0].message.content.strip().split("\n")
        return [line.strip().lstrip("0123456789.-) ") for line in lines if line.strip()]
```

## Validating Decomposition

### Quality Checks

```python
def validate_decomposition(self, original: str, sub_questions: list[str]) -> dict:
    """Check if decomposition is good"""

    issues = []

    # Check 1: Not too many/few
    if len(sub_questions) < 2:
        issues.append("Too few sub-questions")
    if len(sub_questions) > 6:
        issues.append("Too many sub-questions (may cause over-retrieval)")

    # Check 2: Sub-questions are different from original
    for sq in sub_questions:
        if self._similarity(original, sq) > 0.8:
            issues.append(f"Sub-question too similar to original: {sq}")

    # Check 3: Sub-questions are searchable
    for sq in sub_questions:
        if len(sq.split()) < 3:
            issues.append(f"Sub-question too short/vague: {sq}")

    # Check 4: Coverage
    coverage = self._check_coverage(original, sub_questions)
    if coverage < 0.7:
        issues.append("Sub-questions may not fully cover original question")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "sub_questions": sub_questions
    }
```

## Examples Library

### Comparison Questions

```
Q: "How does Product A compare to Product B in terms of price and features?"
→ ["Product A price and features", "Product B price and features"]

Q: "What's the difference between our Enterprise and Pro plans?"
→ ["Enterprise plan details and features", "Pro plan details and features"]
```

### Research Questions

```
Q: "What factors contributed to the success of Project X?"
→ ["Project X goals and outcomes", "Project X team and resources",
   "Project X timeline and milestones", "Project X challenges overcome"]

Q: "Why did customers prefer Competitor Y over us last quarter?"
→ ["Customer feedback about our product Q4", "Competitor Y product features",
   "Customer complaints and issues Q4", "Competitor Y pricing and offers Q4"]
```

### Tracing Questions

```
Q: "Who approved the budget for the marketing campaign that launched Product Z?"
→ ["Product Z marketing campaign", "Marketing campaign budget approval",
   "Budget approval authority"]

Q: "What company owns the subsidiary that supplies our components?"
→ ["Our component suppliers", "Supplier company ownership structure"]
```

## Quick Reference

```python
# Decomposition prompt templates

ENTITY_TEMPLATE = """Identify entities and create a sub-question for each:
Question: {question}
Sub-questions:"""

CAUSAL_TEMPLATE = """Break into cause-effect steps:
1. Initial state
2. Event/change
3. Result
Question: {question}
Sub-questions:"""

TEMPORAL_TEMPLATE = """Identify relevant time periods:
Question: {question}
Time periods and sub-questions:"""

ASPECT_TEMPLATE = """Identify aspects to explore:
Question: {question}
Aspects and sub-questions:"""

GENERAL_TEMPLATE = """Break into 2-4 simpler, searchable sub-questions:
Question: {question}
Sub-questions:"""
```
