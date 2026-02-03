# Fusion Strategies for Hybrid Search

Deep dive into score fusion algorithms for combining multiple search results.

## Overview

When combining results from multiple search methods (dense vectors + BM25), we need to **fuse** the rankings. The main challenge: different search methods produce scores on different scales.

| Method | Score Range | Meaning |
|--------|-------------|---------|
| Cosine Similarity | [-1, 1] | 1 = identical |
| BM25 | [0, ∞) | Higher = better match |
| L2 Distance | [0, ∞) | 0 = identical |

## Reciprocal Rank Fusion (RRF)

### How RRF Works

RRF ignores raw scores and uses only **ranks**:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Where:
- `d` = document
- `k` = constant (default 60)
- `rank_i(d)` = rank of document d in search result i

### Example Calculation

```
Document "iPhone 15 Pro":
- Dense search rank: 2
- BM25 search rank: 1

RRF_score = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325

Document "Samsung S24":
- Dense search rank: 1
- BM25 search rank: 10

RRF_score = 1/(60+1) + 1/(60+10) = 0.0164 + 0.0143 = 0.0307

Winner: "iPhone 15 Pro" (0.0325 > 0.0307)
```

### The k Parameter

`k` controls how much weight goes to lower-ranked results:

| k Value | Behavior |
|---------|----------|
| Small (k=10) | Top results dominate heavily |
| Medium (k=60) | Balanced (default) |
| Large (k=100+) | More weight to lower ranks |

```
k=10:  rank 1 gets 1/11 = 0.091, rank 10 gets 1/20 = 0.050 (1.8x diff)
k=60:  rank 1 gets 1/61 = 0.016, rank 10 gets 1/70 = 0.014 (1.1x diff)
k=100: rank 1 gets 1/101= 0.010, rank 10 gets 1/110= 0.009 (1.1x diff)
```

### When to Use RRF

✅ **Good for**:
- No prior knowledge of which search is better
- Different score scales (don't need normalization)
- Quick setup without tuning
- Generally robust performance

⚠️ **Limitations**:
- Ignores actual similarity scores
- Equal weight to all search methods
- May not be optimal if you know one method is better

### Implementation

```python
from pymilvus import RRFRanker

# Default - recommended starting point
ranker = RRFRanker(k=60)

# When top results are crucial
ranker = RRFRanker(k=20)

# When you want more diversity
ranker = RRFRanker(k=100)
```

## Weighted Fusion

### How Weighted Fusion Works

Combines normalized scores with explicit weights:

```
Final_score = w1 * norm_score1 + w2 * norm_score2
```

Where weights sum to 1: `w1 + w2 = 1`

### Score Normalization

Before weighting, scores must be normalized to [0, 1]:

```
Min-Max Normalization:
norm_score = (score - min_score) / (max_score - min_score)
```

### Example Calculation

```
Dense search (cosine similarity):
- Doc A: 0.95
- Doc B: 0.85
- Normalized: A=1.0, B=0.0 (if only these two)

BM25 search:
- Doc A: 5.2
- Doc B: 8.1
- Normalized: A=0.0, B=1.0

With weights (0.6 dense, 0.4 BM25):
- Doc A: 0.6*1.0 + 0.4*0.0 = 0.6
- Doc B: 0.6*0.0 + 0.4*1.0 = 0.4

Winner: Doc A (0.6 > 0.4)
```

### Choosing Weights

| Scenario | Weights (Dense, BM25) | Rationale |
|----------|----------------------|-----------|
| Keyword-heavy queries | (0.3, 0.7) | Users expect exact matches |
| Semantic queries | (0.7, 0.3) | Meaning matters more |
| Balanced | (0.5, 0.5) | No clear preference |
| E-commerce (SKUs) | (0.4, 0.6) | Product codes are keywords |
| Legal (semantic + terms) | (0.5, 0.5) | Both matter equally |

### When to Use Weighted

✅ **Good for**:
- Known preference for one search type
- Fine-tuning based on evaluation data
- Domain expertise about query patterns

⚠️ **Limitations**:
- Requires score normalization
- Needs weight tuning
- Sensitive to score distribution

### Implementation

```python
from pymilvus import WeightedRanker

# Favor semantic (70% dense, 30% BM25)
ranker = WeightedRanker(0.7, 0.3)

# Favor keywords (30% dense, 70% BM25)
ranker = WeightedRanker(0.3, 0.7)

# Balanced
ranker = WeightedRanker(0.5, 0.5)
```

## Comparison: RRF vs Weighted

| Aspect | RRF | Weighted |
|--------|-----|----------|
| Setup complexity | Low | Medium |
| Tuning required | Minimal (just k) | Yes (weights) |
| Score normalization | Not needed | Required |
| Interpretability | Rank-based | Score-based |
| Default performance | Good | Depends on tuning |
| When one search is clearly better | May not leverage | Can leverage |

## Advanced: Dynamic Weight Selection

For production systems, consider dynamic weights based on query analysis:

```python
def get_dynamic_weights(query: str) -> tuple:
    """Analyze query to determine best weights."""

    # Check for specific patterns
    has_numbers = any(c.isdigit() for c in query)
    has_special_terms = any(term in query.lower() for term in ['sku', 'model', 'part'])
    is_question = query.strip().endswith('?')
    word_count = len(query.split())

    # Decision logic
    if has_numbers or has_special_terms:
        # Likely looking for specific item
        return (0.3, 0.7)  # Favor keywords
    elif is_question or word_count > 5:
        # Likely semantic/conversational
        return (0.7, 0.3)  # Favor semantics
    else:
        # Default balanced
        return (0.5, 0.5)

# Usage
weights = get_dynamic_weights("iPhone 15 Pro Max 256GB")
ranker = WeightedRanker(*weights)
```

## BM25 Algorithm Overview

### What is BM25?

BM25 (Best Match 25) is a probabilistic ranking function:

```
BM25(q, d) = Σ IDF(qi) * (f(qi, d) * (k1 + 1)) / (f(qi, d) + k1 * (1 - b + b * |d|/avgdl))
```

Where:
- `IDF(qi)` = Inverse Document Frequency of term i
- `f(qi, d)` = Term frequency in document
- `|d|` = Document length
- `avgdl` = Average document length
- `k1`, `b` = Tuning parameters

### Key Properties

| Property | Meaning |
|----------|---------|
| **Term Frequency Saturation** | Multiple occurrences help, but with diminishing returns |
| **Document Length Normalization** | Longer docs don't automatically rank higher |
| **IDF Weighting** | Rare terms matter more than common terms |

### BM25 vs TF-IDF

| Aspect | BM25 | TF-IDF |
|--------|------|--------|
| TF saturation | Yes | No |
| Length normalization | Configurable | Basic |
| Performance | Generally better | Simpler |

## Evaluation Metrics

### Measuring Fusion Quality

| Metric | What it Measures | Formula |
|--------|-----------------|---------|
| **MRR** | Position of first relevant result | 1/rank |
| **MAP** | Average precision across queries | Mean of AP |
| **NDCG** | Quality of ranking with graded relevance | DCG/IDCG |

### A/B Testing Fusion Strategies

```python
import random

def ab_test_fusion(queries: list, relevance_labels: dict):
    """Compare RRF vs Weighted fusion."""

    rrf_scores = []
    weighted_scores = []

    for query in queries:
        # Get results with both methods
        rrf_results = search_with_rrf(query)
        weighted_results = search_with_weighted(query)

        # Calculate MRR for each
        rrf_mrr = calculate_mrr(rrf_results, relevance_labels[query])
        weighted_mrr = calculate_mrr(weighted_results, relevance_labels[query])

        rrf_scores.append(rrf_mrr)
        weighted_scores.append(weighted_mrr)

    print(f"RRF MRR: {sum(rrf_scores)/len(rrf_scores):.3f}")
    print(f"Weighted MRR: {sum(weighted_scores)/len(weighted_scores):.3f}")
```

## Recommendations

### Starting Point

1. **Start with RRF(k=60)** — works well without tuning
2. **Collect query logs** — understand your query patterns
3. **Measure baseline** — track MRR/NDCG
4. **Experiment with weights** — if RRF isn't optimal
5. **Consider dynamic weights** — for production systems

### Decision Tree

```
┌─────────────────────────────────────────┐
│     Which Fusion Strategy?              │
├─────────────────────────────────────────┤
│                                         │
│  Do you have evaluation data?           │
│      │                                  │
│      ├── No → Use RRF(k=60) ✓           │
│      │                                  │
│      └── Yes                            │
│           │                             │
│           └── Do you know which search  │
│               method is better?         │
│                   │                     │
│                   ├── No → Use RRF ✓    │
│                   │                     │
│                   └── Yes               │
│                        │                │
│                        └── Use Weighted │
│                            and tune ✓   │
└─────────────────────────────────────────┘
```
