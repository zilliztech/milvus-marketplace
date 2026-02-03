# Similarity Metrics Comparison

Understanding which similarity metric to use for vector search.

## Overview

| Metric | Formula | Range | Best For |
|--------|---------|-------|----------|
| **COSINE** | cos(θ) = A·B / (‖A‖‖B‖) | [-1, 1] | Normalized embeddings, most NLP tasks |
| **L2 (Euclidean)** | √Σ(Ai - Bi)² | [0, ∞) | Raw embeddings, image features |
| **IP (Inner Product)** | A·B = Σ(Ai × Bi) | (-∞, ∞) | Normalized vectors, recommendation |

## When to Use Each

### COSINE Similarity (Recommended Default)

**Use when:**
- Text embeddings from transformer models
- Comparing document similarity
- When magnitude shouldn't affect results

**Why it works:**
- Most embedding models normalize outputs
- Focuses on direction, not magnitude
- "How much do they point the same way?"

```python
# COSINE is the default recommendation
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",
    metric_type="COSINE"  # Range: [-1, 1], higher = more similar
)
```

### L2 (Euclidean) Distance

**Use when:**
- Image feature vectors (CNN outputs)
- When absolute distance matters
- Non-normalized embeddings

**Why it works:**
- Measures actual distance in vector space
- Sensitive to magnitude differences
- "How far apart are they?"

```python
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",
    metric_type="L2"  # Range: [0, ∞), lower = more similar
)
```

### Inner Product (IP)

**Use when:**
- Already normalized vectors (norm = 1)
- Recommendation systems
- When you need maximum performance

**Why it works:**
- Fastest computation
- Equivalent to COSINE for normalized vectors
- Often used in matrix factorization

```python
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",
    metric_type="IP"  # Range: varies, higher = more similar
)
```

## Visual Comparison

```
COSINE: Measures angle between vectors
        ↗ A
       /
      / θ (angle)
     /___________→ B
     Origin

L2: Measures straight-line distance
    A ●────────────● B
      ↑            ↑
      Distance = length of line

IP: Measures projection strength
    A ●
      │\
      │ \
      │  ↘ projection onto B
    ──┴────●────────→ B
```

## Practical Guidelines

### For Text/NLP

| Task | Recommended Metric | Reason |
|------|-------------------|--------|
| Semantic search | COSINE | Standard for text embeddings |
| Document similarity | COSINE | Direction matters more than magnitude |
| Question answering | COSINE | Semantic alignment |
| FAQ matching | COSINE | Find similar meaning |

### For Images

| Task | Recommended Metric | Reason |
|------|-------------------|--------|
| Similar image search | L2 | CNN features aren't normalized |
| Face recognition | COSINE | After normalization |
| Object detection | L2 | Feature distance |

### For Recommendations

| Task | Recommended Metric | Reason |
|------|-------------------|--------|
| User-item similarity | IP | Fast, works with normalized |
| Collaborative filtering | COSINE | Compare user preferences |

## Score Interpretation

### COSINE Scores

| Score | Interpretation |
|-------|----------------|
| > 0.95 | Nearly identical (possible duplicate) |
| 0.85 - 0.95 | Very similar |
| 0.70 - 0.85 | Related |
| 0.50 - 0.70 | Somewhat related |
| < 0.50 | Probably unrelated |

### L2 Scores (Lower = Better)

| Score | Interpretation |
|-------|----------------|
| < 0.5 | Very close |
| 0.5 - 1.0 | Similar |
| 1.0 - 2.0 | Related |
| > 2.0 | Distant |

*Note: L2 scores depend heavily on embedding dimensions*

## Converting Between Metrics

For normalized vectors (‖v‖ = 1):

```
COSINE = IP
L2² = 2 - 2 × COSINE
```

This means:
- If your vectors are normalized, COSINE and IP give equivalent rankings
- You can convert COSINE to L2 and vice versa

## Performance Considerations

| Metric | Computation Speed | Memory |
|--------|------------------|--------|
| IP | Fastest | Baseline |
| COSINE | Fast (IP + normalization) | Baseline |
| L2 | Moderate | Baseline |

**Tip**: If performance is critical and vectors are normalized, use IP instead of COSINE.

## Common Mistakes

### ❌ Using L2 with Normalized Embeddings

Most text embedding models output normalized vectors. Using L2 with normalized vectors works, but COSINE is more intuitive for score interpretation.

### ❌ Using COSINE with Unnormalized Image Features

CNN feature vectors often have varying magnitudes. The magnitude might carry meaningful information (confidence, saliency).

### ❌ Comparing Scores Across Metrics

A COSINE score of 0.8 is NOT equivalent to an L2 score of 0.8. Don't mix metrics when comparing results.

## Recommendation Summary

```
┌─────────────────────────────────────────┐
│         Which Metric Should I Use?       │
├─────────────────────────────────────────┤
│                                          │
│  Text/NLP task?                          │
│      └── Yes → COSINE ✓                  │
│                                          │
│  Image features (raw CNN)?               │
│      └── Yes → L2 ✓                      │
│                                          │
│  Normalized vectors + need speed?        │
│      └── Yes → IP ✓                      │
│                                          │
│  Not sure?                               │
│      └── COSINE (safe default) ✓         │
│                                          │
└─────────────────────────────────────────┘
```
