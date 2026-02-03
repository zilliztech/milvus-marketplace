# Filter Optimization Guide

Strategies for optimizing filtered vector search performance.

## Filter Performance Factors

### 1. Selectivity

**Selectivity** = how much the filter reduces the result set.

| Selectivity | Filter Result | Impact |
|-------------|---------------|--------|
| High (>90%) | Filters out most data | ✅ Faster vector search |
| Medium (50-90%) | Moderate reduction | ⚠️ Depends on index |
| Low (<50%) | Filters little | ❌ May be slower than no filter |

**Rule of thumb**: If filter keeps >50% of data, consider if filtering is necessary.

### 2. Index Coverage

Filters on **indexed fields** are fast; filters on **non-indexed fields** require full scan.

```python
# FAST - field has index
'category == "phones"'  # category has TRIE index

# SLOW - field has no index
'description like "%wireless%"'  # description has no index
```

### 3. Filter Complexity

| Complexity | Example | Performance |
|------------|---------|-------------|
| Simple | `category == "A"` | ✅ Fast |
| Compound | `category == "A" and price > 100` | ✅ Good if both indexed |
| Complex | `(A or B) and (C or D) and not E` | ⚠️ May be slow |

## Optimization Strategies

### Strategy 1: Index Frequently Filtered Fields

```python
# Identify common filter patterns from query logs
common_filters = ["category", "price", "created_at", "status"]

# Add indexes for these fields
index_params = client.prepare_index_params()
index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index("category", index_type="TRIE")
index_params.add_index("price", index_type="STL_SORT")
index_params.add_index("status", index_type="TRIE")
```

### Strategy 2: Choose Correct Index Type

| Scenario | Index Type | Why |
|----------|------------|-----|
| `category == "X"` | TRIE | Exact match, low cardinality |
| `price >= 100 and price <= 500` | STL_SORT | Range query |
| `tags contains "X"` | INVERTED | Array membership |
| `user_id == "abc123"` | Consider no index | Very high cardinality |

### Strategy 3: Partition by High-Cardinality Fields

If you frequently filter by a high-cardinality field, consider **partitioning**:

```python
# Instead of filtering by region at query time
'region == "us-west"'

# Create partitions
client.create_partition("products", "us-west")
client.create_partition("products", "us-east")

# Insert into correct partition
client.insert("products", data, partition_name="us-west")

# Search specific partition (faster than filtering)
client.search("products", data, partition_names=["us-west"])
```

### Strategy 4: Order Conditions by Selectivity

Put the most selective condition **first** when possible:

```python
# If "status = 'archived'" matches only 1% of data
# and "category = 'electronics'" matches 30%

# BETTER - most selective first
'status == "archived" and category == "electronics"'

# WORSE - less selective first
'category == "electronics" and status == "archived"'
```

*Note: Milvus optimizer may reorder, but explicit ordering helps readability*

### Strategy 5: Avoid Expensive Operations

```python
# EXPENSIVE - string pattern matching
'description like "%wireless%"'  # Full scan

# BETTER - exact match on indexed field
'tags contains "wireless"'  # Uses INVERTED index

# EXPENSIVE - NOT with large exclusion
'category != "other"'  # If "other" is rare, this scans almost everything

# BETTER - IN with inclusion
'category in ["phones", "laptops", "tablets"]'  # Explicit inclusion
```

## Filter + Vector Search Interaction

### Pre-filter vs Post-filter

Milvus uses **pre-filtering**: filter is applied before vector search.

```
┌─────────────────────────────────────────────────┐
│  Original Collection: 1,000,000 vectors         │
│                    │                            │
│                    ▼                            │
│  Filter: category = "phones" → 50,000 vectors   │
│                    │                            │
│                    ▼                            │
│  ANN Search: find top 10 similar in 50K         │
│                    │                            │
│                    ▼                            │
│  Results: 10 items                              │
└─────────────────────────────────────────────────┘
```

### When Pre-filtering Hurts

If filter is **very selective** (e.g., matches 100 items), and you ask for top 50:
- Only 100 candidates for ANN
- Less diversity in results
- May not find truly "best" matches

**Solution**: Ensure filter matches at least 10x your limit.

### Filter Selectivity Guidelines

| Filter Matches | Recommended Action |
|----------------|-------------------|
| < limit | Warn user: "Only X items match your filters" |
| limit to 10×limit | Acceptable, but quality may vary |
| 10×limit to 100×limit | ✅ Good range |
| > 100×limit | ✅ Excellent |

## Monitoring Filter Performance

### Measure Filter Impact

```python
import time

# With filter
start = time.time()
results = client.search(collection_name, data, filter='category == "phones"', limit=10)
filtered_time = time.time() - start

# Without filter
start = time.time()
results = client.search(collection_name, data, limit=10)
unfiltered_time = time.time() - start

print(f"Filter overhead: {filtered_time - unfiltered_time:.3f}s")
```

### Check Filter Selectivity

```python
# Total count
total = client.query(collection_name, filter="", output_fields=["count(*)"])[0]["count(*)"]

# Filtered count
filtered = client.query(collection_name, filter='category == "phones"', output_fields=["count(*)"])[0]["count(*)"]

selectivity = 1 - (filtered / total)
print(f"Filter selectivity: {selectivity:.1%}")  # Higher = more selective
```

## Common Anti-patterns

### ❌ Filtering on Unindexed Text Fields

```python
# BAD - full text scan
'title like "%iPhone%"'

# GOOD - use tags or categories
'brand == "Apple" and array_contains(keywords, "iPhone")'
```

### ❌ Using Filter When Partition Works

```python
# BAD - filter on every query
results = client.search(data, filter='tenant_id == "customer_123"')

# GOOD - use partitions for tenant isolation
results = client.search(data, partition_names=["customer_123"])
```

### ❌ Complex OR Chains

```python
# BAD - many OR conditions
'cat == "a" or cat == "b" or cat == "c" or cat == "d" or ...'

# GOOD - use IN
'cat in ["a", "b", "c", "d", ...]'
```

### ❌ Negation-heavy Filters

```python
# BAD - exclude most data
'status != "active"'  # If 95% is active

# GOOD - include what you want
'status in ["inactive", "pending", "archived"]'
```

## Performance Benchmarks (Reference)

*Results vary by hardware and data distribution*

| Filter Type | Indexed | Selectivity | Latency (1M vectors) |
|-------------|---------|-------------|----------------------|
| No filter | - | 0% | ~20ms |
| category == "X" | TRIE | 90% | ~25ms |
| price > 100 | STL_SORT | 50% | ~30ms |
| text like "%X%" | None | 30% | ~200ms |
| Complex AND/OR | Mixed | 70% | ~50ms |

## Summary Checklist

Before deploying filtered search:

- [ ] Identify frequently filtered fields
- [ ] Add appropriate indexes for those fields
- [ ] Test filter selectivity with realistic data
- [ ] Monitor query latency with filters vs without
- [ ] Consider partitions for very high-cardinality filters
- [ ] Validate filter syntax with small dataset first
- [ ] Add input validation for user-provided filter values
