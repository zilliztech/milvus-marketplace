# Testing Guide

How to verify the correctness and quality of development results.

## Test Types

```
Functional Testing → Quality Evaluation → Performance Testing
```

## 1. Functional Testing

### Basic Functionality Verification

```python
def test_basic_workflow():
    """Verify basic workflow"""
    # 1. Connect
    from pymilvus import MilvusClient

    client = MilvusClient(uri="http://localhost:19530")

    # 2. Insert test data
    test_data = ["Test text 1", "Test text 2", "Test text 3"]
    embeddings = model.encode(test_data).tolist()
    data = [{"text": text, "embedding": emb} for text, emb in zip(test_data, embeddings)]
    client.insert(collection_name="test_collection", data=data)

    # 3. Search verification
    query = "test"
    query_embedding = model.encode([query]).tolist()
    results = client.search(
        collection_name="test_collection",
        data=query_embedding,
        limit=3,
        output_fields=["text"]
    )

    # 4. Assert
    assert len(results[0]) > 0, "Search should return results"
    print("✓ Basic functionality test passed")
```

### Edge Case Testing

```python
def test_edge_cases():
    """Edge cases"""
    # Empty query
    results = search("")
    assert results is not None, "Empty query should be handled"

    # Very long text
    long_text = "x" * 10000
    results = search(long_text)
    assert results is not None, "Long text should be handled"

    # Special characters
    results = search("!@#$%^&*()")
    assert results is not None, "Special characters should be handled"

    print("✓ Edge case tests passed")
```

### Error Handling Testing

```python
def test_error_handling():
    """Error handling"""
    from pymilvus import MilvusClient

    # Connection failure
    try:
        client = MilvusClient(uri="http://invalid:19530", timeout=5)
        client.list_collections()
    except Exception as e:
        print(f"✓ Connection failure handled correctly: {type(e).__name__}")

    # Invalid collection
    try:
        client = MilvusClient(uri="http://localhost:19530")
        client.query(collection_name="nonexistent_collection", filter="", limit=1)
    except Exception as e:
        print(f"✓ Invalid collection handled correctly: {type(e).__name__}")
```

## 2. Search Quality Evaluation

### Prepare Evaluation Dataset

```python
# Evaluation data format
eval_data = [
    {
        "query": "How to read files in Python",
        "relevant_ids": [101, 102, 103],  # Relevant document IDs
    },
    {
        "query": "Machine learning introduction",
        "relevant_ids": [201, 202],
    },
]
```

### Calculate Recall

```python
def evaluate_recall(eval_data, search_func, k=10):
    """Calculate Recall@K"""
    total_recall = 0

    for item in eval_data:
        results = search_func(item["query"], limit=k)
        result_ids = [r["id"] for r in results]

        relevant = set(item["relevant_ids"])
        retrieved = set(result_ids)

        recall = len(relevant & retrieved) / len(relevant)
        total_recall += recall

    avg_recall = total_recall / len(eval_data)
    print(f"Recall@{k}: {avg_recall:.2%}")
    return avg_recall
```

### Calculate Precision

```python
def evaluate_precision(eval_data, search_func, k=10):
    """Calculate Precision@K"""
    total_precision = 0

    for item in eval_data:
        results = search_func(item["query"], limit=k)
        result_ids = [r["id"] for r in results]

        relevant = set(item["relevant_ids"])
        retrieved = set(result_ids)

        precision = len(relevant & retrieved) / len(retrieved) if retrieved else 0
        total_precision += precision

    avg_precision = total_precision / len(eval_data)
    print(f"Precision@{k}: {avg_precision:.2%}")
    return avg_precision
```

### MRR (Mean Reciprocal Rank)

```python
def evaluate_mrr(eval_data, search_func, k=10):
    """Calculate MRR"""
    total_rr = 0

    for item in eval_data:
        results = search_func(item["query"], limit=k)
        result_ids = [r["id"] for r in results]

        rr = 0
        for i, rid in enumerate(result_ids):
            if rid in item["relevant_ids"]:
                rr = 1 / (i + 1)
                break
        total_rr += rr

    mrr = total_rr / len(eval_data)
    print(f"MRR: {mrr:.3f}")
    return mrr
```

### Quality Benchmarks

| Metric | Pass | Good | Excellent |
|--------|------|------|-----------|
| Recall@10 | >60% | >75% | >85% |
| Precision@10 | >40% | >60% | >75% |
| MRR | >0.3 | >0.5 | >0.7 |

## 3. Performance Testing

### Latency Testing

```python
import time

def test_latency(search_func, queries, iterations=100):
    """Test search latency"""
    latencies = []

    for _ in range(iterations):
        query = random.choice(queries)
        start = time.perf_counter()
        search_func(query)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    print(f"P50 Latency: {sorted(latencies)[50]:.1f}ms")
    print(f"P95 Latency: {sorted(latencies)[95]:.1f}ms")
    print(f"P99 Latency: {sorted(latencies)[99]:.1f}ms")
```

### Throughput Testing

```python
import concurrent.futures

def test_throughput(search_func, queries, duration=10):
    """Test QPS"""
    count = 0
    start = time.time()

    while time.time() - start < duration:
        query = random.choice(queries)
        search_func(query)
        count += 1

    qps = count / duration
    print(f"QPS: {qps:.1f}")
    return qps
```

### Concurrency Testing

```python
def test_concurrent(search_func, queries, workers=10):
    """Concurrency test"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        start = time.time()
        futures = [executor.submit(search_func, q) for q in queries[:100]]
        concurrent.futures.wait(futures)
        elapsed = time.time() - start

    print(f"{workers} concurrent, 100 requests: {elapsed:.2f}s")
```

### Performance Benchmarks

| Scenario | P95 Latency | QPS |
|----------|-------------|-----|
| Small scale (<100k) | <50ms | >100 |
| Medium scale (100k-1M) | <100ms | >50 |
| Large scale (>1M) | <200ms | >20 |

## Testing Checklist

- [ ] Basic functionality working
- [ ] Edge cases handled
- [ ] Errors properly caught
- [ ] Recall meets target
- [ ] Latency meets target
- [ ] Concurrency stable

## Next Steps

After tests pass:
- Deploy to production → `deployment.md`
- Demo showcase → `demo.md`
