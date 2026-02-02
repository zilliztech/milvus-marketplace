# Plagiarism/Content Spinning Detection

## Use Cases

- Academic paper plagiarism check
- News/blog content spinning detection
- Assignment plagiarism detection
- Patent similarity analysis

## Detection Strategies

### 1. Exact Plagiarism

Complete copy, can be detected via hash.

```python
import hashlib

def exact_match(content: str) -> str:
    """Calculate content hash"""
    # Preprocessing: remove whitespace, normalize case
    normalized = ''.join(content.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()
```

### 2. Synonym Substitution

Express same meaning with different words, requires semantic vector detection.

```python
# Original: Deep learning is a branch of machine learning
# Paraphrased: Deep learning belongs to the machine learning subfield
# Vector similarity will be high
```

### 3. Sentence Rearrangement

Shuffled sentence order, requires per-sentence detection.

```python
def check_sentence_plagiarism(content: str, threshold: float = 0.85):
    """Sentence-level plagiarism detection"""
    import re

    # Split into sentences
    sentences = re.split(r'[.!?]', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    plagiarized_sentences = []

    for sent in sentences:
        result = detector.check_duplicate(sent)
        if result["is_duplicate"] and result["similarity"] >= threshold:
            plagiarized_sentences.append({
                "sentence": sent,
                "similarity": result["similarity"],
                "match_source": result["match_source"]
            })

    return {
        "total_sentences": len(sentences),
        "plagiarized_count": len(plagiarized_sentences),
        "plagiarism_ratio": len(plagiarized_sentences) / len(sentences) if sentences else 0,
        "details": plagiarized_sentences
    }
```

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
import hashlib
import re

class PlagiarismDetector:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.collection_name = "document_library"
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

        # Metadata
        schema.add_field("source", DataType.VARCHAR, max_length=512)       # Source
        schema.add_field("author", DataType.VARCHAR, max_length=128)
        schema.add_field("publish_time", DataType.INT64)
        schema.add_field("doc_type", DataType.VARCHAR, max_length=32)      # paper/news/article

        index_params = self.client.prepare_index_params()
        index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index("content_hash", index_type="TRIE")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def _normalize(self, text: str) -> str:
        """Text normalization"""
        # Remove whitespace
        text = ''.join(text.split())
        # Remove punctuation
        text = re.sub(r'[.,!?;:"\'\[\]<>]', '', text)
        return text.lower()

    def _hash(self, text: str) -> str:
        return hashlib.md5(self._normalize(text).encode()).hexdigest()

    def _split_sentences(self, text: str) -> list:
        """Split into sentences"""
        sentences = re.split(r'[.!?\n]', text)
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    def check_document(self, content: str, chunk_size: int = 500) -> dict:
        """Check document for plagiarism"""
        # 1. Full document hash detection
        content_hash = self._hash(content)
        exact_match = self.client.query(
            collection_name=self.collection_name,
            filter=f'content_hash == "{content_hash}"',
            output_fields=["source", "author"],
            limit=1
        )
        if exact_match:
            return {
                "is_plagiarized": True,
                "type": "exact_copy",
                "similarity": 1.0,
                "match_source": exact_match[0]["source"],
                "match_author": exact_match[0]["author"]
            }

        # 2. Chunk detection
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        plagiarized_chunks = []

        for i, chunk in enumerate(chunks):
            embedding = self.model.encode(chunk).tolist()
            results = self.client.search(
                collection_name=self.collection_name,
                data=[embedding],
                limit=1,
                output_fields=["content", "source"]
            )

            if results[0] and results[0][0]["distance"] > 0.85:
                plagiarized_chunks.append({
                    "chunk_index": i,
                    "chunk_content": chunk[:100] + "...",
                    "similarity": results[0][0]["distance"],
                    "match_source": results[0][0]["entity"]["source"],
                    "match_content": results[0][0]["entity"]["content"][:100] + "..."
                })

        # 3. Sentence-level detection (finer granularity)
        sentences = self._split_sentences(content)
        plagiarized_sentences = []

        for sent in sentences:
            embedding = self.model.encode(sent).tolist()
            results = self.client.search(
                collection_name=self.collection_name,
                data=[embedding],
                limit=1,
                output_fields=["source"]
            )

            if results[0] and results[0][0]["distance"] > 0.9:
                plagiarized_sentences.append({
                    "sentence": sent,
                    "similarity": results[0][0]["distance"],
                    "match_source": results[0][0]["entity"]["source"]
                })

        # Calculate overall plagiarism rate
        chunk_plagiarism_ratio = len(plagiarized_chunks) / len(chunks) if chunks else 0
        sentence_plagiarism_ratio = len(plagiarized_sentences) / len(sentences) if sentences else 0

        overall_ratio = max(chunk_plagiarism_ratio, sentence_plagiarism_ratio)

        return {
            "is_plagiarized": overall_ratio > 0.3,
            "type": "partial" if overall_ratio > 0 else "original",
            "overall_similarity": overall_ratio,
            "chunk_analysis": {
                "total": len(chunks),
                "plagiarized": len(plagiarized_chunks),
                "ratio": chunk_plagiarism_ratio,
                "details": plagiarized_chunks[:5]  # Return only first 5
            },
            "sentence_analysis": {
                "total": len(sentences),
                "plagiarized": len(plagiarized_sentences),
                "ratio": sentence_plagiarism_ratio,
                "details": plagiarized_sentences[:10]  # Return only first 10
            }
        }

    def add_to_library(self, content: str, source: str, author: str = "",
                       doc_type: str = "article"):
        """Add document to library"""
        import time
        import uuid

        # Store in chunks
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]

        data = []
        for i, chunk in enumerate(chunks):
            data.append({
                "id": str(uuid.uuid4()),
                "content_hash": self._hash(chunk),
                "content": chunk,
                "embedding": self.model.encode(chunk).tolist(),
                "source": source,
                "author": author,
                "publish_time": int(time.time()),
                "doc_type": doc_type
            })

        self.client.insert(collection_name=self.collection_name, data=data)
```

## Examples

```python
detector = PlagiarismDetector()

# Add original document to library
detector.add_to_library(
    content="Deep learning is a branch of machine learning that uses multi-layer neural networks...",
    source="Introduction to AI",
    author="John Smith"
)

# Check new document
result = detector.check_document(
    "Deep learning belongs to the machine learning subfield, with its core being multi-layer neural network structure..."
)

if result["is_plagiarized"]:
    print(f"Plagiarism detected! Similarity: {result['overall_similarity']:.1%}")
    print("\nSuspicious passages:")
    for detail in result["sentence_analysis"]["details"]:
        print(f"  - {detail['sentence'][:50]}... (similarity: {detail['similarity']:.1%})")
        print(f"    Source: {detail['match_source']}")
```

## Threshold Reference

| Scenario | Sentence Threshold | Paragraph Threshold | Overall Judgment |
|----------|-------------------|---------------------|------------------|
| Academic Paper | 0.90 | 0.85 | >15% suspicious |
| News Spinning | 0.85 | 0.80 | >30% suspicious |
| Assignment Check | 0.88 | 0.82 | >20% suspicious |
