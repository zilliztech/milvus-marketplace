---
name: contextual-retrieval
description: "Use when user needs parent-child document retrieval with context expansion. Triggers on: contextual retrieval, parent document, hierarchical chunking, context window, small-to-big, child chunks with parent context."
---

# Contextual Retrieval

Retrieve small chunks for precision, but return parent context for completeness. Solves the "lost in the middle" problem where matched chunks lack surrounding context.

## Use Cases

- Long document Q&A (contracts, manuals, research papers)
- When precise matching is needed but answers require broader context
- Technical documentation where code snippets need surrounding explanation
- Legal documents where clauses reference surrounding sections

## Architecture

```
Document ──→ Parent Chunks (1024 tokens)
                   │
                   ├──→ Child Chunk 1 (256 tokens) ──→ Vector Index
                   ├──→ Child Chunk 2 (256 tokens) ──→ Vector Index
                   └──→ Child Chunk 3 (256 tokens) ──→ Vector Index

Query ──→ Search Child Chunks ──→ Retrieve Parent Context ──→ LLM
```

## Complete Implementation

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

class ContextualRetrieval:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.dim = 1024
        self.collection_name = "contextual_retrieval"
        
        # Chunking configs
        self.parent_chunk_size = 1024
        self.child_chunk_size = 256
        self.chunk_overlap = 50
        
        self._init_collection()

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("child_text", DataType.VARCHAR, max_length=65535)
        schema.add_field("parent_text", DataType.VARCHAR, max_length=65535)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)  # Hash of parent
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field("child_index", DataType.INT32)  # Position within parent
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256}
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def _create_parent_chunks(self, text: str) -> list:
        """Split document into parent chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

    def _create_child_chunks(self, parent_text: str) -> list:
        """Split parent chunk into child chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=20,
            separators=["\n", ". ", " ", ""]
        )
        return splitter.split_text(parent_text)

    def add_document(self, text: str, doc_id: str = "doc"):
        """Index document with parent-child structure"""
        parent_chunks = self._create_parent_chunks(text)
        
        data = []
        for parent_text in parent_chunks:
            parent_id = hashlib.md5(parent_text.encode()).hexdigest()[:16]
            child_chunks = self._create_child_chunks(parent_text)
            
            for idx, child_text in enumerate(child_chunks):
                embedding = self.model.encode(child_text).tolist()
                data.append({
                    "child_text": child_text,
                    "parent_text": parent_text,
                    "parent_id": parent_id,
                    "doc_id": doc_id,
                    "child_index": idx,
                    "embedding": embedding
                })
        
        if data:
            self.client.insert(collection_name=self.collection_name, data=data)
        
        return len(data)

    def search(self, query: str, limit: int = 5, return_parent: bool = True):
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            limit: Number of results
            return_parent: If True, return parent context; if False, return child only
        """
        query_embedding = self.model.encode(query).tolist()
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            output_fields=["child_text", "parent_text", "parent_id", "doc_id"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}}
        )
        
        if return_parent:
            # Deduplicate by parent_id to avoid returning same context multiple times
            seen_parents = set()
            unique_results = []
            for hit in results[0]:
                parent_id = hit["entity"]["parent_id"]
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    unique_results.append({
                        "matched_chunk": hit["entity"]["child_text"],
                        "context": hit["entity"]["parent_text"],
                        "score": hit["distance"],
                        "doc_id": hit["entity"]["doc_id"]
                    })
            return unique_results
        else:
            return [{
                "text": hit["entity"]["child_text"],
                "score": hit["distance"],
                "doc_id": hit["entity"]["doc_id"]
            } for hit in results[0]]

    def search_with_context_window(self, query: str, limit: int = 3, window_size: int = 2):
        """
        Search and return surrounding parent chunks for even broader context.
        
        Args:
            query: Search query
            limit: Number of results
            window_size: Number of surrounding parents to include
        """
        # First get matching results
        results = self.search(query, limit=limit, return_parent=True)
        
        # For each result, we already have the parent context
        # This method is a placeholder for future enhancement where
        # we could retrieve neighboring parent chunks
        return results


# Usage Example
if __name__ == "__main__":
    retriever = ContextualRetrieval()
    
    # Add a long document
    document = """
    # Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn and improve from experience without being explicitly 
    programmed. It focuses on developing algorithms that can access data 
    and use it to learn for themselves.
    
    ## Supervised Learning
    
    Supervised learning is where you have input variables (X) and an output 
    variable (Y) and you use an algorithm to learn the mapping function 
    from the input to the output. The goal is to approximate the mapping 
    function so well that when you have new input data (X), you can predict 
    the output variables (Y) for that data.
    
    Common algorithms include linear regression, logistic regression, and 
    support vector machines.
    
    ## Unsupervised Learning
    
    Unsupervised learning is where you only have input data (X) and no 
    corresponding output variables. The goal is to model the underlying 
    structure or distribution in the data to learn more about the data.
    
    Common algorithms include k-means clustering and principal component 
    analysis.
    """
    
    retriever.add_document(document, doc_id="ml_intro")
    
    # Search - returns parent context even when child chunk matches
    results = retriever.search("What is supervised learning?")
    
    for r in results:
        print(f"Score: {r['score']:.3f}")
        print(f"Matched: {r['matched_chunk'][:100]}...")
        print(f"Context: {r['context'][:200]}...")
        print("---")
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| parent_chunk_size | 1024 | Token size for parent chunks (context) |
| child_chunk_size | 256 | Token size for child chunks (search targets) |
| chunk_overlap | 50 | Overlap between chunks |
| return_parent | True | Whether to return parent context or just matched chunk |

## When to Use

| Scenario | Recommendation |
|----------|---------------|
| Short documents (<2000 tokens) | Use standard semantic-search |
| Precise Q&A on long docs | Use contextual-retrieval with small child_chunk_size |
| Summarization tasks | Use standard RAG with larger chunks |
| Legal/technical docs | Use contextual-retrieval for clause + context |

## Related Skills

- Chunking strategies: `core:chunking`
- Standard semantic search: `retrieval-system:semantic-search`
- RAG pipelines: `rag-toolkit:rag`
- Reranking: `core:rerank`

## Integration with Pilot

Add to pilot routing table:

| User Intent | Scenario |
|-------------|----------|
| Parent document retrieval | `retrieval-system:contextual-retrieval` |
| Context window retrieval | `retrieval-system:contextual-retrieval` |
| Hierarchical chunking | `retrieval-system:contextual-retrieval` |
