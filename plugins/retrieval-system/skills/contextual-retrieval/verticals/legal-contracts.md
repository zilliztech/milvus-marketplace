# Contextual Retrieval for Legal Contracts

Legal documents require special handling because:

1. **Clauses reference other clauses** - "Subject to Section 5.2..."
2. **Defined terms** - Terms defined once, used throughout
3. **Context changes meaning** - Same phrase means different things in different sections

## Configuration for Legal Docs

```python
retriever = ContextualRetrieval(
    parent_chunk_size=2048,    # Full clause with preamble
    child_chunk_size=512,      # Enough for a sub-clause
    chunk_overlap=200          # High overlap for cross-references
)
```

## Example: Contract Search

```python
contract = """
ARTICLE 5 - TERMINATION

5.1 Termination for Convenience
Either party may terminate this Agreement upon thirty (30) days prior 
written notice to the other party.

5.2 Termination for Cause
Either party may terminate this Agreement immediately upon written notice 
if the other party:
(a) Materially breaches any provision of this Agreement and fails to cure 
    such breach within fifteen (15) days after receiving written notice; or
(b) Becomes insolvent or files for bankruptcy protection.

5.3 Effect of Termination
Upon termination:
(a) All licenses granted hereunder shall immediately terminate;
(b) Each party shall return or destroy Confidential Information;
(c) Sections 6, 7, and 8 shall survive termination.
"""

retriever.add_document(contract, doc_id="service_agreement")

# Query: "what happens if they go bankrupt"
# Returns: Full Article 5 context, not just "bankruptcy" match
results = retriever.search("what happens if they go bankrupt")
```

## Best Practices

1. **Preserve section numbering** - Critical for legal reference
2. **Don't split defined terms** - Keep definitions with their terms
3. **Higher parent sizes** - Legal context often spans paragraphs
