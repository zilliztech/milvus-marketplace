# Contextual Retrieval for Technical Documentation

Technical documentation (API references, manuals, tutorials) benefits significantly from contextual retrieval because:

1. **Code snippets need explanation** - A matched code block without surrounding prose is often useless
2. **Parameters reference other sections** - API params often link to type definitions elsewhere
3. **Step-by-step instructions** - Matching one step without context breaks the workflow

## Configuration for Technical Docs

```python
retriever = ContextualRetrieval(
    parent_chunk_size=1500,    # Larger to capture full code blocks
    child_chunk_size=300,      # Small for precise matching
    chunk_overlap=100          # Higher overlap for code
)
```

## Example: API Documentation

```python
api_doc = """
## Authentication

All API requests require a Bearer token in the Authorization header.

### Getting a Token

POST /api/v1/auth/token

Request body:
```json
{
  "client_id": "your_client_id",
  "client_secret": "your_client_secret"
}
```

Response:
```json
{
  "access_token": "eyJhbGc...",
  "expires_in": 3600,
  "token_type": "Bearer"
}
```

### Using the Token

Include the token in all subsequent requests:

```bash
curl -H "Authorization: Bearer eyJhbGc..." https://api.example.com/v1/users
```

Tokens expire after 1 hour. Refresh before expiry to avoid interruption.
"""

retriever.add_document(api_doc, doc_id="api_auth")

# Query: "how to authenticate"
# Returns: The full authentication section (parent) even though
#          only "Bearer token" child chunk matched
results = retriever.search("how to authenticate")
```

## Best Practices

1. **Pre-process code blocks** - Ensure code blocks aren't split mid-function
2. **Use markdown-aware chunking** - Split on headers when possible
3. **Index multiple granularities** - Sometimes users want just the code, sometimes the explanation
