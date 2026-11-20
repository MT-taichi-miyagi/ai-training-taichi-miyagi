# Final agent state flow

## State

- `message`, `history`, `messages`: current user request and Bedrock Converse history
- `tool_use`: validated Bedrock tool request
- `search_result`, `citations`: Day05 retrieval result and parsed source excerpts
- `answer_text`, `structured_output`, `reply`: generated and validated answer
- `errors`, `retries`, `steps`: validation errors and bounded execution counters

## Graph

```text
START
  -> request_tool (Bedrock Converse; require one registered search tool)
  -> execute_tool (Day05 answer_with_rag)
      -> no citations -> no_evidence_fallback -> END
      -> citations -> generate_answer -> validate_answer
           -> valid -> finalize -> END
           -> invalid, retry_count < 1 -> retry_answer -> generate_answer
           -> invalid, retry exhausted -> explicit RuntimeError
```

The graph has an 8-node-step limit. Any unsupported tool, invalid tool arguments,
Bedrock error, or RAG execution error fails explicitly rather than being hidden.

## Data source constraint

The current search tool reads the Day05 local corpus. Day05 README describes its
source as URL `なし` with license `METATEAM`; it is not verified as public data.
Replace it with a confirmed public corpus before treating the final assignment's
public-data requirement as complete.
