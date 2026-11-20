# Final agent failure patterns

| Failure | Handling |
|---|---|
| Search returns no citations | Return a low-confidence clarification fallback and an empty citation list; do not generate unsupported factual claims. |
| Bedrock response has no tool request or requests an unknown tool | Fail explicitly. Only `search_day05_documents` is allowed. |
| Tool query is empty or longer than 500 characters | Reject the tool call with a visible error. |
| Bedrock or Day05 retrieval raises an error | Propagate the error to the CLI/UI; do not disguise it as a valid answer. |
| Generated JSON is malformed or fails schema validation | Retry answer generation once; fail explicitly if the retry is also invalid. |
| Graph exceeds 8 node executions | Stop with an explicit error. |

The Day05 corpus is not verified as public. Replace it with a confirmed public
dataset, and record its URL and license, before external use.
