# Final agent evaluation

## Automated local checks

- Citation parsing returns only citations present in the Day05 RAG `Sources:` output.
- A `- (none)` RAG result routes to a clarification fallback with confidence `low`.
- Structured output validation rejects missing/invalid required fields.
- The LangGraph has a fixed 8-step limit and JSON regeneration is limited to one retry.

## Not yet verified

- Live AWS Bedrock credentials, model access, and Converse tool-use behavior.
- Answer quality against a confirmed public dataset.
- Public-data compliance: Day05 README currently lists URL `なし` and license
  `METATEAM`; this source is not confirmed to be public.

Run the CLI and Streamlit checks after configuring AWS credentials and replacing
the current retrieval corpus with a source cleared for public use. Record actual
case results and improvements here.
