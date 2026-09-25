# Support Assistant

## Architecture

```text
8 policy documents
      |
      v
ingestion/chunking
      |
      v
all-MiniLM-L6-v2
      |
      v
ChromaDB
      |
      v
FastAPI /ask
      |
      v
LangGraph classify_intent
      |
      +--------------------+
      |                    |
policy_question      general_question
      |                    |
      v                    v
retrieve_and_answer   direct_answer
      |                    |
      +---------+----------+
                |
                v
       Pydantic response
       answer/sources/confidence
```

`ingest()` handles document ingestion and embedding. ChromaDB stores the vectors. `classify_intent` performs the required keyword routing in mock mode. `retrieve_and_answer` retrieves the top three chunks and creates the deterministic mock answer from the most similar chunk. `direct_answer` produces the fixed general-question response.

## Mock mode

The graded baseline is deterministic and requires no LLM API. `MOCK_LLM` can remain unset or be set to `1`.

The optional real-LLM extension can be layered into the generation functions when `MOCK_LLM=0`; it must not be required for the baseline.

## Example calls

Policy query:

```json
{
  "query": "What is the delivery fee?"
}
```

General query:

```json
{
  "query": "What is your favorite color?"
}
```

Run locally:

```bash
uvicorn support_assistant.main:app --reload
```

Run Docker:

```bash
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```

The structured policy prompt is connected to the deterministic retrieval execution path.
