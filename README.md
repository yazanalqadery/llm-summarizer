# llm-summarizer

A command-line tool that reads documents from a JSON file and processes each one with an LLM through OpenRouter: either a short text summary, or a structured analysis (summary, keywords, sentiment). It sends the requests concurrently with a configurable limit and retries temporary failures, so one bad document never costs you the rest of the batch.

## Features

- **Concurrent requests** with `asyncio` and `httpx.AsyncClient`, capped by a semaphore to stay under provider rate limits
- **Retries with exponential backoff** for timeouts, `429` and `5xx` responses (never for `4xx` mistakes like a bad key)
- **Failure isolation:** failed documents are recorded in the output file while the others are saved normally
- **Structured output:** the `analyze` mode sends a JSON schema generated from a Pydantic model, requires providers that support it, and validates the reply, so a malformed answer becomes one recorded failure
- **Validated data:** input documents and API responses are checked with Pydantic models
- **Meaningful exit codes**, so cron jobs and CI can tell when something went wrong
- **Tested:** pytest suite for the retry rules, the retry loop, the result splitting, the models and the request builders, with no network calls

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- An [OpenRouter](https://openrouter.ai) API key

## Setup

```powershell
uv sync
```

Create a file named `.env` in the project root with your key. It is ignored by Git, so the key never gets committed:

```
OPENROUTER_API_KEY=your-key-here
```

## Usage

```powershell
uv run llm-summarizer docs.json
uv run llm-summarizer docs.json out.json --max-concurrent 5
uv run llm-summarizer docs.json analysis.json --mode analyze
```

| Argument | Meaning | Default |
|---|---|---|
| `docs_path` | JSON file with the documents to process | required |
| `out_path` | where to write the results | `summaries.json` |
| `--max-concurrent` | maximum requests in flight at once | `2` |
| `--mode` | `summarize` or `analyze` | `summarize` |

**Exit codes:**

| Code | Meaning |
|---|---|
| `0` | every document was processed |
| `1` | the input file is missing (the tool prints one error line), or at least one document failed. In the second case the output file still contains the successes and a `failures` list |
| `2` | the command-line arguments are invalid |

### Modes

- **`summarize`** (default): a short text summary for each document.
- **`analyze`**: structured output. The request carries a JSON schema generated from the `Analysis` model, and the reply is validated against it. Each result has a summary, a few keywords and a sentiment label.

`analyze` sends the request with `require_parameters`, so it is only routed to providers that support structured outputs. If the chosen model has no such provider, the request fails with a `4xx`, which is recorded in `failures` and is not retried.

### Input format

A list of documents, each with an `id`, a `title` and a `text`:

```json
[
  {"id": 1, "title": "Remote work", "text": "Many companies now allow..."},
  {"id": 2, "title": "Electric cars", "text": "Electric car sales have grown..."}
]
```

### Output format

Summarize mode:

```json
{
  "summaries": [
    {"id": 1, "title": "Remote work", "summary": "..."}
  ],
  "failures": [
    {"id": 2, "title": "Electric cars", "error": "Client error '400 Bad Request' ..."}
  ]
}
```

Analyze mode:

```json
{
  "summaries": [
    {
      "id": 1,
      "title": "Remote work",
      "summary": "...",
      "keywords": ["remote work", "commuting"],
      "sentiment": "neutral"
    }
  ],
  "failures": []
}
```

## How it works

1. `loader.py` reads the input file and validates each item as a `Document`.
2. `client.py` builds one request per document. A semaphore limits how many run at once, and `with_retry` retries only the failures that `is_retryable` marks as temporary, waiting 2 s, then 4 s, between tries. In `analyze` mode, the reply text is parsed with `Analysis.model_validate_json`, so bad JSON or missing fields raise a validation error.
3. `pipeline.py` collects the results with `asyncio.gather(..., return_exceptions=True)`, splits them into successes and failures, saves the output file and returns the number of failures. `run` and `run_analysis` share one batch function and differ only in the worker and splitter they pass in.

## Benchmark

10 documents summarized with `gpt-4o-mini` in summarize mode (measured on a home connection, so your numbers will vary):

| `--max-concurrent` | Time |
|---|---|
| 1 | 15.7 s |
| 5 | 3.2 s |
| 10 | 2.1 s |

To reproduce it:

```powershell
uv run python make_bench_docs.py
uv run python compare_timing.py
```

`make_bench_docs.py` builds `bench_docs.json` by cycling through the documents in `docs.json`, and `compare_timing.py` runs the pipeline once per limit and prints the timings. Both send real requests.

## Project layout

```
src/llm_summarizer/
    __init__.py       command-line entry point
    models.py         Pydantic models (Document, Summary, AnalyzedDoc, Analysis, ChatCompletion, ...)
    loader.py         load_docs
    client.py         request builders, API calls, semaphore, retry logic
    pipeline.py       run, run_analysis, split_results, split_analyses, save_results
tests/                pytest tests (no network calls)
compare_timing.py     benchmark across concurrency limits
make_bench_docs.py    builds the benchmark input
```

## Tests

```powershell
uv run pytest tests -v
uv run ruff check src tests
```

## Configuration notes

- The model is set by the `MODEL` constant in `src/llm_summarizer/client.py` (currently `openai/gpt-4o-mini`).
- On networks that re-sign HTTPS traffic (some corporate proxies and VPNs), the tool uses the operating system's certificate store through `truststore`, so connections don't fail with `CERTIFICATE_VERIFY_FAILED`.