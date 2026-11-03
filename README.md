# llm-summarizer

A command-line tool that reads documents from a JSON file and summarizes each one with an LLM through OpenRouter. It sends the requests concurrently with a configurable limit and retries temporary failures, so one bad document never costs you the rest of the batch.

## Features

- **Concurrent requests** with `asyncio` and `httpx.AsyncClient`, capped by a semaphore to stay under provider rate limits
- **Retries with exponential backoff** for timeouts, `429` and `5xx` responses (never for `4xx` mistakes like a bad key)
- **Failure isolation:** failed documents are recorded in the output file while the others are saved normally
- **Validated data:** input documents and API responses are checked with Pydantic models
- **Meaningful exit codes**, so cron jobs and CI can tell when something went wrong
- **Tested:** pytest suite for the retry rules, the retry loop and the result splitting, with no network calls

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
```

| Argument | Meaning | Default |
|---|---|---|
| `docs_path` | JSON file with the documents to summarize | required |
| `out_path` | where to write the results | `summaries.json` |
| `--max-concurrent` | maximum requests in flight at once | `2` |

**Exit codes:** `0` if every document was summarized. `1` if the input file is missing (the tool prints one error line) or if any document failed. In the second case the output file still contains the successful summaries and a `failures` list.

### Input format

A list of documents, each with an `id`, a `title` and a `text`:

```json
[
  {"id": 1, "title": "Remote work", "text": "Many companies now allow..."},
  {"id": 2, "title": "Electric cars", "text": "Electric car sales have grown..."}
]
```

### Output format

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

## How it works

1. `loader.py` reads the input file and validates each item as a `Document`.
2. `client.py` sends one request per document. A semaphore limits how many run at once, and `with_retry` retries only the failures that `is_retryable` marks as temporary, waiting 2 s, then 4 s, between tries.
3. `pipeline.py` collects the results with `asyncio.gather(..., return_exceptions=True)`, splits them into summaries and failures, saves the output file and returns the number of failures.

## Benchmark

10 documents summarized with `gpt-4o-mini` (measured on a home connection, so your numbers will vary):

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
    models.py         Pydantic models (Document, Summary, ChatCompletion, ...)
    loader.py         load_docs
    client.py         API calls, semaphore, retry logic
    pipeline.py       run, split_results, save_results
tests/                pytest tests (no network calls)
compare_timing.py     benchmark across concurrency limits
make_bench_docs.py    builds the benchmark input
```

## Tests

```powershell
uv run pytest tests -v
```

## Configuration notes

- The model is set by the `MODEL` constant in `src/llm_summarizer/client.py` (currently `openai/gpt-4o-mini`).
- On networks that re-sign HTTPS traffic (some corporate proxies and VPNs), the tool uses the operating system's certificate store through `truststore`, so connections don't fail with `CERTIFICATE_VERIFY_FAILED`.