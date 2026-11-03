import asyncio
import json
import os
import ssl
import httpx
import truststore
from llm_summarizer.client import summarize_with_retry
from llm_summarizer.loader import load_docs
from llm_summarizer.models import ChatCompletion, Document, Summary

def split_results(
    docs: list[Document],
    results: list[ChatCompletion | BaseException],
) -> tuple[list[Summary], list[dict]]:
    summaries: list[Summary] = []
    failures: list[dict] = []
    for d, r in zip(docs, results):
        if isinstance(r, BaseException):
            failures.append({
                "id": d.id,
                "title": d.title,
                "error": str(r),
            })
        else:
            summaries.append(Summary(
                id=d.id,
                title=d.title,
                summary=r.choices[0].message.content,
            ))
    return summaries, failures

def save_results(summaries: list[Summary], failures: list[dict], path: str) -> None:
    data = {
        "summaries": [s.model_dump() for s in summaries],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

async def run(docs_path: str, out_path: str, max_concurrent: int = 2) -> None:
    docs = load_docs(docs_path)
    key = os.getenv("OPENROUTER_API_KEY")
    sem = asyncio.Semaphore(max_concurrent)
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    async with httpx.AsyncClient(verify=ctx) as client: 
        results = await asyncio.gather(*[summarize_with_retry(sem, client, doc.text, key) for doc in docs], return_exceptions=True)
    
    summaries, failures = split_results(docs, results)
    save_results(summaries, failures, out_path)
    print(f"{len(summaries)} summarized, {len(failures)} failed -> {out_path}")
