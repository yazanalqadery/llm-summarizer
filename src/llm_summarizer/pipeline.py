import asyncio
import json
import os
import ssl

import httpx
import truststore
from pydantic import BaseModel

from llm_summarizer.client import analyze_with_retry, summarize_with_retry
from llm_summarizer.loader import load_docs
from llm_summarizer.models import (
    Analysis,
    AnalyzedDoc,
    ChatCompletion,
    Document,
    Summary,
)


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

def split_analyses(
    docs: list[Document],
    results: list[Analysis | BaseException],
) -> tuple[list[AnalyzedDoc], list[dict]]:
    analyzed: list[AnalyzedDoc] = []
    failures: list[dict] = []
    for d, r in zip(docs, results):
        if isinstance(r, BaseException):
            failures.append({"id": d.id, "title": d.title, "error": str(r)})
        else:
            analyzed.append(
                AnalyzedDoc(
                    id=d.id,
                    title=d.title,
                    summary=r.summary,
                    keywords=r.keywords,
                    sentiment=r.sentiment,
                )
            )
    return analyzed, failures

def save_results(summaries: list[BaseModel], failures: list[dict], path: str) -> None:
    data = {
        "summaries": [s.model_dump() for s in summaries],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

async def _run_batch(
    docs_path: str,
    out_path: str,
    max_concurrent: int,
    worker,
    splitter,
    label: str,
) -> int:
    docs = load_docs(docs_path)
    key = os.environ["OPENROUTER_API_KEY"]
    sem = asyncio.Semaphore(max_concurrent)
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    async with httpx.AsyncClient(verify=ctx) as client:
        results = await asyncio.gather(
            *[worker(sem, client, d.text, key) for d in docs],
            return_exceptions=True,
        )

    done, failures = splitter(docs, results)
    save_results(done, failures, out_path)
    print(f"{len(done)} {label}, {len(failures)} failed -> {out_path}")
    return len(failures)

async def run(docs_path: str, out_path: str, max_concurrent: int = 2) -> int:
    return await _run_batch(
        docs_path, out_path, max_concurrent, summarize_with_retry, split_results, "summarized"
    )


async def run_analysis(docs_path: str, out_path: str, max_concurrent: int = 2) -> int:
    return await _run_batch(
        docs_path, out_path, max_concurrent, analyze_with_retry, split_analyses, "analyzed"
    )
                                                  