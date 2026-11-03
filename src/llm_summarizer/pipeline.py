import json
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