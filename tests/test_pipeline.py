from llm_summarizer.models import Analysis, ChatCompletion, Document
from llm_summarizer.pipeline import split_analyses, split_results


def make_docs() -> list[Document]:
    return [
        Document(id=1, title="A", text="x"),
        Document(id=2, title="B", text="y"),
        Document(id=3, title="C", text="z"),
    ]


def make_ok(text: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {"choices": [{"message": {"role": "assistant", "content": text}}]}
    )


def test_split_separates_successes_and_failures():
    results = [make_ok("done"), ValueError("bad doc"), make_ok("done")]
    summaries, failures = split_results(make_docs(), results)
    assert len(summaries) == 2  
    assert len(failures) == 1     


def test_failure_keeps_the_right_document_id():
    results = [make_ok("done"), ValueError("bad doc"), make_ok("done")]
    summaries, failures = split_results(make_docs(), results)
    assert failures[0]["id"] == 2
    assert [s.id for s in summaries] == [1, 3] 

def test_all_success_has_no_failures():
    results = [make_ok("done"), make_ok("done"), make_ok("done")]
    summaries, failures = split_results(make_docs(), results)
    assert len(summaries) == 3
    assert len(failures) == 0

def test_all_failures_has_no_summaries():
    results = [ValueError("bad doc"), ValueError("bad doc"), ValueError("bad doc")]
    summaries, failures = split_results(make_docs(), results)
    assert len(summaries) == 0
    assert len(failures) == 3

def make_analysis(sentiment: str = "neutral") -> Analysis:
    return Analysis(summary="s", keywords=["a", "b"], sentiment=sentiment)


def test_analyses_keep_keywords_and_sentiment():
    analyzed, failures = split_analyses(make_docs()[:1], [make_analysis("positive")])
    assert analyzed[0].keywords == ["a","b"]    
    assert analyzed[0].sentiment == "positive"     
    assert failures == []                 


def test_analysis_failure_keeps_the_document_id():
    results = [make_analysis(), ValueError("bad json"), make_analysis()]
    analyzed, failures = split_analyses(make_docs(), results)
    assert failures[0]["id"] == 2                 
    assert [a.id for a in analyzed] == [1,3]           

def test_all_analyses_failing_has_no_results():
    results = [ValueError("bad json"), ValueError("bad json"), ValueError("bad json")]
    analyzed, failures = split_analyses(make_docs(), results)
    assert analyzed == []
    assert len(failures) == 3