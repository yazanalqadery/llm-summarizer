from llm_summarizer.client import build_analysis_request, build_request


def test_analysis_request_asks_for_a_schema():
    req = build_analysis_request("hello", "KEY")
    body = req["json"]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["required"] == [
        "summary",
        "keywords",
        "sentiment",
    ]
    assert body["provider"]["require_parameters"] is True


def test_summary_request_has_no_schema():
    body = build_request("hello", "KEY")["json"]
    assert "response_format" not in body
    assert "provider" not in body