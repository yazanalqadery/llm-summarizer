import httpx

from llm_summarizer.client import is_retryable


def make_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(code, request=request)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        return e


def test_rate_limit_is_retryable():
    assert is_retryable(make_error(429)) is True 


def test_bad_key_is_not_retryable():
    assert is_retryable(make_error(401)) is False

def test_service_unavailable_is_retryable():
    assert is_retryable(make_error(503)) is True

def test_value_error_is_not_retryable():
    assert is_retryable(ValueError("bad doc")) is False