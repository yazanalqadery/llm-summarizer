import asyncio

import httpx
import pytest

from llm_summarizer import client
from llm_summarizer.client import with_retry


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(client.asyncio, "sleep", fake_sleep)


def make_flaky(fail_times: int, error: Exception):
    calls = {"n": 0}

    async def call():
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise error
        return "ok"

    return call, calls


def test_succeeds_after_two_timeouts():
    call, calls = make_flaky(2, httpx.ReadTimeout("slow"))
    result = asyncio.run(with_retry(call, max_tries=3))
    assert result == "ok"
    assert calls["n"] == 3   


def test_value_error_is_not_retried():
    call, calls = make_flaky(5, ValueError("my mistake"))
    with pytest.raises(ValueError):
        asyncio.run(with_retry(call, max_tries=3))
    assert calls["n"] == 1

def test_gives_up_after_max_tries():
    call, calls = make_flaky(10, httpx.ReadTimeout("slow"))
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(with_retry(call, max_tries=2))
    assert calls["n"] == 2

def test_zero_tries_raises_runtime_error():
    call, calls = make_flaky(0, ValueError("x"))
    with pytest.raises(RuntimeError):
        asyncio.run(with_retry(call, max_tries=0))
    assert calls["n"] == 0