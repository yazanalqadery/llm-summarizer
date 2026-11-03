import asyncio
from collections.abc import Awaitable, Callable
from llm_summarizer.models import ChatCompletion
import httpx

MODEL = "openai/gpt-4o-mini"
RETRYABLE = {429, 500, 502, 503, 504}

def summarize(text: str, api_key: str) -> ChatCompletion:
    response = httpx.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model":MODEL,
            "messages": [{"role": "user", "content": f"Summarize in 2 sentences:\n\n{text}"}],
            "max_tokens": 150
        },
        timeout=30
    )
    response.raise_for_status()

    return ChatCompletion.model_validate(response.json())

async def summarize_async(client: httpx.AsyncClient, text: str, api_key: str) -> ChatCompletion:
    response = await client.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model":MODEL,
            "messages": [{"role": "user", "content": f"Summarize in 2 sentences:\n\n{text}"}],
            "max_tokens": 150
        },
        timeout=30
    )
    response.raise_for_status()
    
    return ChatCompletion.model_validate(response.json())

async def summarize_limited(sem: asyncio.Semaphore, client: httpx.AsyncClient, text: str, api_key: str) -> ChatCompletion:
    async with sem:
        return await summarize_async(client, text, api_key)
    
def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE
    return False

async def with_retry(
    make_call: Callable[[], Awaitable[ChatCompletion]],
    max_tries: int = 3,
) -> ChatCompletion:
    for attempt in range(1, max_tries + 1):
        try:
            return await make_call()
        except Exception as e:
            if not is_retryable(e):
                raise
            if attempt == max_tries:
                raise
            await asyncio.sleep(2 ** attempt)
    raise Exception(f"Failed to make call after {max_tries} tries")

async def summarize_with_retry(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    text: str,
    api_key: str,
) -> ChatCompletion:
    return await with_retry(lambda: summarize_limited(sem, client, text, api_key), max_tries=3)