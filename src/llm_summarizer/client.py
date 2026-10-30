import asyncio
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