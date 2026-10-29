from llm_summarizer.models import ChatCompletion
import httpx

MODEL = "openai/gpt-4o-mini"

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