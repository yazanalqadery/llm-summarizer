import asyncio

from dotenv import load_dotenv

from llm_summarizer.pipeline import run


def main() -> None:
    load_dotenv()
    asyncio.run(run("docs.json", "summaries.json"))