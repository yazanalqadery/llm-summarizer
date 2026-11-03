import argparse
import asyncio

from dotenv import load_dotenv

from llm_summarizer.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize documents with an LLM.")
    parser.add_argument("docs_path")
    parser.add_argument("out_path", nargs="?", default="summaries.json")
    parser.add_argument("--max-concurrent", type=int, default=2)
    args = parser.parse_args()

    load_dotenv()
    try:
        failed = asyncio.run(run(args.docs_path, args.out_path, args.max_concurrent))
    except FileNotFoundError as e:
        print(f"Error: file not found: {e.filename}")
        raise SystemExit(1)
    if failed:
        raise SystemExit(1)            