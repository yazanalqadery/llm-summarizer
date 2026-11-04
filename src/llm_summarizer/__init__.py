import argparse
import asyncio

from dotenv import load_dotenv

from llm_summarizer.pipeline import run, run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize documents with an LLM.")
    parser.add_argument("docs_path")
    parser.add_argument("out_path", nargs="?", default="summaries.json")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--mode", choices=["summarize", "analyze"], default="summarize")
    args = parser.parse_args()

    runner = run if args.mode == "summarize" else run_analysis
    load_dotenv()
    try:
        failed = asyncio.run(runner(args.docs_path, args.out_path, args.max_concurrent))
    except FileNotFoundError as e:
        print(f"Error: file not found: {e.filename}")
        raise SystemExit(1)
    if failed:
        raise SystemExit(1)            