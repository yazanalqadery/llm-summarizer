import asyncio
import time

from dotenv import load_dotenv

from llm_summarizer.pipeline import run

load_dotenv()


async def main() -> None:
    for limit in (1, 5, 10):
        start = time.perf_counter()
        await run("bench_docs.json", "bench_out.json", max_concurrent=limit)
        elapsed = time.perf_counter() - start
        print(f"max_concurrent={limit}: {elapsed:.1f} s")


asyncio.run(main())