"""
Main entry point for the llama-benchy CLI.
"""

import asyncio
import datetime
import sys
from . import __version__
from .config import BenchmarkConfig
from .corpus import TokenizedCorpus
from .prompts import PromptGenerator
from .client import LLMClient
from .runner import BenchmarkRunner
from .progress import ProgressEmitter

async def main_async():
    # Unbuffer stdout so status/progress lines show up live under `nohup` or
    # any other redirect-to-file (Python fully buffers stdout when it's not
    # a tty; this forces line buffering regardless).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    # 1. Parse Configuration
    config = BenchmarkConfig.from_args()

    # 1b. If JSONL is going to stdout, route llama-benchy's status prints to
    # stderr so they don't corrupt the JSONL stream a consumer is parsing.
    if config.emit_progress == "-":
        sys.stdout = sys.stderr

    # 2. Print Header
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"llama-benchy ({__version__})")
    print(f"Date: {current_time}")
    print(f"Benchmarking model: {config.model} at {config.base_url}")
    print(f"Concurrency levels: {config.concurrency_levels}")

    # 3. Prepare Data
    corpus = TokenizedCorpus(config.book_url, config.tokenizer, config.model)
    print(f"Total tokens available in text corpus: {len(corpus)}")

    # 4. Initialize Components
    prompt_gen = PromptGenerator(corpus)
    client = LLMClient(
        config.base_url,
        config.api_key,
        config.served_model_name,
        config.extra_body,
        config.exact_tg,
        config.no_force_length,
        config.temperature,
        config.top_p,
        config.top_k,
        config.count_reasoning,
        config.chat_template_kwargs,
    )

    progress = None
    if config.emit_progress:
        progress = ProgressEmitter(config.emit_progress, llama_benchy_version=__version__)
    runner = BenchmarkRunner(config, client, prompt_gen, progress=progress)

    # 5. Run Benchmark Suite
    status = "ok"
    try:
        await runner.run_suite()
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException:
        status = "error"
        raise
    finally:
        if progress is not None:
            try:
                progress.bench_complete(status=status)
            finally:
                progress.close()

    print(f"\nllama-benchy ({__version__})")
    print(f"date: {current_time} | latency mode: {config.latency_mode}")

def main():
    """Entry point for the CLI command."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == "__main__":
    main()
