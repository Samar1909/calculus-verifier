"""CLI entry point: run the evaluation pipeline and persist results."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone

from neurosym_calc.analysis import accuracy_summary, mcnemar_initial_vs_final
from neurosym_calc.config import ModelConfig, PipelineConfig
from neurosym_calc.pipeline import EvaluationPipeline
from neurosym_calc.verifier import shutdown_verifier_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neuro-symbolic calculus LLM evaluation")
    parser.add_argument(
        "--models", nargs="+", default=["qwen2.5:3b-instruct"], help="Model ids (Ollama tags by default)"
    )
    parser.add_argument("--num-problems", type=int, default=20)
    parser.add_argument("--max-reprompt-rounds", type=int, default=2)
    parser.add_argument("--max-concurrent-requests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="neurosym_calc/data/results")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    config = PipelineConfig(
        models=[ModelConfig(name=m) for m in args.models],
        num_problems=args.num_problems,
        max_reprompt_rounds=args.max_reprompt_rounds,
        max_concurrent_requests=args.max_concurrent_requests,
        seed=args.seed,
        results_dir=args.results_dir,
    )

    pipeline = EvaluationPipeline(config)
    df = await pipeline.run()

    os.makedirs(config.results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = os.path.join(config.results_dir, f"summary_{timestamp}.csv")
    attempts_path = os.path.join(config.results_dir, f"attempts_{timestamp}.csv")

    df.to_csv(summary_path, index=False)
    EvaluationPipeline.attempts_dataframe(pipeline.last_outcomes).to_csv(
        attempts_path, index=False
    )

    print(f"\nSaved per-problem summary to {summary_path}")
    print(f"Saved per-attempt log to {attempts_path}\n")
    print(accuracy_summary(df).to_string(index=False))

    for model_cfg in config.models:
        stats = mcnemar_initial_vs_final(df, model=model_cfg.name)
        print(
            f"\nMcNemar (initial vs. final) for {model_cfg.name}: "
            f"stat={stats['statistic']:.3f} p={stats['p_value']:.4f} n={stats['n']}"
        )


def main() -> None:
    try:
        asyncio.run(main_async())
    finally:
        shutdown_verifier_pool()


if __name__ == "__main__":
    main()
