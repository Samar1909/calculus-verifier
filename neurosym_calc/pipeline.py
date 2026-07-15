"""Concurrent evaluation pipeline tying generator -> inference -> verifier
-> failure ledger together, with an N-round self-correction feedback loop.

Throughput is maximized with an asyncio task queue bounded by a
semaphore; HTTP rate limits are handled by `InferenceEngine`'s
retry/backoff, and pathological `sympy.simplify` calls are isolated
behind a subprocess timeout in `verifier.verify_answer` so neither can
crash the batch job.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import aiohttp
import pandas as pd

from neurosym_calc.config import ModelConfig, PipelineConfig
from neurosym_calc.db_manager import FailureLedger, FailureRecord, ast_fingerprint
from neurosym_calc.generator import Problem, ProblemGenerator
from neurosym_calc.inference import ChatMessage, ChatResult, InferenceEngine
from neurosym_calc.verifier import VerificationResult, verify_answer

logger = logging.getLogger(__name__)


@dataclass
class AttemptRecord:
    model: str
    problem_str: str
    domain: str
    rule: str
    ground_truth: str
    round_index: int
    is_correct: bool
    raw_answer: str | None
    error: str | None
    used_few_shot_hint: bool
    timed_out: bool


@dataclass
class ProblemOutcome:
    model: str
    problem_str: str
    domain: str
    rule: str
    initial_correct: bool
    final_correct: bool
    rounds_attempted: int
    self_corrected: bool
    attempts: list[AttemptRecord] = field(default_factory=list)


class EvaluationPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.generator = ProblemGenerator(seed=self.config.seed)
        self.ledger = FailureLedger(self.config)
        self.last_outcomes: list[ProblemOutcome] = []

    async def _run_one(
        self,
        engine: InferenceEngine,
        model_cfg: ModelConfig,
        problem: Problem,
        semaphore: asyncio.Semaphore,
    ) -> ProblemOutcome:
        loop = asyncio.get_running_loop()
        fingerprint = ast_fingerprint(problem.expression)
        outcome = ProblemOutcome(
            model=model_cfg.name,
            problem_str=problem.problem_str,
            domain=problem.domain,
            rule=problem.rule,
            initial_correct=False,
            final_correct=False,
            rounds_attempted=0,
            self_corrected=False,
        )

        async with semaphore:
            messages: list[ChatMessage] = engine.build_initial_messages(problem.problem_str)
            ledger_record_id: str | None = None

            for round_index in range(self.config.max_reprompt_rounds + 1):
                try:
                    chat_result: ChatResult = await engine.chat(model_cfg, messages)
                except Exception as exc:
                    logger.error("Inference failed for %s: %s", model_cfg.name, exc)
                    outcome.attempts.append(
                        AttemptRecord(
                            model=model_cfg.name,
                            problem_str=problem.problem_str,
                            domain=problem.domain,
                            rule=problem.rule,
                            ground_truth=str(problem.ground_truth),
                            round_index=round_index,
                            is_correct=False,
                            raw_answer=None,
                            error=f"inference error: {exc}",
                            used_few_shot_hint=False,
                            timed_out=False,
                        )
                    )
                    break

                verification: VerificationResult = await loop.run_in_executor(
                    None,
                    verify_answer,
                    chat_result.content,
                    problem.ground_truth,
                    self.config.verify_timeout_s,
                )

                used_hint = False
                outcome.rounds_attempted = round_index + 1
                if round_index == 0:
                    outcome.initial_correct = verification.is_correct

                outcome.attempts.append(
                    AttemptRecord(
                        model=model_cfg.name,
                        problem_str=problem.problem_str,
                        domain=problem.domain,
                        rule=problem.rule,
                        ground_truth=str(problem.ground_truth),
                        round_index=round_index,
                        is_correct=verification.is_correct,
                        raw_answer=verification.raw_answer,
                        error=verification.error,
                        used_few_shot_hint=used_hint,
                        timed_out=verification.timed_out,
                    )
                )

                if verification.is_correct:
                    outcome.final_correct = True
                    if ledger_record_id is not None:
                        self.ledger.record_correction(ledger_record_id, chat_result.content)
                        outcome.self_corrected = True
                    break

                # Failed verification: log to ledger, retrieve a similar past
                # failure+correction as a dynamic few-shot hint, and reprompt.
                if ledger_record_id is None:
                    ledger_record_id = self.ledger.add_failure(
                        FailureRecord(
                            problem_str=problem.problem_str,
                            ast_fingerprint=fingerprint,
                            reasoning_trace=chat_result.content,
                            model_answer=verification.raw_answer or "",
                            ground_truth=str(problem.ground_truth),
                            domain=problem.domain,
                            rule=problem.rule,
                        )
                    )

                if round_index >= self.config.max_reprompt_rounds:
                    break  # no rounds left; stop without reprompting

                similar = self.ledger.find_similar_failure(fingerprint)
                few_shot_hint = self.ledger.format_few_shot(similar) if similar else None
                used_hint = few_shot_hint is not None

                failure_reason = verification.error or "answer did not match ground truth"
                messages = engine.build_reprompt_messages(
                    messages, chat_result.content, failure_reason, few_shot_hint
                )

        return outcome

    async def run(self) -> pd.DataFrame:
        problems = self.generator.generate_batch(
            self.config.num_problems, self.config.domains
        )
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
        async with aiohttp.ClientSession(connector=connector) as session:
            engine = InferenceEngine(self.config, session)
            tasks = [
                self._run_one(engine, model_cfg, problem, semaphore)
                for model_cfg in self.config.models
                for problem in problems
            ]
            outcomes: list[ProblemOutcome] = await asyncio.gather(
                *tasks, return_exceptions=False
            )

        self.last_outcomes = outcomes
        return self._to_dataframe(outcomes)

    @staticmethod
    def _to_dataframe(outcomes: list[ProblemOutcome]) -> pd.DataFrame:
        rows = []
        for outcome in outcomes:
            rows.append(
                {
                    "model": outcome.model,
                    "problem_str": outcome.problem_str,
                    "domain": outcome.domain,
                    "rule": outcome.rule,
                    "initial_correct": outcome.initial_correct,
                    "final_correct": outcome.final_correct,
                    "rounds_attempted": outcome.rounds_attempted,
                    "self_corrected": outcome.self_corrected,
                    "num_attempts_logged": len(outcome.attempts),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def attempts_dataframe(outcomes: list[ProblemOutcome]) -> pd.DataFrame:
        """Long-form table (one row per attempt/round) for finer-grained
        analysis than the per-problem summary from `run()`."""
        rows = []
        for outcome in outcomes:
            for attempt in outcome.attempts:
                rows.append(attempt.__dict__)
        return pd.DataFrame(rows)
