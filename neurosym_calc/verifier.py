"""Symbolic verification of LLM-produced calculus answers.

Extracts the contents of the model's `<answer>...</answer>` tag,
normalizes common notational variants, parses it with SymPy, and
checks semantic equivalence against the ground truth via
`simplify(parsed - ground_truth) == 0`.

Simplification of deeply nested / adversarial expressions can hang,
so the actual comparison runs in a worker process with a hard
timeout; a timeout is reported as "unverifiable" rather than crashing
the caller.
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from functools import lru_cache

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    convert_xor,
    standard_transformations,
    parse_expr,
)

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

# Notational variants -> canonical SymPy-parseable form. Applied in order.
_NORMALIZATION_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\\left|\\right"), ""),  # strip LaTeX sizing
    (re.compile(r"\\cdot"), "*"),
    (re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}"), r"(\1)/(\2)"),
    (re.compile(r"\\sqrt\{([^{}]*)\}"), r"sqrt(\1)"),
    (re.compile(r"\\ln\b"), "log"),
    (re.compile(r"\\(sin|cos|tan|exp|log)\b"), r"\1"),
    # e^(...) or e^x -> exp(...) / exp(x), evaluated before generic '^' -> '**'
    (re.compile(r"\be\^\(([^()]*)\)"), r"exp(\1)"),
    (re.compile(r"\be\^([A-Za-z0-9_.]+)"), r"exp(\1)"),
    (re.compile(r"\s+"), " "),
]


class AnswerExtractionError(ValueError):
    """Raised when no <answer> tag (or unparseable content) is found."""


@dataclass
class VerificationResult:
    is_correct: bool
    parsed_answer: str | None
    raw_answer: str | None
    error: str | None = None
    timed_out: bool = False


def extract_answer(model_output: str) -> str:
    """Pulls the last `<answer>...</answer>` block out of a model response."""
    matches = _ANSWER_TAG_RE.findall(model_output)
    if not matches:
        raise AnswerExtractionError("No <answer>...</answer> tag found in model output.")
    return matches[-1].strip()


def normalize_notation(raw: str) -> str:
    """Rewrites common notational variants into a SymPy-parseable string."""
    text = raw.strip()
    for pattern, repl in _NORMALIZATION_RULES:
        text = pattern.sub(repl, text)
    return text.strip()


def _parse(text: str) -> sp.Expr:
    local_dict = {"e": sp.E, "pi": sp.pi}
    return parse_expr(text, local_dict=local_dict, transformations=_TRANSFORMATIONS)


def _compare_worker(parsed_srepr: str, ground_truth_srepr: str) -> bool:
    """Runs in a subprocess so a runaway `simplify` cannot hang the pipeline."""
    parsed = sp.sympify(parsed_srepr)
    truth = sp.sympify(ground_truth_srepr)
    diff = sp.simplify(parsed - truth)
    return diff == 0


@lru_cache(maxsize=1)
def _executor() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=2)


def verify_answer(
    model_output: str,
    ground_truth: sp.Expr,
    timeout_s: float = 5.0,
) -> VerificationResult:
    """End-to-end verification: extract -> normalize -> parse -> compare."""
    try:
        raw_answer = extract_answer(model_output)
    except AnswerExtractionError as exc:
        return VerificationResult(False, None, None, error=str(exc))

    normalized = normalize_notation(raw_answer)

    try:
        parsed = _parse(normalized)
    except Exception as exc:  # sympy parsing errors are broad by nature
        return VerificationResult(False, None, raw_answer, error=f"parse error: {exc}")

    future = _executor().submit(_compare_worker, sp.srepr(parsed), sp.srepr(ground_truth))
    try:
        is_correct = future.result(timeout=timeout_s)
        return VerificationResult(is_correct, str(parsed), raw_answer)
    except FutureTimeoutError:
        return VerificationResult(
            False, str(parsed), raw_answer, error="simplify timed out", timed_out=True
        )
    except Exception as exc:
        return VerificationResult(False, str(parsed), raw_answer, error=f"compare error: {exc}")


def shutdown_verifier_pool() -> None:
    """Call at process exit to release the worker pool cleanly."""
    if _executor.cache_info().currsize:
        _executor().shutdown(wait=False, cancel_futures=True)
