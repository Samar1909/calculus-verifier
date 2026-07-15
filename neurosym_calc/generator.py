"""Programmatic generator for single-variable calculus problems.

Uses SymPy both to *construct* problems (by composing power/product/
quotient/chain-rule building blocks) and to derive the ground-truth
symbolic answer, so every generated item is verified correct by
construction.
"""

from __future__ import annotations

import random
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, cast

import sympy as sp


class IntegrationTimeout(Exception):
    """Raised when `sp.integrate` takes too long on a single draw."""


@contextmanager
def _time_limit(seconds: int) -> Iterator[None]:
    def _on_alarm(_signum: int, _frame: object) -> None:
        raise IntegrationTimeout

    previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

x = sp.symbols("x")

_ATOMS: list[sp.Expr] = cast(
    "list[sp.Expr]",
    [
        x,
        2 * x,
        x**2,
        x**3,
        sp.sin(x),
        sp.cos(x),
        sp.exp(x),
        sp.log(x),
        sp.sqrt(x),
    ],
)


@dataclass
class Problem:
    """A single generated problem plus its verified ground truth."""

    problem_str: str
    expression: sp.Expr
    ground_truth: sp.Expr
    domain: str  # "derivative" | "integral"
    rule: str  # e.g. "power", "product", "quotient", "chain"
    variable: sp.Symbol = field(default=x)

    def to_dict(self) -> dict:
        return {
            "problem_str": self.problem_str,
            "expression": sp.srepr(self.expression),
            "ground_truth": sp.srepr(self.ground_truth),
            "domain": self.domain,
            "rule": self.rule,
            "variable": str(self.variable),
        }


class ProblemGenerator:
    """Generates derivative and integral problems tagged by the calculus
    rule primarily required to solve them."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # -- building blocks ----------------------------------------------

    def _random_atom(self) -> sp.Expr:
        return self._rng.choice(_ATOMS)

    def _power_expr(self) -> sp.Expr:
        base = self._rng.choice([x, 2 * x, 3 * x + 1])
        power = self._rng.randint(2, 6)
        return base**power

    def _product_expr(self) -> sp.Expr:
        a, b = self._rng.sample(_ATOMS, 2)
        return a * b

    def _quotient_expr(self) -> sp.Expr:
        num, den = self._rng.sample(_ATOMS, 2)
        return num / (den + self._rng.randint(1, 3))

    def _chain_expr(self) -> sp.Expr:
        inner = self._rng.choice([x**2, 2 * x + 1, x**2 + x, 3 * x - 2])
        outer_fn: Callable[[sp.Expr], sp.Expr] = self._rng.choice(
            [sp.sin, sp.cos, sp.exp, sp.sqrt, lambda e: e**3]
        )
        return outer_fn(inner)

    _RULE_BUILDERS: dict[str, str] = {
        "power": "_power_expr",
        "product": "_product_expr",
        "quotient": "_quotient_expr",
        "chain": "_chain_expr",
    }

    def _build_expression(self, rule: str) -> sp.Expr:
        builder = getattr(self, self._RULE_BUILDERS[rule])
        return builder()

    # -- public API ------------------------------------------------------

    def generate_derivative(self, rule: str | None = None) -> Problem:
        rule = rule or self._rng.choice(list(self._RULE_BUILDERS))
        expr = self._build_expression(rule)
        ground_truth = sp.simplify(sp.diff(expr, x))
        problem_str = f"Find the derivative with respect to x of f(x) = {sp.sstr(expr)}."
        return Problem(
            problem_str=problem_str,
            expression=expr,
            ground_truth=ground_truth,
            domain="derivative",
            rule=rule,
        )

    def generate_integral(self, rule: str | None = None) -> Problem:
        """Generates an indefinite-integral problem.

        The integrand is built the same way as for derivatives, but we
        verify it is symbolically integrable by SymPy before accepting
        it (a handful of chain/quotient constructions have no closed
        elementary form).
        """
        rule = rule or self._rng.choice(list(self._RULE_BUILDERS))
        non_elementary = (
            sp.fresnelc,
            sp.fresnels,
            sp.erf,
            sp.erfc,
            sp.Si,
            sp.Ci,
            sp.li,
            sp.Ei,
        )
        for _ in range(25):  # bounded retries for non-integrable/slow draws
            expr = self._build_expression(rule)
            try:
                with _time_limit(5):
                    antiderivative = sp.integrate(expr, x)
            except IntegrationTimeout:
                continue
            if antiderivative.has(sp.Integral) or antiderivative.has(*non_elementary):
                continue
            ground_truth = sp.simplify(antiderivative)
            problem_str = (
                f"Find the indefinite integral with respect to x of f(x) = "
                f"{sp.sstr(expr)}. (Ignore the constant of integration.)"
            )
            return Problem(
                problem_str=problem_str,
                expression=expr,
                ground_truth=ground_truth,
                domain="integral",
                rule=rule,
            )
        raise RuntimeError(f"Could not generate an integrable expression for rule={rule!r}")

    def generate_batch(
        self, n: int, domains: tuple[str, ...] = ("derivative", "integral")
    ) -> list[Problem]:
        problems: list[Problem] = []
        for _ in range(n):
            domain = self._rng.choice(domains)
            rule = self._rng.choice(list(self._RULE_BUILDERS))
            if domain == "derivative":
                problems.append(self.generate_derivative(rule))
            else:
                problems.append(self.generate_integral(rule))
        return problems


if __name__ == "__main__":
    gen = ProblemGenerator(seed=1)
    for p in gen.generate_batch(5):
        print(f"[{p.domain}/{p.rule}] {p.problem_str}  ->  {p.ground_truth}")
