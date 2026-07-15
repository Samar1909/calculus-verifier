"""Statistical analysis helpers for pipeline result DataFrames."""

from __future__ import annotations

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_initial_vs_final(df: pd.DataFrame, model: str | None = None) -> dict:
    """Runs McNemar's test on paired (initial_correct, final_correct)
    outcomes to measure whether the feedback loop significantly changes
    the correctness rate for a given model (or all models pooled).

    Returns the 2x2 contingency table plus the test statistic/p-value.
    """
    data = df if model is None else df[df["model"] == model]

    both_correct = int(((data.initial_correct) & (data.final_correct)).sum())
    only_final = int(((~data.initial_correct) & (data.final_correct)).sum())
    only_initial = int(((data.initial_correct) & (~data.final_correct)).sum())
    neither = int(((~data.initial_correct) & (~data.final_correct)).sum())

    table = [[both_correct, only_initial], [only_final, neither]]
    result = mcnemar(table, exact=(min(only_initial, only_final) < 25))
    statistic = getattr(result, "statistic")
    p_value = getattr(result, "pvalue")

    return {
        "model": model or "all",
        "table": table,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n": len(data),
    }


def accuracy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model / per-domain initial vs. final accuracy and self-correction rate."""
    grouped = df.groupby(["model", "domain"]).agg(
        n=("problem_str", "count"),
        initial_accuracy=("initial_correct", "mean"),
        final_accuracy=("final_correct", "mean"),
        self_correction_rate=("self_corrected", "mean"),
        avg_rounds=("rounds_attempted", "mean"),
    )
    return grouped.reset_index()
