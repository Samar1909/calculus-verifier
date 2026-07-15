"""Vector-database ledger of failed model attempts.

Every time a model fails verification, we index:
  * the problem's AST (SymPy `srepr`, a structural fingerprint)
  * the model's raw reasoning trace
  * (once known) the eventual successful correction

On a later failure, we retrieve the most structurally similar past
failure + its correction and hand it back to the caller so it can be
injected into the reprompt as a dynamic few-shot example.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import chromadb
import sympy as sp

from neurosym_calc.config import PipelineConfig


def ast_fingerprint(expr: sp.Expr) -> str:
    """A whitespace-normalized `srepr` string used as the text we embed.

    `srepr` encodes the full expression tree (node types + structure),
    so problems that are structurally similar (same rule, similar
    shape) end up with lexically similar fingerprints — which is what
    the embedding-based nearest-neighbor search keys off of.
    """
    return " ".join(sp.srepr(expr).replace("(", " ( ").replace(",", " , ").split())


@dataclass
class FailureRecord:
    problem_str: str
    ast_fingerprint: str
    reasoning_trace: str
    model_answer: str
    ground_truth: str
    domain: str
    rule: str
    correction: str | None = None  # filled in once the model self-corrects


class FailureLedger:
    """Chroma-backed vector store of failed reasoning traces."""

    def __init__(self, config: PipelineConfig) -> None:
        self._client = chromadb.PersistentClient(path=config.vector_db_path)
        self._collection = self._client.get_or_create_collection(
            name=config.vector_db_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._top_k = config.similar_failure_top_k

    def add_failure(self, record: FailureRecord) -> str:
        record_id = str(uuid.uuid4())
        self._collection.add(
            ids=[record_id],
            documents=[record.ast_fingerprint],
            metadatas=[
                {
                    "problem_str": record.problem_str,
                    "reasoning_trace": record.reasoning_trace[:4000],
                    "model_answer": record.model_answer,
                    "ground_truth": record.ground_truth,
                    "domain": record.domain,
                    "rule": record.rule,
                    "correction": record.correction or "",
                }
            ],
        )
        return record_id

    def record_correction(self, record_id: str, correction: str) -> None:
        self._collection.update(
            ids=[record_id],
            metadatas=[{"correction": correction[:4000]}],
        )

    def find_similar_failure(self, fingerprint: str) -> dict | None:
        """Returns the metadata of the most structurally similar *past*
        failure that already has a recorded correction, or None."""
        if self._collection.count() == 0:
            return None
        result = self._collection.query(
            query_texts=[fingerprint],
            n_results=min(self._top_k + 5, self._collection.count()),
        )
        metadatas = result.get("metadatas", [[]])[0]
        for meta in metadatas:
            if meta.get("correction"):
                return dict(meta)
        return None

    def format_few_shot(self, meta: dict) -> str:
        return (
            f"Problem: {meta['problem_str']}\n"
            f"Incorrect first attempt: {meta['model_answer']}\n"
            f"Corrected reasoning/answer: {meta['correction']}"
        )
