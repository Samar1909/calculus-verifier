"""Runtime configuration for the pipeline.

Credentials are loaded from a `.env` file via python-dotenv; nothing
secret is ever hardcoded here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


@dataclass
class ModelConfig:
    """One LLM endpoint to evaluate."""

    name: str
    provider: str = "auto"  # HF Router provider tag, e.g. "novita", "auto"
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass
class PipelineConfig:
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))

    # Base URL for the OpenAI-compatible chat-completions endpoint. Defaults
    # to a local Ollama server (no token/billing required); point this at
    # HF_ROUTER_BASE_URL to use Hugging Face's hosted Inference Providers.
    inference_base_url: str = OLLAMA_BASE_URL

    models: list[ModelConfig] = field(
        default_factory=lambda: [
            ModelConfig(name="qwen2.5:3b-instruct"),
        ]
    )

    # Dataset
    num_problems: int = 20
    domains: tuple[str, ...] = ("derivative", "integral")
    seed: int | None = 42

    # Feedback loop
    max_reprompt_rounds: int = 2

    # Concurrency / rate limiting
    # With GPU acceleration, we can handle multiple concurrent requests.
    # Set to 1 for CPU-only, increase to 4-8 for GPU with sufficient VRAM.
    max_concurrent_requests: int = 4
    request_timeout_s: float = 60.0  # GPU should be much faster than CPU
    max_retries: int = 4
    backoff_base_s: float = 1.5

    # Symbolic verification
    verify_timeout_s: float = 5.0

    # Vector DB failure ledger
    vector_db_path: str = "neurosym_calc/data/failure_ledger"
    vector_db_collection: str = "calc_failures"
    similar_failure_top_k: int = 1

    # Output
    results_dir: str = "neurosym_calc/data/results"
