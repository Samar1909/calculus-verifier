"""Async client for open-weight chat models served via the Hugging Face
Router API (OpenAI-compatible `/chat/completions` surface).

Handles:
  * prompt construction that isolates the final answer in `<answer>` tags
  * feedback re-prompting after a failed verification
  * retry with exponential backoff on HTTP 429 / transient errors
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import aiohttp

from neurosym_calc.config import HF_ROUTER_BASE_URL, ModelConfig, PipelineConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a meticulous calculus assistant. Solve the problem step by step, "
    "showing your reasoning. On the final line, give ONLY the fully simplified "
    "symbolic result wrapped in <answer></answer> tags, e.g. <answer>3*x**2 + "
    "cos(x)</answer>. Do not include 'f'(x) =' or units inside the tags — just "
    "the expression."
)


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_payload(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResult:
    model: str
    content: str
    messages: list[ChatMessage] = field(default_factory=list)
    raw_response: dict | None = None


class RateLimitError(RuntimeError):
    pass


class InferenceEngine:
    """Thin async wrapper around the HF Router chat-completions endpoint,
    parameterized over one or more `ModelConfig`s."""

    def __init__(self, config: PipelineConfig, session: aiohttp.ClientSession) -> None:
        if config.inference_base_url == HF_ROUTER_BASE_URL and not config.hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set. Add it to a .env file (see .env.example)."
            )
        self._config = config
        self._session = session

    def build_initial_messages(self, problem_str: str) -> list[ChatMessage]:
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=problem_str),
        ]

    def build_reprompt_messages(
        self,
        messages: list[ChatMessage],
        assistant_reply: str,
        failure_reason: str,
        few_shot_hint: str | None = None,
    ) -> list[ChatMessage]:
        """Appends the model's failed attempt and a neutral correction
        request, optionally injecting a retrieved few-shot example from
        the failure ledger."""
        updated = list(messages) + [ChatMessage(role="assistant", content=assistant_reply)]

        feedback_lines = [
            "Your previous answer did not verify as symbolically correct "
            f"({failure_reason}). Please re-derive the result carefully, "
            "double-check each differentiation/integration rule you applied, "
            "and resubmit."
        ]
        if few_shot_hint:
            feedback_lines.append(
                "Here is a structurally similar problem the model previously "
                f"got wrong, along with the eventual correct reasoning, for "
                f"reference:\n{few_shot_hint}"
            )
        feedback_lines.append(
            "Give your final answer on the last line as <answer>...</answer>."
        )
        updated.append(ChatMessage(role="user", content="\n\n".join(feedback_lines)))
        return updated

    async def chat(
        self,
        model_cfg: ModelConfig,
        messages: list[ChatMessage],
    ) -> ChatResult:
        """Single chat-completion call with retry/backoff on rate limits
        and transient server errors."""
        is_hf_router = self._config.inference_base_url == HF_ROUTER_BASE_URL
        url = f"{self._config.inference_base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if is_hf_router:
            headers["Authorization"] = f"Bearer {self._config.hf_token}"

        model_name = (
            f"{model_cfg.name}:{model_cfg.provider}"
            if is_hf_router and model_cfg.provider != "auto"
            else model_cfg.name
        )
        payload = {
            "model": model_name,
            "messages": [m.to_payload() for m in messages],
            "temperature": model_cfg.temperature,
            "max_tokens": model_cfg.max_tokens,
        }

        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_s)
                async with self._session.post(
                    url, json=payload, headers=headers, timeout=timeout
                ) as resp:
                    if resp.status == 429:
                        raise RateLimitError(f"429 from {model_cfg.name}")
                    if resp.status >= 500:
                        raise RuntimeError(f"{resp.status} server error from {model_cfg.name}")
                    resp.raise_for_status()
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return ChatResult(
                        model=model_cfg.name,
                        content=content,
                        messages=messages,
                        raw_response=data,
                    )
            except (RateLimitError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                delay = self._config.backoff_base_s * (2**attempt)
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    model_cfg.name,
                    attempt + 1,
                    self._config.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Exhausted {self._config.max_retries} retries calling {model_cfg.name}"
        ) from last_exc
