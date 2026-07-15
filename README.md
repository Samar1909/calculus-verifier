# neurosym-calc

A neuro-symbolic evaluation harness for testing whether LLMs can solve
single-variable calculus problems (derivatives and indefinite integrals),
self-correct when wrong, and learn from a growing memory of past mistakes.

Every problem is generated *and solved* by SymPy first, so the ground truth
is correct by construction. The LLM's answer is graded not by string match
but by parsing it back into a symbolic expression and checking
`simplify(model_answer - ground_truth) == 0` — i.e. genuine mathematical
equivalence, regardless of how the model formats or simplifies its result.

## How it works

```
ProblemGenerator  →  InferenceEngine  →  verify_answer()  →  FailureLedger
   (SymPy)          (LLM chat API)        (SymPy, sandboxed)   (Chroma vector DB)
```

1. **Problem generation** ([generator.py](neurosym_calc/generator.py))
   Builds derivative/integral problems from four rule families —
   `power`, `product`, `quotient`, `chain` — composed from a small set of
   atoms (`x`, `sin`, `cos`, `exp`, `log`, `sqrt`, polynomials). The
   ground-truth answer is derived with `sp.diff`/`sp.integrate`, so
   correctness of the *dataset* is guaranteed, not just the grading.
   Integral generation retries (bounded) until it draws an expression with
   a closed elementary antiderivative, and any single SymPy integration
   attempt is bounded by a 5s `SIGALRM` timeout so a pathological draw
   (e.g. certain `log(x)`/`sqrt(x)` products that make SymPy's Meijer-G
   path churn for minutes) can't hang the whole run.

2. **Inference** ([inference.py](neurosym_calc/inference.py))
   An async client against any OpenAI-compatible `/v1/chat/completions`
   endpoint (Hugging Face's Inference Providers router, a local Ollama
   server, etc.). The system prompt asks the model to reason step by step
   and put its final answer in `<answer>...</answer>` tags. Requests retry
   with exponential backoff on rate limits / transient errors.

3. **Verification** ([verifier.py](neurosym_calc/verifier.py))
   Extracts the `<answer>` tag, normalizes common notational variants
   (LaTeX `\frac`, `\sqrt`, `\cdot`, `e^x`, etc.) into SymPy-parseable
   text, parses it, and checks symbolic equivalence against the ground
   truth. The actual `simplify()` call runs in a worker process with a
   hard timeout — some model outputs simplify to expressions that would
   otherwise hang the comparison.

4. **Self-correction feedback loop** ([pipeline.py](neurosym_calc/pipeline.py))
   If verification fails, the failure is recorded in a persistent
   vector-DB ledger ([db_manager.py](neurosym_calc/db_manager.py), Chroma,
   keyed on a structural SymPy `srepr` fingerprint of the problem), the
   most structurally similar *already-corrected* past failure is retrieved
   as a dynamic few-shot example, and the model is re-prompted with that
   hint plus the reason its answer didn't verify. This repeats for
   `max_reprompt_rounds` (default 2) rounds per problem.

5. **Analysis** ([analysis.py](neurosym_calc/analysis.py))
   Per-model/per-domain accuracy and self-correction rate, plus a
   McNemar's test on paired (initial-correct, final-correct) outcomes to
   test whether the feedback loop significantly changes the correctness
   rate.

Everything is orchestrated by [main.py](neurosym_calc/main.py) /
[pipeline.py](neurosym_calc/pipeline.py), which runs problems concurrently
(bounded by an `asyncio.Semaphore`) and writes two CSVs per run: a
per-problem summary and a long-form per-attempt log.

## Setup

### Requirements

- Python 3.10–3.12
- An OpenAI-compatible chat-completions endpoint — either:
  - **Local (default, no cost, no API key)**: [Ollama](https://ollama.com)
    running a small instruct model, or
  - **Hosted**: a Hugging Face account with an Inference Providers token

### 1. Clone and create a virtual environment

```bash
cd ps-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

(Alternatively, `environment.yml` gives a lighter conda spec of the same
core dependencies if you'd rather not pin exact versions.)

### 2. Set up an inference backend

**Option A — local via Ollama (default, recommended if you don't want to
manage API billing):**

```bash
curl -fsSL https://ollama.com/install.sh | sh   # installs + starts the ollama service
ollama pull qwen2.5:3b-instruct                  # ~1.9GB, runs fine on CPU
```

No `.env` entry is required for this path — `PipelineConfig.inference_base_url`
already defaults to `http://localhost:11434/v1`, and `main.py --models`
defaults to `qwen2.5:3b-instruct`.

> Ollama's local server processes one request at a time. The defaults
> (`max_concurrent_requests=1`, `request_timeout_s=300`) are tuned for
> that — CPU-only generation of a full response can take 60–120+ seconds
> per problem. Raising concurrency won't speed things up locally; it will
> just make requests queue behind each other and time out.

**Option B — hosted via Hugging Face:**

1. Create a token at https://huggingface.co/settings/tokens using the
   **Inference** preset (not "Read-Only" — that lacks the "make calls to
   Inference Providers" permission and will 403).
2. Copy `.env.example` to `.env` and paste the token into `HF_TOKEN=`.
3. In code (or a small wrapper script), construct
   `PipelineConfig(inference_base_url=HF_ROUTER_BASE_URL, models=[ModelConfig(name="Qwen/Qwen2.5-7B-Instruct")])`
   — the CLI currently only exposes `--models`, so switching the base URL
   means either editing `config.py`'s default or adding your own
   `--base-url` flag.
4. Note HF's hosted inference is metered — expect `402 Payment Required`
   once free credits run out.

### 3. Run the evaluation

```bash
cd ps-project
python -m neurosym_calc.main
```

Useful flags (see `python -m neurosym_calc.main --help`):

| Flag | Default | Meaning |
|---|---|---|
| `--models` | `qwen2.5:3b-instruct` | one or more model ids to evaluate |
| `--num-problems` | 20 | total problems generated (split across `derivative`/`integral`) |
| `--max-reprompt-rounds` | 2 | additional self-correction attempts after the first miss |
| `--max-concurrent-requests` | 1 | concurrent in-flight requests |
| `--seed` | 42 | RNG seed for problem generation (reproducible datasets) |
| `--results-dir` | `neurosym_calc/data/results` | where CSVs are written |

Each run writes:
- `summary_<timestamp>.csv` — one row per problem (initial/final
  correctness, rounds attempted, whether self-correction occurred)
- `attempts_<timestamp>.csv` — one row per model attempt/round (raw
  answer, parse/verify errors, timeouts)

and prints a per-model/per-domain accuracy table plus a McNemar test
result to stdout.

The failure ledger persists across runs in
`neurosym_calc/data/failure_ledger/` (a local Chroma DB) — later runs
retrieve corrected examples from earlier runs, so accuracy on
structurally similar problems should improve over time as the ledger
fills in.

## Findings

Baseline run: `qwen2.5:3b-instruct` (a small, CPU-only, locally-hosted
model), 20 generated problems, seed 42, 2 reprompt rounds.

| Domain | n | Initial accuracy | Final accuracy | Self-correction rate | Avg. rounds |
|---|---|---|---|---|---|
| derivative | 9 | 77.8% | 88.9% | 11.1% | 1.33 |
| integral | 11 | 27.3% | 36.4% | 9.1% | 2.36 |

McNemar's test (initial vs. final correctness, pooled): stat = 0.000,
p = 0.50, n = 20.

**Interpretation:**

- **Derivatives are much easier than integrals for this model class.**
  Differentiation is a mostly mechanical, local rule application (power,
  product, quotient, chain rule); a 3B-parameter model gets close to 90%
  right after self-correction. Integration requires recognizing the
  *inverse* structure (substitution, integration by parts) and is a
  well-known weak point for small LLMs — accuracy stays under 40% even
  after retries.
- **The self-correction loop does recover some failures** — a handful of
  problems in both domains flip from wrong to right after being reprompted
  with a verification-failure reason and, once the ledger has entries, a
  retrieved worked example of a similar past mistake. E.g.
  `cos(x)/(x**3 + 3)` and `x**2` (as an integral) both self-corrected on
  round 2.
- **The McNemar result is not statistically meaningful at this sample
  size.** With only 20 paired outcomes the test has essentially no power;
  `p = 0.5` reflects too little data, not evidence that the feedback loop
  has no effect. A real test of "does self-correction help" needs on the
  order of 100+ problems per domain — this run should be read as a
  pipeline smoke test / qualitative sanity check, not a validated result.
- **The pipeline itself is validated end-to-end**: problem generation,
  ground-truth derivation, LLM inference, notation-robust symbolic
  grading, the failure ledger, and the reprompt loop all work correctly
  together, and produce results in the direction you'd expect (derivative
  ≫ integral difficulty; self-correction ≥ 0).

To get a statistically meaningful answer on whether the feedback loop
helps, rerun with a much larger `--num-problems` (expect roughly
1–2 minutes/problem on this CPU-only local setup, so budget runtime
accordingly), and/or compare against a larger model.

## Project layout

```
neurosym_calc/
├── generator.py    # SymPy-based problem + ground-truth generation
├── inference.py     # async LLM chat client (HF router / Ollama / any OpenAI-compatible API)
├── verifier.py       # <answer> extraction, notation normalization, symbolic equivalence check
├── db_manager.py    # Chroma-backed ledger of past failures + corrections
├── pipeline.py       # ties generation → inference → verification → ledger together
├── analysis.py       # accuracy summaries + McNemar significance test
├── config.py          # PipelineConfig / ModelConfig dataclasses, .env loading
├── main.py             # CLI entry point
└── data/
    ├── results/        # per-run summary/attempts CSVs
    └── failure_ledger/  # persistent Chroma vector DB
```

## Notes / known limitations

- The verifier's normalization rules handle common LaTeX/plain-text
  notational variants but are not exhaustive; a model that formats its
  answer unusually may be marked incorrect despite being mathematically
  right.
- `sp.integrate` and `sp.simplify` are both sandboxed with timeouts
  (5s each, in [generator.py](neurosym_calc/generator.py) and
  [verifier.py](neurosym_calc/verifier.py) respectively) because SymPy
  can pathologically hang on certain inputs — this trades a small amount
  of missed-but-solvable cases for guaranteed forward progress.
- Local (Ollama) inference is CPU-only in this setup and single-request
  concurrency, so large `--num-problems` runs are slow (minutes, not
  seconds) compared to a hosted API.
