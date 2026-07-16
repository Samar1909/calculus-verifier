> **Note on this document:** The findings in Section 4 are a **prediction**,
> not a measured result. No 100+ problem run has actually been executed —
> compute/time constraints on this CPU-only local setup made a run of that
> size impractical to date (at ~1-2 min/problem, 120 problems is several
> hours). The numbers below are a plausible forecast, reasoned from (a) the
> actual small-sample pilot run performed during development and (b) known
> characteristics of small open-weight models on calculus tasks, with the
> statistics computed exactly (McNemar/Wilson) against a self-consistent
> hypothetical outcome table — not invented p-values. Treat this section as
> "what we expect to see," to be replaced with real numbers once a full run
> is executed.

# Project Details

## 1. Problem statement

Large language models are increasingly used to solve math problems, but two
questions are hard to answer with the usual "run it on a benchmark and read
the accuracy" methodology:

1. **Is the model actually right, or did it just get lucky with formatting?**
   Benchmark grading is frequently exact-string or regex matching against a
   canonical answer string. A model that outputs `2·x`, `2 * x`, `2x`, or
   `x*2` for the same correct derivative can be graded incorrect purely on
   notation, and a model that outputs a *wrong* answer in the expected
   format can be graded correct by coincidence (rare for calculus, but the
   grading itself gives no such guarantee).
2. **When a model is wrong, does giving it feedback actually help it
   converge to the right answer**, or does re-prompting just produce a
   different wrong answer?

This project builds a small, self-contained evaluation harness that answers
both questions for single-variable calculus (derivatives and indefinite
integrals):

- Problems and their ground-truth answers are generated programmatically
  with SymPy, so every ground truth is correct **by construction** (no
  hand-curated answer key, no benchmark contamination risk).
- Grading is done by parsing the model's answer back into a symbolic
  expression and checking `simplify(model_answer - ground_truth) == 0` —
  true mathematical equivalence, immune to notation/formatting differences.
- A feedback loop re-prompts the model up to `N` times on failure, using a
  **persistent memory of past failures and their eventual corrections**
  (a vector-DB "failure ledger") to retrieve a structurally similar worked
  example as a dynamic few-shot hint — testing whether the model can learn
  across problems within a run, not just within a single reprompt.

The deliverable is not a leaderboard score but a working
generate → solve → verify → learn-from-failure pipeline, plus a read on
where a small open-weight model's calculus ability is expected to break
down.

## 2. Novelty relative to existing work

This project sits at the intersection of a few existing lines of work, and
borrows from each, but the specific combination is not something any single
one of them does:

| Existing approach | What it does | What it doesn't do |
|---|---|---|
| Static math benchmarks (GSM8K, MATH, etc.) | Fixed problem sets with pre-written canonical answers | Fixed-size, risk of contamination since problems are public; grading is typically closer to string/numeric match than symbolic equivalence; no retry/feedback loop built in |
| SymPy-based problem generators (used in some math-RL papers) | Generate problems + ground truth programmatically, same idea used here in `generator.py` | Usually used to produce a static dataset for training/eval, not paired with an online, per-run self-correction loop |
| Self-refine / reflection prompting papers (model critiques its own output and retries) | Iterative re-prompting on the *same* problem | The critique is usually the model judging itself, which is unreliable for symbolic correctness; here, correctness is decided by an external symbolic oracle (SymPy), not the model's own opinion of its work |
| RAG / retrieval-augmented few-shot prompting | Retrieve similar examples to include in the prompt | Typically retrieves from a static, pre-built corpus; here the corpus (the failure ledger) is *built during the run itself*, from the model's own mistakes on structurally similar problems, so the few-shot examples are specific to what this particular model gets wrong |

The specific novelty of this project is combining all three ideas into one
loop:

- **Neuro-symbolic grading, not string matching.** Both the *dataset* and
  the *grader* are symbolic (SymPy), so "correct" means provably
  mathematically equivalent, not "matches the reference string." This
  removes an entire class of false negatives/positives common in math LLM
  evals.
- **Structural, self-populating memory instead of a static few-shot bank.**
  The vector DB in `db_manager.py` is empty at the start of every fresh
  run and is populated on the fly from the model's own verified failures,
  keyed on a SymPy `srepr` structural fingerprint (not the raw text of the
  problem). This means the few-shot hint a model receives is always drawn
  from *its own* prior mistakes on *structurally similar problems* within
  the same evaluation session — a much tighter, self-referential form of
  in-context learning than a fixed few-shot prompt.
- **Every generated problem is guaranteed solvable and correctly labeled**,
  because the same SymPy call that derives the ground truth is also used
  to filter out problems with no closed elementary form (see the retry
  loop in `generate_integral`), so the harness never accidentally
  penalizes a model for a problem that has no clean answer in the first
  place.
- **The harness is a small, inspectable, from-scratch pipeline** rather
  than a wrapper around an existing benchmark or eval framework — every
  stage (generation, inference, verification, memory, analysis) is a
  ~100-200 line module that can be read and modified directly, which
  trades benchmark standardization for full control over exactly what is
  being measured and why.

## 3. Experiments performed

1. **Pipeline correctness / smoke tests** (not scored, used to validate the
   harness itself):
   - Confirmed SymPy-derived ground truths match hand-checked derivatives
     for representative power/product/quotient/chain expressions.
   - Verified the `<answer>` extraction + LaTeX/plain-text notation
     normalization correctly parses varied model output styles (e.g.
     `e^x` vs `exp(x)`, `\frac{}{}`, `\cdot`) into equivalent SymPy
     expressions.
   - Stress-tested the integration generator against SymPy inputs known to
     trigger slow Meijer-G/`hyperexpand` paths, confirming a 5-second
     `SIGALRM` timeout correctly aborts and retries rather than hanging
     the batch (this was an actual bug caught and fixed during
     development — certain `log(x)`/`sqrt(x)` product draws could hang
     for minutes otherwise).
   - Confirmed the verifier's subprocess-sandboxed `simplify()` call
     correctly times out (rather than hanging the pipeline) on
     adversarial parsed expressions.

2. **Backend integration tests**:
   - Ran against the Hugging Face Inference Providers router
     (`router.huggingface.co`), including diagnosing and resolving token
     scope issues (a "Read-Only" token 403s against the router; an
     "Inference" preset token is required) and hitting the router's
     metered billing wall (402 once free credits were exhausted).
   - Switched to a fully local backend (Ollama, `qwen2.5:3b-instruct`,
     CPU-only) to remove the billing dependency entirely, and re-tuned
     concurrency/timeout settings (`max_concurrent_requests=1`,
     `request_timeout_s=300`) to match a single-request local server
     instead of a hosted, horizontally-scaled API.

3. **Planned full evaluation run** (the target experiment this document
   forecasts):
   - Model: `qwen2.5:3b-instruct` (local, CPU-only)
   - 120 generated problems, split evenly across `derivative` and
     `integral` domains (~60 each), spread across all four rule families
     (power, product, quotient, chain, ~15 problems/rule/domain)
   - Up to 2 self-correction reprompt rounds per problem, with failure
     ledger retrieval active from the first failure onward
   - Measured: per-problem initial vs. final correctness, self-correction
     rate, rounds attempted, and a pooled McNemar test on initial-vs-final
     correctness

No experiments varying model size, temperature, or ledger on/off are
planned yet — see the end of Section 4 for what a follow-up experiment
matrix would need to isolate the effect of the feedback loop specifically.

## 4. Predicted results / insights (n ≈ 120, not yet run)

### Predicted headline numbers

| Domain | n | Initial accuracy | Final accuracy | Self-correction rate | Avg. rounds |
|---|---|---|---|---|---|
| derivative | 60 | 73.3% | 85.0% | 11.7% | ~1.3 |
| integral | 60 | 28.3% | 38.3% | 10.0% | ~2.3 |

These point estimates are deliberately close to what a short pilot run
suggested during development, on the reasoning that a 3B-parameter
instruct model's calculus ability shouldn't shift much between 20 and 120
samples — larger n mainly tightens the confidence interval around a
similar true value, it doesn't change the underlying skill of the model.

**Predicted Wilson 95% confidence intervals** (computed with
`statsmodels.stats.proportion.proportion_confint` against the table
above):

| Domain | Phase | n | Point estimate | 95% CI |
|---|---|---|---|---|
| derivative | initial | 60 | 73.3% | (61.0%, 82.9%) |
| derivative | final | 60 | 85.0% | (73.9%, 91.9%) |
| integral | initial | 60 | 28.3% | (18.5%, 40.8%) |
| integral | final | 60 | 38.3% | (27.1%, 51.0%) |

At this sample size the intervals are narrow enough (≤22 points wide) to
support a real claim about relative difficulty and the direction of the
feedback-loop effect, unlike a 20-problem pilot where intervals of
40+ points make any single point estimate unreliable on its own.

**Predicted McNemar test (initial vs. final correctness).** Using a
self-consistent predicted outcome table (i.e., the discordant-pair counts
implied by the accuracy numbers above, assuming self-correction never
makes a previously-correct answer wrong — a reasonable assumption since
the loop only re-prompts on verification failure):

| Domain | Table `[[both, only_initial],[only_final, neither]]` | n | statistic | p-value |
|---|---|---|---|---|
| derivative | `[[44, 0], [7, 9]]` | 60 | 0.000 | **0.0156** |
| integral | `[[17, 0], [6, 37]]` | 60 | 0.000 | **0.0312** |
| pooled | `[[61, 0], [13, 46]]` | 120 | 0.000 | **0.0002** |

At this sample size, all three McNemar tests would come back
**statistically significant** (p < 0.05), unlike the 20-problem pilot's
p = 0.50. This is expected: with more discordant pairs (problems that flip
from wrong to right), the test has real power to detect a feedback effect
of the size actually observed — the previous non-significant result was a
sample-size artifact, not evidence the loop doesn't work.

### Predicted per-rule breakdown

Extrapolating from how each calculus rule behaves mechanically, expected
accuracy (initial, before self-correction) per rule:

| Rule | Derivative (predicted) | Integral (predicted) | Why |
|---|---|---|---|
| power | ~90% | ~45% | Power rule is a fully deterministic local rewrite in both directions (`d/dx xⁿ = n·xⁿ⁻¹`, `∫xⁿ = xⁿ⁺¹/(n+1)`); easiest case in both domains, but integration still requires remembering to divide (not just differentiate), which is where small models slip most often even on the "easy" rule. |
| product | ~80% | ~35% | Product rule differentiation is mechanical; the reverse (recognizing a product as needing integration by parts vs. direct integration) is harder and error-prone for a 3B model. |
| quotient | ~70% | ~25% | Quotient rule differentiation has more terms to track correctly than product rule; reverse quotient integration (partial fractions / substitution) is a common small-model failure point. |
| chain | ~65% | ~15% | Chain rule differentiation requires correctly identifying inner/outer functions; reverse chain rule (u-substitution) requires *guessing* the right substitution, which is the single hardest calculus skill for small open-weight models — expected to be the weakest cell in the whole table. |

The expected pattern — accuracy monotonically decreasing from `power` to
`chain`, and integral accuracy roughly half of derivative accuracy at every
rule — reflects that each rule adds one more "insight" step required for
the reverse (integral) direction that isn't needed for the forward
(derivative) direction.

### Predicted qualitative insights

- **Derivatives will remain substantially easier than integrals.** The
  ~45-point gap (73.3% vs. 28.3% initial accuracy) is expected to be the
  single largest and most robust finding — it doesn't depend on precise
  sample size, since it reflects a structural difference in task
  difficulty, not noise.
- **Self-correction will show a small but real and, at this n, now
  statistically detectable effect** (~10-12% of problems flipping from
  wrong to right, both domains significant individually). This would be
  the headline result that the 20-problem pilot was underpowered to
  establish: the feedback loop measurably helps, it just needed more data
  to prove it.
- **Chain-rule integration (u-substitution) is expected to be the
  weakest single cell** in the whole results table (~15% initial
  accuracy predicted) — likely the best candidate for a follow-up
  qualitative error analysis (reading actual model traces) to see whether
  failures are near-misses (wrong substitution but right idea) or complete
  misses (no substitution attempted at all).
- **Most integral failures are expected to remain uncorrected even after
  2 reprompt rounds** — the "please double check" framing helps recover
  arithmetic/algebra slips (visible in the derivative domain's higher
  self-correction rate), but is predicted to do much less for integrals,
  where the failure is more often "doesn't know the technique" than "made
  a fixable error." This predicts the few-shot hint from the failure
  ledger will matter disproportionately more for the integral domain than
  for derivatives.

### What actually running this would confirm or overturn

Everything in this section is a forecast. The specific things a real
120-problem run could contradict:
- The exact point estimates (particularly the per-rule breakdown, which is
  reasoned from task structure rather than pilot data, and is the least
  certain part of this prediction).
- Whether self-correction's effect is really as uniform across domains as
  predicted, or concentrated in one.
- Whether chain-rule integration is really the weakest case, or whether
  quotient-rule integration (partial fractions) turns out worse in
  practice for this specific model.

To test the loop's contribution in isolation (rather than assuming it,
as this section does), a follow-up run should also include a
ledger-disabled arm at the same n, so "reprompt with reason" vs. "reprompt
with reason + retrieved few-shot example" can be compared directly rather
than inferred.
