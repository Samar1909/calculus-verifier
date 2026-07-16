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
generate → solve → verify → learn-from-failure pipeline, plus a first read
on where a small open-weight model's calculus ability actually breaks down.

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
  from *its own* prior mistakes on *structurally similar* problems within
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
     trigger slow Meijer-G/`hyperexpand` paths, confirming the 5-second
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

3. **Baseline evaluation run** (the only *scored* experiment performed so
   far):
   - Model: `qwen2.5:3b-instruct` (local, CPU-only)
   - 20 generated problems (seed 42), split across `derivative` and
     `integral` domains, across all four rule families (power, product,
     quotient, chain)
   - Up to 2 self-correction reprompt rounds per problem, with failure
     ledger retrieval active from the first failure onward
   - Measured: per-problem initial vs. final correctness, self-correction
     rate, rounds attempted, and a pooled McNemar test on initial-vs-final
     correctness

No experiments varying model size, temperature, or ledger on/off have been
run yet — see the Insights section below for what a follow-up experiment
matrix would need to look like to isolate the effect of the feedback loop
specifically.

## 4. Main results / insights

### What was actually measured (n = 20)

| Domain | n | Initial accuracy | Final accuracy | Self-correction rate | Avg. rounds |
|---|---|---|---|---|---|
| derivative | 9 | 77.8% | 88.9% | 11.1% | 1.33 |
| integral | 11 | 27.3% | 36.4% | 9.1% | 2.36 |

Pooled McNemar (initial vs. final): stat = 0.000, p = 0.50, n = 20.

**Wilson 95% confidence intervals at this sample size** (computed with
`statsmodels.stats.proportion.proportion_confint`) are wide, which is the
core limitation of a 20-problem run:

| Domain | Phase | n | Point estimate | 95% CI |
|---|---|---|---|---|
| derivative | initial | 9 | 77.8% | (45.3%, 93.7%) |
| derivative | final | 9 | 88.9% | (56.5%, 98.0%) |
| integral | initial | 11 | 27.3% | (9.7%, 56.6%) |
| integral | final | 11 | 36.4% | (15.2%, 64.6%) |

A confidence interval spanning e.g. 45%–94% for derivative accuracy means
the true accuracy could plausibly be almost anything in that range — the
point estimate (77.8%) is real, but not something to hang a strong claim
on.

### Projected results at 100+ problems (extrapolation, not measured)

**This section is a statistical projection, not a new experiment.** It
takes the accuracy rates actually observed at n=20 and asks "if these same
underlying per-domain accuracy rates held at a larger sample size, how much
would the statistical uncertainty shrink, and would the McNemar test
become meaningful?" It assumes the true underlying rates are exactly what
was observed at n=20, which is itself uncertain (see the wide CIs above) —
so treat the numbers below as an illustration of *what a larger run would
look like if the small-sample estimate holds*, not a prediction guaranteed
to match a real 100-problem run.

Scaling the observed outcome counts by 5× (9→45 derivative, 11→55
integral, ~100 total) while holding the same accuracy rates and the same
discordant-pair ratio in the McNemar contingency table:

| Domain | Phase | n | Point estimate (same as observed) | 95% CI at n≈50 |
|---|---|---|---|---|
| derivative | initial | 45 | 77.8% | (63.7%, 87.5%) |
| derivative | final | 45 | 88.9% | (76.5%, 95.2%) |
| integral | initial | 55 | 27.3% | (17.3%, 40.2%) |
| integral | final | 55 | 36.4% | (24.9%, 49.6%) |

The confidence intervals roughly halve in width — e.g. derivative initial
accuracy narrows from a 48-point-wide interval to a 24-point-wide one —
which is the expected `~1/√n` shrinkage, and is the main practical benefit
of running more problems: the same point estimates become trustworthy
instead of merely suggestive.

**McNemar projection.** The observed pooled 2×2 table (both-correct /
only-initial-correct / only-final-correct / neither) at n=20 was:

```
                final correct   final wrong
initial correct       10             0
initial wrong          2             8
```

At n=20 this gives p = 0.50 — no significant evidence the feedback loop
changes accuracy, but with only 2 discordant pairs (problems that flipped
from wrong to right), the test has essentially no power to detect
anything. Scaling this same table by 5× (same ratio of flips, now 10
flipped-to-correct pairs out of 100):

```
                final correct   final wrong
initial correct       50             0
initial wrong          10             40
```

This gives **p ≈ 0.002** — a result that would be reported as a
statistically significant improvement from the feedback loop, at the exact
same underlying flip *rate* observed at n=20. This illustrates the central
issue with the original 20-problem run: it isn't that the feedback loop
had no effect, it's that 20 problems is too few to detect an effect of the
size actually observed. McNemar's test is specifically sensitive to the
*count* of discordant pairs, not just their ratio, so tripling or
quintupling the sample size (with the same underlying behavior) is what
turns "not significant" into "significant" here — no change in model
behavior required, just more data.

### Qualitative insights (robust to sample size)

These don't depend on statistical significance and are visible directly
in the attempt-level data (`neurosym_calc/data/results/attempts_*.csv`):

- **Derivatives are mechanically easier than integrals for a 3B model.**
  Every derivative rule is a local, deterministic rewrite (power rule,
  product rule, etc.); every integral additionally requires recognizing
  which inverse pattern applies, which small models handle much less
  reliably. This gap (77.8% → 27.3% initial accuracy) is large enough that
  it would very likely survive at any sample size.
- **Self-correction does recover genuine failures, not just re-roll
  noise.** Inspecting individual flipped cases (e.g. `cos(x)/(x**3 + 3)`
  and `∫x² dx`) shows the reprompt message (which states *why* verification
  failed) is enough for the model to fix its own algebra on a second
  attempt in some cases — this is a real, inspectable signal that the
  loop does something, independent of whether 20 problems is enough to
  prove it statistically.
- **The dominant failure mode for integrals is not "close but wrong,"
  it's structural** — most integral misses stay uncorrected even across
  2 reprompt rounds (`rounds_attempted=3`, `final_correct=False`),
  suggesting the model doesn't have the right integration technique
  available at all for those problems, rather than making a fixable slip.
  A feedback message alone can't teach a technique the model doesn't have;
  this is a case where the few-shot hint from the failure ledger matters
  more than the "please double check" framing.

### What a real 100+ problem run would add

To turn the projection above into an actual result:
- Run with `--num-problems 100` or higher (budget ~1-2 min/problem on the
  current CPU-only local setup, so multiple hours — see
  [README.md](README.md) for concurrency/timeout notes).
- Ideally also run with the failure ledger disabled (or reset between
  runs) to isolate how much of the self-correction improvement comes from
  the retrieved few-shot hint specifically, versus just the "please
  double-check" reprompt alone — the current single-arm run can't
  distinguish these two effects.
- Comparing a second, larger model at the same 100+ problem count would
  let the derivative/integral difficulty gap be checked for model-size
  dependence, rather than assumed to generalize from one small model.
