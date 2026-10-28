# Pre-registration — "Small Juries, Early Verdicts"

Faithful English translation of the internal pre-registration saved on **2026-10-27 at 18:00 UTC** (file timestamp), before
any judge was run on any split: the 50-item pilot started at 18:05 UTC, the first test verdicts were written at 18:16 UTC (a
run later discarded for a keying bug) and the definitive runs started at 19:03 UTC (jobs `cb37ff9c0ff94b63`,
`0556d890240f4133`, `4441360f556f4b0f`, `e096792838e34e6e`). It was
recorded in the project's working notes, not in a public time-stamped registry (e.g. OSF); readers should weigh it accordingly.
Design decisions were taken with the JEV-Orchestrator decision engine (absolute prefixes k = 10/20; holistic judge vs. six
BenchClaw-style dimensions; pilot of 50 then 600 test runs; one model per GPU job).

## Data
`nebius/SWE-agent-trajectories` (CC-BY-4.0). 1,765 stratified trajectories (one per repository and outcome class), split by
repository and fixed before any judge was run (`src/01_sample.py`, `src/05_splits.py`). Test = 600 trajectories of the test
repositories (stratified by outcome), fixed with seed 20261027 before running any judge. A pilot of 50 development
trajectories was used only to check the prompt format (test AUROC never inspected).

## Conditions
- Absolute step prefixes k ∈ {10, 20} (main result). Fractional prefixes (33 % / 66 %) only as an upper bound, because they
  leak trajectory length.
- Judges: Gemma 4 E2B-it and E4B-it (bf16, vLLM), temperature 0, JSON output.
  - **Holistic:** probability of success in [0, 1].
  - **Six-dimension jury** (rubric design inspired by BenchClaw): localization, bug reproduction, coherence of the edits with
    the issue, test evidence, loop signals, patch scope; each 1–5, aggregated by a logistic regression fit on the development split.
- Baseline: prefix trace features without an LLM (E1).

## Hypotheses
- **H1:** the E4B jury at k = 20 has a higher AUROC than the baseline (paired bootstrap, 2,000 resamples; 95 % CI of the
  difference > 0).
- **H2:** the six-dimension jury beats the holistic judge of the same size (ΔAUROC with 95 % CI > 0) or adds information when
  stacked with the baseline (ΔAUROC of the stack ≥ +0.02).
- **H3:** E2B reaches ≥ 90 % of the AUROC of E4B at less than half the cost (tokens and seconds).
- **H4 (application):** the policy "abort if P(failure) > τ" (τ chosen on development data) resolves more tasks per unit of
  budget than never aborting, in a simulation on the test split.

## Rules
Metrics: AUROC, ECE (10 bins), Brier; bootstrap confidence intervals. No prompt and no threshold is tuned on the test split.
Negative results are reported in the same way as positive ones.

## Declared deviations and additions (written 2026-10-28, after the results)
1. Because absolute prefixes require runs longer than k steps, the test sets for k = 10 / 20 contain 429 / 239 of the 600 runs
   (112 / 38 resolved). H4 is still evaluated on all 600 runs (runs that end before step k are never aborted).
2. H2's stacking leg was coded as "point estimate ≥ +0.02"; we additionally report its bootstrap CI and treat CI > 0 as the
   strict criterion.
3. H3: both judges see byte-identical prompts, so the token ratio is 1.00 by construction; the cost leg is therefore decided
   by GPU seconds.
4. Added after pre-registration: two-sided bootstrap p-values with Holm correction over 8 confirmatory contrasts; an
   additional open-model control (Qwen3-4B-Instruct-2507) with identical prompts; a blind validation of the jury dimensions
   against an independent LLM annotation (80 items); jury-vs-baseline and jury-vs-fixed-cap contrasts for H4.
