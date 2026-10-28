# Experiment log

Every run is listed, including the discarded and the negative ones. Machine-readable version: [`experiments.csv`](experiments.csv)
(`id, date, hypothesis, configuration, commit, metric, decision`). Hardware: one Debian server with 2× RTX 3090 (24 GB) and
15 GB RAM; GPU jobs ran one model per job through a FIFO queue; CPU work was pinned to 4 cores. Seeds: 20261027 (sampling,
splits, judges, bootstrap), 20261028 (validation sample). Job ids are the queue ids on that server.

## Protocol for every measurement
Hypothesis written → design → control (no-LLM trace features, the same model without the rubric, a different open model,
a fixed step cap) → run with fixed seeds → pre-registered metric → decision (accepted / rejected / inconclusive).

## 2026-10-27

**F0 — data and licences.** `nebius/SWE-agent-trajectories` is CC-BY-4.0 (80,036 runs, 1,276 repositories, 16.7% resolved).
`SWE-Gym/OpenHands-Sampled-Trajectories` was rejected because its metadata carries no licence. Gemma 4 E2B/E4B-it are
Apache-2.0 and not gated.

**E0 — serving smoke test** (job `8c49b6ff42884310`). vLLM 0.32.0, Gemma 4 E4B-it bf16 on one RTX 3090: 20/20 valid JSON,
prefill 6,377 tok/s, engine start 243 s. A first submission (`1da8f14f41534e9b`) failed instantly because Git Bash rewrote
the interpreter path; no GPU time used. *Accepted.*

**E1 — no-LLM control.** 43 prefix-only features; logistic regression and gradient boosting; 30% repository hold-out and
5-fold grouped CV. Absolute prefixes: AUROC 0.60 (k = 10) and 0.62–0.66 (k = 20). Fractional prefixes: 0.68–0.76, identified
as **length leakage** (the size of a fractional prefix encodes the final run length). *Decision:* absolute prefixes are the
main setting; fractional ones are an upper bound only.

**Pre-registration** saved 18:00 UTC (see [`PREREGISTRATION.md`](../PREREGISTRATION.md)).

**E2-pilot** (jobs `dd1b72acc5a945eb`, `a630e4710eac4d7f`; two earlier submissions `268191e9a2744cd3`, `c0951e355df44e2e`
failed on a missing `pandas` in the GPU environment). 50 development runs, both judges, both Gemma sizes: 100% valid JSON.
*Prompt v1 frozen.*

**E2-v0** (job `9b3f199a9a634d0e`). *Rejected — bug.* Outputs were keyed by `instance_id`, which is not unique (the same issue
has a resolved and an unresolved run). Outputs quarantined; every later file is keyed by `uid = instance_id|target`.

**E2** (jobs `cb37ff9c0ff94b63` E4B and `0556d890240f4133` E2B for k = 10/20; `4441360f556f4b0f` E4B and
`e096792838e34e6e` E2B for fractional prefixes). 20,596 verdicts, 1 invalid JSON. Three queued duplicates
(`bbe88cf2dd8d4ab0`, `89db9aa6a14a4808`, `8ad5b9a17a8046c1`) were cancelled before running.

## 2026-10-28

**Pre-registered analysis** (`src/07_analysis.py`). Results in `results/metrics_preregistered.csv` and
`results/hypotheses_preregistered.json`.

| Hypothesis | Result | Decision |
|---|---|---|
| H1, E4B jury > baseline at k = 20 | Δ +0.021 [−0.090, +0.131] | not supported (38 positives) |
| H1 at k = 10 (secondary) | Δ +0.102 [+0.034, +0.170], Holm p = 0.020 | supported |
| H2a, jury > holistic (E4B) | k = 10 +0.022 [−0.021, +0.065]; k = 20 −0.049 [−0.125, +0.026] | not supported |
| H2b, stack adds ≥ 0.02 | k = 10 +0.096 [+0.041, +0.151], Holm p = 0.006; k = 20 +0.042 [−0.027, +0.109] | supported at k = 10 |
| H3, E2B ≥ 90% of E4B at < ½ cost | AUROC ratio 0.93–1.01; GPU-time ratio 0.55–0.59 | rejected (cost leg) |
| H4, jury abort > never abort | k = 10 +29.3% [+13.8, +45.2], Holm p = 0.004; k = 20 +26.6% [+15.3, +39.0], Holm p = 0.004 | supported |

**S1 — supplementary statistics** (`src/08_supplementary.py`, added after pre-registration). Two-sided bootstrap p-values,
Holm over 8 confirmatory contrasts; H4 contrasts: jury vs trace-feature policy at k = 10 +19.4 points [+6.9, +32.6]
(p = 0.002); jury vs fixed cap +46.1 points [+28.9, +62.7]; at k = 20 the jury equals the cap (+1.7 [−3.7, +7.9]).

**V1/V2 — rubric validation** (`src/09_export_validation.py`, `src/10_validation.py`). 80 random test prompts at k = 10
(seed 20261028), rated blind by four Claude Sonnet sub-agents (20 items each) on the identical prompt. E4B quadratic-weighted
κ: localization 0.65, loop 0.63, reproduction 0.49, test evidence 0.45, scope 0.22, coherence 0.13; mean-score Spearman
0.73 [0.60, 0.83]. On these 80 items the reference annotation reaches AUROC 0.83 [0.73, 0.92] and the calibrated E4B jury
0.77. *Accepted, with the caveat that the reference is an LLM.*

**R1 — internal adversarial review** (an independent Claude Opus reviewer read the draft and the result files). All numbers
matched except four wording/rounding issues (fixed). Major points and our response:
- *The fixed cap at step 10 is a strawman* → added caps chosen on dev (5–60 steps; best = 20) and a jointly tuned policy
  (`src/11_robustness.py`). E4B: jury at 10 +29.3% [+13.6, +46.0]; best dev cap +24.9% [+11.3, +38.2]; jury − cap +4.4
  [−8.9, +18.1] (n.s.); jury at 10 + cap at 30 +36.4% [+19.3, +54.0]; combined − jury +7.0 [+1.5, +13.1]. The "+46 points
  over a cap" headline was removed.
- *Is the signal the run or the issue?* → within-issue pairs (52 issues with both outcomes): E4B jury ranks the resolved run
  higher 0.54 [0.40, 0.67] — inconclusive; an issue-only ablation (E4B sees the issue, no agent step) was run (E1-issue).
- *Calibration overclaimed* → ECE CI [0.019, 0.070], slope 1.12, noise floor 0.032 (median), re-weighted ECE 0.072 reported.
- *H4 robustness* → judgement overhead +22.8% [+7.8, +38.5]; τ ± 0.10: +20% to +29%; re-weighted to 16.7%: +34.7%.
- *Validation overclaimed* → one rating per item; paired AUROC reference − calibrated jury +0.06 [−0.02, +0.14].
- Prior art added: Speculative Rejection (arXiv:2410.20290), process reward models (arXiv:2305.20050).

**E1-issue — issue-only ablation** (job `0a0269895a034f27`). Gemma 4 E4B, prompt v1, issue text only (log cut before the first agent action), the same 429 test and 760 dev runs as k = 10; 0 invalid JSON; 402 s on one RTX 3090. AUROC issue-only holistic 0.533 [0.482, 0.582], issue-only jury 0.535 [0.481, 0.586] vs k = 10 jury 0.677; difference +0.142 [+0.063, +0.218], p = 0.001. *Accepted:* the signal is in the run, not in issue difficulty. A first submission (`559ffb666c064898`) was cancelled by us while queued (RAM request too high) and a second (`0a0269895a034f27`) was interrupted by a server reboot before starting; the result comes from job `78abe62ed44c4df4`.

**C1 — open-model control** (job `0f4417cc04eb4aed`). Qwen3-4B-Instruct-2507 with byte-identical prompts on test and dev,
k = 10, k = 20 and frac66. Interrupted twice for reasons outside the study (an external SIGTERM on `0f4417cc04eb4aed`; a server reboot during queued `e4ee6a6673434997`); the resumable runner kept every finished verdict and `e4cfedc4491740ad` completed test and dev for k = 10 and 20 (plus test frac66). We cancelled its last stage (dev frac66, an upper-bound setting not used in the paper) to free the shared GPU; Qwen fractional results are therefore not reported. 0 invalid JSON. Results (k = 10): holistic 0.663 [0.610, 0.715], jury 0.637 [0.580, 0.692]; holistic − trace features +0.088 [+0.014, +0.161]; E4B jury − Qwen jury +0.040 [−0.001, +0.081] (p = 0.053); jury abort policy +13.8% [−1.7, +30.7] vs +29.3% for E4B. Agreement with the blind reference: localization κw 0.64, reproduction 0.60, loop 0.16. *Accepted:* the effect is not Gemma-specific; Gemma 4 E4B gives the strongest jury.
