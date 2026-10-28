# Blind validation of the jury rubric

* `key.csv` maps item number `k` to the run `uid` (`instance_id|target`). The annotators never saw this file.
* `annotations/ann_[A-D].jsonl`: ratings (1-5) of the six rubric dimensions for items 000-079, written by four independent
  Claude Sonnet sub-agents (20 items each). Each annotator read only its 20 item files; each file contained exactly the prompt
  the Gemma jury received (issue + first 10 agent steps + rubric) and nothing else (no outcome, no Gemma scores).
* The item texts are not redistributed (they contain trajectory text from `nebius/SWE-agent-trajectories`); regenerate them
  with `src/09_export_validation.py` (seed 20261028).
* Agreement statistics: `src/10_validation.py` → `results/judge_agreement.csv`, `results/reference_predictive.json`.

This is an LLM reference, not a human annotation; the paper reports it as such.
