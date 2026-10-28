"""Anti-leakage tests: features exclude eval_logs/generated_patch/exit_status; the 100% prefix does not end in submit."""
import json
import pandas as pd
from common import *
cols = pd.read_csv(ROOT + '/out/features.csv', nrows=1).columns
assert not any(c in cols for c in ('exit_status', 'eval_logs', 'generated_patch', 'n_ai_total')), cols
n = bad = 0
for line in open(ROOT + '/data/sample.jsonl'):
    r = json.loads(line)
    assert set(r) == {'instance_id', 'repo', 'model', 'target', 'exit_status', 'n_ai_total', 'traj'}
    p = prefix_fraction(r['traj'], 1.0)
    n += 1
    if p and p[-1]['role'] == 'ai' and is_submit(p[-1]['text']):
        bad += 1
    assert len(p) <= len(r['traj'])
print('trajectories', n, 'prefix100 ending in submit:', bad)
assert bad == 0
print('OK anti-leakage')
