"""E2 step 1: fix test(600), development and pilot(50) BEFORE running judges. Seed 20261027.
Canonical split by repo over the 1276 repos of the sample (shuffle of a sorted list, fixed seed).
Note: the F0 30% split (03_baseline.py) yields ~530 trajectories < 600, so the test set uses 40% of repos (see PREREGISTRATION.md, deviation 1)."""
import json, sys, numpy as np, pandas as pd
from common import *
TEST_FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.40
rows = []
for l in open(ROOT + '/data/sample.jsonl'):
    r = json.loads(l)
    rows.append((r['instance_id'], r['repo'], int(r['target']), r['n_ai_total'], r['model']))
df = pd.DataFrame(rows, columns=['instance_id', 'repo', 'target', 'n_ai_total', 'model'])
repos = np.array(sorted(df.repo.unique()))
np.random.default_rng(SEED).shuffle(repos)
test_repos = set(repos[:int(TEST_FRAC * len(repos))])
df['split'] = np.where(df.repo.isin(test_repos), 'test', 'dev')
assert not (set(df[df.split == 'test'].repo) & set(df[df.split == 'dev'].repo))
print('split', df.groupby('split').target.agg(['size', 'sum']).to_dict())
rng = np.random.default_rng(SEED)
t = df[df.split == 'test']
print('test n', len(t), 'pos', t.target.sum(), 'n>10', (t.n_ai_total > 10).sum(), 'n>20', (t.n_ai_total > 20).sum(),
      'pos n>20', ((t.n_ai_total > 20) & (t.target == 1)).sum())
if len(sys.argv) > 2 and sys.argv[2] == 'dry':
    sys.exit()
N = 600
# stratified by outcome, proportional to the prevalence in the test split
npos = int(round(N * t.target.mean()))
pos = t[t.target == 1].sort_values('instance_id')
neg = t[t.target == 0].sort_values('instance_id')
tp = pos.iloc[rng.permutation(len(pos))[:npos]]
tn = neg.iloc[rng.permutation(len(neg))[:N - npos]]
test = pd.concat([tp, tn]).sort_values('instance_id')
assert len(test) == N
d = df[df.split == 'dev'].sort_values('instance_id')
pil = d.iloc[rng.permutation(len(d))[:50]]
out = ROOT + '/out/'
test.assign(set='test').to_csv(out + 'ids_test600.csv', index=False)
pil.assign(set='pilot').to_csv(out + 'ids_pilot50.csv', index=False)
d.assign(set='dev').to_csv(out + 'ids_dev_all.csv', index=False)
print('test', len(test), 'pos', test.target.sum(), 'n>10', (test.n_ai_total > 10).sum(), 'n>20', (test.n_ai_total > 20).sum(),
      'pos n>20', ((test.n_ai_total > 20) & (test.target == 1)).sum(), 'repos', test.repo.nunique())
print('dev', len(d), 'pos', d.target.sum(), 'n>20', (d.n_ai_total > 20).sum(), 'pilot pos', pil.target.sum())
