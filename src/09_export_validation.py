"""Jury validation: exports 80 test items (abs10 prefix) rendered EXACTLY as the judge saw them.
Annotators receive only items/<k>.txt (no outcome and no Gemma scores). The uid<->k key is kept in key.csv.
Usage (server, .venv-vllm): python 09_export_validation.py"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_common import build_prompt, get_prefix, JURY_TASK
from transformers import AutoTokenizer
ROOT = os.environ.get('SJEV_ROOT', '/home/ai/proyectos/gemma4-paper')  # data/models/outputs root
OUT = ROOT + '/out/validation'
N, SPEC, SEED = 80, 'abs10', 20261028
os.makedirs(OUT + '/items', exist_ok=True)
tok = AutoTokenizer.from_pretrained(ROOT + '/models/gemma-4-E4B-it')
count = lambda t: len(tok.encode(t, add_special_tokens=False))
judged = [json.loads(l)['uid'] for l in open(ROOT + '/out/e2/e4b__test__%s__jury.jsonl' % SPEC)]
judged = sorted(set(judged))
rng = np.random.default_rng(SEED)
pick = [judged[i] for i in sorted(rng.choice(len(judged), N, replace=False))]
order = rng.permutation(N)          # random presentation order, independent of the outcome
want = set(pick)
trajs = {}
for l in open(ROOT + '/data/sample.jsonl'):
    r = json.loads(l)
    u = '%s|%d' % (r['instance_id'], int(r['target']))
    if u in want:
        trajs[u] = r['traj']
with open(OUT + '/key.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['k', 'uid'])
    for k, j in enumerate(order):
        u = pick[j]
        prompt, n = build_prompt(get_prefix(trajs[u], SPEC), 'jury', count)
        open('%s/items/%03d.txt' % (OUT, k), 'w').write(prompt)
        w.writerow([k, u])
print('exported', N, 'items to', OUT)
