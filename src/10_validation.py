"""Judge validation: agreement between the small-model jury and an independent blind annotation of the same 80 test items
(prefix abs10). The reference annotators are Claude (Sonnet) sub-agents that saw exactly the prompt the jury saw and never
saw the outcome or any jury score. This is NOT a human annotation; it is reported as LLM-vs-LLM agreement.
Metrics per dimension: quadratically weighted Cohen's kappa, Spearman rho, exact and within-1 agreement, with 2,000
item-bootstrap 95% CIs. Usage: python 10_validation.py <key.csv> <annotations_dir> <e2_jsonl_dir> <out_dir>"""
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

KEY, ANN, E2, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
DIMS = ['localization', 'reproduction', 'coherence', 'test_evidence', 'loop', 'scope']
SEED, B = 20261028, 2000
key = pd.read_csv(KEY, dtype={'k': int})
ann = pd.concat([pd.read_json(f, lines=True, dtype={'k': str}) for f in sorted(glob.glob(ANN + '/ann_*.jsonl'))])
ann['k'] = ann.k.astype(int)
assert ann.k.is_unique and len(ann) == len(key), (len(ann), len(key))
ref = key.merge(ann, on='k')
ref['target'] = ref.uid.str.split('|').str[1].astype(int)
rng = np.random.default_rng(SEED)
boots = [rng.integers(0, len(ref), len(ref)) for _ in range(B)]


def qwk(a, b):
    if len(set(a)) == 1 and len(set(b)) == 1:
        return np.nan
    return cohen_kappa_score(a, b, weights='quadratic', labels=[1, 2, 3, 4, 5])


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


rows = []
for tag in ('e4b', 'e2b', 'q4b'):
    fn = f'{E2}/{tag}__test__abs10__jury.jsonl'
    if not os.path.exists(fn):
        continue
    j = pd.DataFrame([json.loads(l) for l in open(fn)]).drop_duplicates('uid', keep='last')
    j = j[j.parsed.notna()]
    for d in DIMS:
        j['m_' + d] = [p[d] for p in j.parsed]
    m = ref.merge(j[['uid'] + ['m_' + d for d in DIMS]], on='uid')
    for d in DIMS + ['mean']:
        a = m[DIMS].mean(1).values if d == 'mean' else m[d].values
        b = m[['m_' + x for x in DIMS]].mean(1).values if d == 'mean' else m['m_' + d].values
        r = dict(model=tag, dim=d, n=len(m), spearman=spearmanr(a, b).statistic,
                 sp_lo=ci([spearmanr(a[i], b[i]).statistic for i in boots])[0], sp_hi=ci([spearmanr(a[i], b[i]).statistic for i in boots])[1])
        if d != 'mean':
            a, b = a.astype(int), b.astype(int)
            ks = [qwk(a[i], b[i]) for i in boots]
            r.update(qwk=qwk(a, b), qwk_lo=ci(ks)[0], qwk_hi=ci(ks)[1], exact=float((a == b).mean()),
                     within1=float((abs(a - b) <= 1).mean()), mean_ref=float(a.mean()), mean_model=float(b.mean()))
        rows.append(r)
res = pd.DataFrame(rows)
# Reference-annotation predictive check (descriptive only: 80 items)
y = ref.target.values
pred = dict(n=len(ref), n_pos=int(y.sum()))
for d in DIMS + ['mean']:
    s = ref[DIMS].mean(1).values if d == 'mean' else ref[d].values
    pred['auroc_ref_' + d] = roc_auc_score(y, s)
pred['auroc_ref_mean_ci'] = ci([roc_auc_score(y[i], ref[DIMS].mean(1).values[i]) for i in boots if y[i].min() != y[i].max()])
res.to_csv(f'{OUT}/judge_agreement.csv', index=False)
json.dump(pred, open(f'{OUT}/reference_predictive.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 220)
print(res.round(3).to_string())
print(json.dumps(pred, indent=1, default=float))
