"""Robustness analyses requested by the adversarial internal review (added after pre-registration; all CPU, all on
released files). Nothing here is tuned on the test split: every threshold and cap is chosen on development predictions.
 A. Step caps chosen on dev (5..60 steps), and the combined policy "jury abort at step 10 + dev-chosen cap".
 B. H4 robustness: one agent step charged per judgement, neighbourhood of tau, prevalence re-weighting to the dataset's 16.7%.
 C. Within-issue pairs: for issues that have both a resolved and a failed judged run, how often is the resolved run ranked higher?
 D. Calibration: ECE CI, calibration slope, prevalence-reweighted ECE, ECE noise floor of a perfectly calibrated predictor.
 E. Issue-only ablation (Gemma 4 E4B sees the issue but no agent step), paired with the k=10 jury.
 F. Validation: paired AUROC difference reference annotation vs E4B on the 80 validation items.
Usage: python 11_robustness.py <results_dir> <verdicts_dir> <validation_dir>"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RES, VER, VAL = sys.argv[1:4]
SEED, B, K, PREV = 20261027, 2000, 10, 0.167
DIMS = ['localization', 'reproduction', 'coherence', 'test_evidence', 'loop', 'scope']
rng = np.random.default_rng(SEED + 7)
out = {}
te_runs = pd.read_csv(f'{RES}/test_runs.csv')
dv_runs = pd.read_csv(f'{RES}/dev_runs.csv')
tp = pd.read_csv(f'{RES}/test_predictions.csv')
dp = pd.read_csv(f'{RES}/dev_predictions_all_models.csv')


def cboot(groups, n=B):
    reps = np.unique(groups)
    idx = {r: np.where(groups == r)[0] for r in reps}
    return [np.concatenate([idx[r] for r in rng.choice(reps, len(reps))]) for _ in range(n)]


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def pboot(d):
    d = np.asarray(d)
    return float(min(1, max(1 / len(d), 2 * min((d <= 0).mean(), (d >= 0).mean()))))


def attach(runs, preds, model, col):
    p = preds[(preds.model == model) & (preds.prefix == 'abs10')][['uid', col]]
    r = runs.merge(p, on='uid', how='left')
    r['judged'] = r[col].notna().astype(int)
    r[col] = r[col].fillna(1.0)          # runs with <= 10 steps are never judged and never aborted at step 10
    return r


def eff(df, p=None, tau=None, cap=None, w=None, overhead=0.0):
    n = df.n_ai_total.values.astype(float)
    y = df.target.values.astype(float)
    w = np.ones(len(df)) if w is None else w
    ab = np.zeros(len(df), bool) if p is None else ((1 - df[p].values) > tau)
    steps = np.where(ab, K, n)
    res = np.where(ab, 0, y)
    if cap is not None:
        capped = (~ab) & (n > cap)
        steps = np.where(capped, cap, steps)
        res = np.where(capped, 0, res)
    steps = steps + overhead * (df.judged.values if 'judged' in df else 0)
    return (w * res).sum() / (w * steps).sum()


TAUS = np.round(np.linspace(0.30, 0.99, 70), 4)
CAPS = list(range(5, 65, 5))
for model in ('e4b', 'e2b', 'q4b'):
    if model not in set(tp.model):
        continue
    T = attach(te_runs, tp, model, 'jury_p').merge(tp[(tp.model == model) & (tp.prefix == 'abs10')][['uid', 'base_p']], on='uid', how='left')
    T['base_p'] = T.base_p.fillna(1.0)
    D = attach(dv_runs, dp, model, 'jury_oof').merge(dp[(dp.model == model) & (dp.prefix == 'abs10')][['uid', 'base_oof']], on='uid', how='left')
    D['base_oof'] = D.base_oof.fillna(1.0)
    bidx = cboot(T.repo.values)
    none = lambda df, w=None: eff(df, w=w)
    tau_j = max(TAUS, key=lambda t: eff(D, 'jury_oof', t))
    cap_s = max(CAPS, key=lambda c: eff(D, cap=c))
    tj2, cp2 = max(((t, c) for t in TAUS for c in CAPS if c > K), key=lambda tc: eff(D, 'jury_oof', tc[0], tc[1]))
    pol = {'jury_at_10': lambda df: eff(df, 'jury_p', tau_j),
           f'cap_at_{cap_s}_dev_chosen': lambda df: eff(df, cap=cap_s),
           'cap_at_10': lambda df: eff(df, cap=10),
           f'jury_at_10_plus_cap_{cp2}': lambda df: eff(df, 'jury_p', tj2, cp2),
           'jury_at_10_overhead_1_step': lambda df: eff(df, 'jury_p', tau_j, overhead=1.0)}
    rel = {k: np.array([f(T.iloc[i]) / none(T.iloc[i]) - 1 for i in bidx]) for k, f in pol.items()}
    R = {'tau_jury_dev': float(tau_j), 'cap_dev': cap_s, 'combined_dev': [float(tj2), cp2]}
    for k, f in pol.items():
        R[k] = dict(gain=f(T) / none(T) - 1, ci=ci(rel[k]), p=pboot(rel[k]))
    for a, b in (('jury_at_10', f'cap_at_{cap_s}_dev_chosen'), (f'jury_at_10_plus_cap_{cp2}', f'cap_at_{cap_s}_dev_chosen'),
                 (f'jury_at_10_plus_cap_{cp2}', 'jury_at_10')):
        d = rel[a] - rel[b]
        R[f'{a}__minus__{b}'] = dict(delta=R[a]['gain'] - R[b]['gain'], ci=ci(d), p=pboot(d))
    # tau neighbourhood (test, descriptive)
    R['tau_neighbourhood_gain'] = {f'{t:.2f}': eff(T, 'jury_p', t) / none(T) - 1 for t in TAUS if abs(t - tau_j) <= 0.10 + 1e-9}
    # prevalence re-weighting of the test set to the full-dataset success rate
    pt = T.target.mean()
    w = np.where(T.target.values == 1, PREV / pt, (1 - PREV) / (1 - pt))
    R['reweighted_to_16.7pct'] = dict(jury_at_10=eff(T, 'jury_p', tau_j, w=w) / eff(T, w=w) - 1,
                                      cap_dev=eff(T, cap=cap_s, w=w) / eff(T, w=w) - 1,
                                      combined=eff(T, 'jury_p', tj2, cp2, w=w) / eff(T, w=w) - 1)
    out[f'A_B_budget_{model}'] = R

# C. within-issue pairs (k = 10)
C = {}
for model in ('e4b', 'e2b', 'q4b'):
    d = tp[(tp.model == model) & (tp.prefix == 'abs10')]
    if d.empty:
        continue
    g = d.groupby('instance_id').target.agg(['min', 'max'])
    both = g[(g['min'] == 0) & (g['max'] == 1)].index
    pr = d[d.instance_id.isin(both)].pivot_table(index='instance_id', columns='target', values=['jury_p', 'hol_p', 'base_p'])
    row = {'n_pairs': int(len(both))}
    for s in ('jury_p', 'hol_p', 'base_p'):
        c = np.where(pr[(s, 1)] > pr[(s, 0)], 1.0, np.where(pr[(s, 1)] == pr[(s, 0)], 0.5, 0.0))
        bs = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(B)]
        row[s] = dict(concordance=float(c.mean()), ci=ci(bs))
    rest = d[~d.instance_id.isin(both)]
    row['auroc_jury_other_issues'] = dict(n=len(rest), auroc=roc_auc_score(rest.target, rest.jury_p))
    C[model] = row
out['C_within_issue_pairs'] = C

# D. calibration of the E4B jury at k = 10
d = tp[(tp.model == 'e4b') & (tp.prefix == 'abs10')].reset_index(drop=True)
y, p = d.target.values, d.jury_p.values


def ece(y, p, w=None, nb=10):
    w = np.ones(len(y)) if w is None else w
    b = np.minimum((p * nb).astype(int), nb - 1)
    return sum(w[b == k].sum() / w.sum() * abs(np.average(y[b == k], weights=w[b == k]) - np.average(p[b == k], weights=w[b == k]))
               for k in range(nb) if (b == k).any())


lg = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
slope = LogisticRegression(C=1e6).fit(lg(p).reshape(-1, 1), y).coef_[0][0]
bi = cboot(d.repo.values)
pt = y.mean()
w = np.where(y == 1, PREV / pt, (1 - PREV) / (1 - pt))
floor = [ece(rng.binomial(1, p), p) for _ in range(B)]
out['D_calibration_e4b_k10'] = dict(ece=ece(y, p), ece_ci=ci([ece(y[i], p[i]) for i in bi]), slope=float(slope),
                                    ece_reweighted_16_7pct=ece(y, p, w),
                                    ece_noise_floor_median=float(np.median(floor)), ece_noise_floor_95=float(np.percentile(floor, 95)))

# E. issue-only ablation (E4B)
E = {}
fn = {s: {j: f'{VER}/e4b__{s}__issue10__{j}.jsonl' for j in ('holistic', 'jury')} for s in ('test', 'dev')}
if all(os.path.exists(f) for s in fn.values() for f in s.values()):
    def load(s):
        h = pd.DataFrame([json.loads(l) for l in open(fn[s]['holistic'])]).drop_duplicates('uid', keep='last')
        j = pd.DataFrame([json.loads(l) for l in open(fn[s]['jury'])]).drop_duplicates('uid', keep='last')
        h['issue_hol'] = [x['p_success'] if isinstance(x, dict) else 0.25 for x in h.parsed]
        for k in DIMS:
            j['i_' + k] = [x[k] if isinstance(x, dict) else 3.0 for x in j.parsed]
        return h[['uid', 'issue_hol']].merge(j[['uid'] + ['i_' + k for k in DIMS]], on='uid')
    Dv = load('dev').merge(dp[(dp.model == 'e4b') & (dp.prefix == 'abs10')][['uid', 'repo', 'target', 'jury_oof']], on='uid')
    Te = load('test').merge(d[['uid', 'repo', 'target', 'jury_p', 'hol_p', 'base_p']], on='uid')
    ic = ['i_' + k for k in DIMS]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(Dv[ic], Dv.target)
    Te['issue_jury'] = m.predict_proba(Te[ic])[:, 1]
    oof = np.zeros(len(Dv))
    for a, b in GroupKFold(5).split(Dv, Dv.target, Dv.repo):
        oof[b] = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(Dv.iloc[a][ic], Dv.iloc[a].target).predict_proba(Dv.iloc[b][ic])[:, 1]
    st = LogisticRegression(C=1e6).fit(np.c_[lg(oof), lg(Dv.jury_oof.values)], Dv.target)
    Te['issue_plus_run'] = st.predict_proba(np.c_[lg(Te.issue_jury.values), lg(Te.jury_p.values)])[:, 1]
    yb = Te.target.values
    bi = cboot(Te.repo.values)
    A = {k: np.array([roc_auc_score(yb[i], Te[k].values[i]) for i in bi]) for k in ('issue_hol', 'issue_jury', 'jury_p', 'issue_plus_run', 'base_p')}
    for k in A:
        E[k] = dict(auroc=roc_auc_score(yb, Te[k]), ci=ci(A[k]))
    for a, b in (('jury_p', 'issue_jury'), ('jury_p', 'issue_hol'), ('issue_plus_run', 'issue_jury')):
        dd = A[a] - A[b]
        E[f'{a}__minus__{b}'] = dict(delta=E[a]['auroc'] - E[b]['auroc'], ci=ci(dd), p=pboot(dd))
    E['n'] = len(Te)
    E['invalid_json'] = int(sum(1 for s in ('test', 'dev') for j in ('holistic', 'jury') for l in open(fn[s][j]) if json.loads(l)['parsed'] is None))
out['E_issue_only_ablation_e4b_k10'] = E or 'not available'

# F. validation: paired AUROC on the 80 items
key = pd.read_csv(f'{VAL}/key.csv')
ann = pd.concat([pd.read_json(f'{VAL}/annotations/{f}', lines=True, dtype={'k': str}) for f in sorted(os.listdir(f'{VAL}/annotations'))])
ann['k'] = ann.k.astype(int)
v = key.merge(ann, on='k').merge(d[['uid', 'target', 'jury_p'] + DIMS].rename(columns={x: 'g_' + x for x in DIMS}), on='uid')
v['ref_mean'] = v[DIMS].mean(1)
v['jury_mean'] = v[['g_' + x for x in DIMS]].mean(1)
yv = v.target.values
bs = [rng.integers(0, len(v), len(v)) for _ in range(B)]
F = {k: dict(auroc=roc_auc_score(yv, v[k])) for k in ('ref_mean', 'jury_mean', 'jury_p')}
for a, b in (('ref_mean', 'jury_mean'), ('ref_mean', 'jury_p')):
    dd = [roc_auc_score(yv[i], v[a].values[i]) - roc_auc_score(yv[i], v[b].values[i]) for i in bs if yv[i].min() != yv[i].max()]
    F[f'{a}__minus__{b}'] = dict(delta=F[a]['auroc'] - F[b]['auroc'], ci=ci(dd), p=pboot(dd))
F['n'], F['n_pos'] = len(v), int(yv.sum())
F['e4b_jury_auroc_all_429'] = roc_auc_score(y, p)
out['F_validation_paired'] = F
json.dump(out, open(f'{RES}/robustness.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))
