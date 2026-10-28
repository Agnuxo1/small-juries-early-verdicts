"""Supplementary analysis (declared as a deviation from / extension of the preregistration):
1) two-sided bootstrap p-values (repo cluster, B=2000) and Holm correction over the confirmatory contrasts;
2) additional H4 contrasts (jury - no-LLM baseline, jury - fixed cap);
3) control with another open model of similar size (Qwen3-4B-Instruct-2507, tag q4b) using the same prompts;
4) efficiency curves versus tau (descriptive) for figures.
Reuses the functions of 07_analysis.py without changing them. Usage: python 08_supplementary.py <e2_jsonl_dir> <output_dir>"""
import sys, os, json, importlib.util
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

IN, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
sys.argv = [sys.argv[0], IN, OUT]
spec_ = importlib.util.spec_from_file_location('a7', os.path.join(os.path.dirname(os.path.abspath(__file__)), '07_analysis.py'))
a7 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(a7)
a7.MODELS = [m for m in ('e2b', 'e4b', 'q4b') if os.path.exists(f'{IN}/{m}__test__abs10__jury.jsonl')]
print('models', a7.MODELS, flush=True)
rng = np.random.default_rng(a7.SEED + 1)


def pboot(d):
    d = np.asarray(d)
    d = d[~np.isnan(d)]
    return float(min(1.0, max(1.0 / len(d), 2 * min((d <= 0).mean(), (d >= 0).mean()))))


rows, curves, preds, devpreds = [], [], [], []
for spec in ('abs10', 'abs20', 'frac66'):
    R = a7.evaluate(spec)
    if not R['models']:
        continue
    t0 = next(iter(R['models'].values()))['test']
    y, g = t0.target.values, t0.repo.values
    bidx = a7.boot_idx(g, rng)
    sc = {'base': t0.base_p.values}
    for tag, M in R['models'].items():
        te = M['test']
        assert (te.uid.values == t0.uid.values).all()
        sc[f'{tag}_hol'], sc[f'{tag}_jury'] = te.hol_p.values, te.jury_p.values
        sc[f'{tag}_stack_jury'], sc[f'{tag}_stack_hol'] = te.stack_p.values, te.stack_hol_p.values
        sc[f'{tag}_hol_platt'] = te.hol_p_platt.values
        preds.append(te.assign(model=tag, prefix=spec))
        devpreds.append(M['dev'].assign(model=tag, prefix=spec))
    A = {k: np.array([a7.auc(y, v, ii) for ii in bidx]) for k, v in sc.items()}
    for k, v in sc.items():
        lo, hi = a7.ci(A[k])
        rows.append(dict(kind='auroc', prefix=spec, a=k, b='', n=len(y), n_pos=int(y.sum()), est=roc_auc_score(y, v), lo=lo, hi=hi, p=np.nan))
    pairs = [(f'{m}_jury', 'base') for m in R['models']] + [(f'{m}_hol', 'base') for m in R['models']] + \
            [(f'{m}_jury', f'{m}_hol') for m in R['models']] + [(f'{m}_stack_jury', 'base') for m in R['models']] + \
            [(f'{m}_stack_hol', 'base') for m in R['models']]
    if 'q4b' in R['models']:
        pairs += [('e4b_jury', 'q4b_jury'), ('e4b_hol', 'q4b_hol'), ('e2b_jury', 'q4b_jury')]
    for a, b in pairs:
        d = A[a] - A[b]
        lo, hi = a7.ci(d)
        rows.append(dict(kind='delta_auroc', prefix=spec, a=a, b=b, n=len(y), n_pos=int(y.sum()),
                         est=roc_auc_score(y, sc[a]) - roc_auc_score(y, sc[b]), lo=lo, hi=hi, p=pboot(d)))
    if not spec.startswith('abs'):
        continue
    kk = int(spec[3:])
    for tag, M in R['models'].items():
        cols_d = M['dev'][['uid', 'jury_oof', 'base_oof']].assign(judged=1)
        cols_t = M['test'][['uid', 'jury_p', 'base_p']].assign(judged=1)
        dv = a7.ids['dev'].merge(cols_d, on='uid', how='left').fillna({'jury_oof': 1.0, 'base_oof': 1.0, 'judged': 0})
        te = a7.ids['test'].merge(cols_t, on='uid', how='left').fillna({'jury_p': 1.0, 'base_p': 1.0, 'judged': 0}).reset_index(drop=True)
        bfull = a7.boot_idx(te.repo.values, rng)

        def eff(df, pcol, tau):
            ab = (1 - df[pcol].values) > tau
            return np.where(ab, 0, df.target.values).sum() / np.where(ab, kk, df.n_ai_total.values).sum()

        taus = np.round(np.linspace(0.30, 0.99, 70), 4)
        none = lambda df: df.target.sum() / df.n_ai_total.sum()
        best = {nm: max(taus, key=lambda t: eff(dv, pd_, t)) for nm, pd_ in (('jury', 'jury_oof'), ('base', 'base_oof'))}
        cap = te.assign(capp=np.where(te.judged.values == 1, 0.0, 1.0))
        rel = {nm: np.array([eff(te.iloc[ii], pt, best[nm]) / none(te.iloc[ii]) - 1 for ii in bfull])
               for nm, pt in (('jury', 'jury_p'), ('base', 'base_p'))}
        rel['cap'] = np.array([eff(cap.iloc[ii], 'capp', 0.5) / none(te.iloc[ii]) - 1 for ii in bfull])
        point = {'jury': eff(te, 'jury_p', best['jury']) / none(te) - 1, 'base': eff(te, 'base_p', best['base']) / none(te) - 1,
                 'cap': eff(cap, 'capp', 0.5) / none(te) - 1}
        for a, b in (('jury', None), ('base', None), ('cap', None), ('jury', 'base'), ('jury', 'cap')):
            d = rel[a] if b is None else rel[a] - rel[b]
            lo, hi = a7.ci(d)
            rows.append(dict(kind='h4_rel_gain', prefix=spec, a=f'{tag}_{a}', b=(f'{tag}_{b}' if b else 'no_abort'), n=len(te),
                             n_pos=int(te.target.sum()), est=point[a] - (point[b] if b else 0), lo=lo, hi=hi, p=pboot(d),
                             tau=float(best.get(a, np.nan)) if b is None and a != 'cap' else np.nan))
        for t in taus:
            curves.append(dict(prefix=spec, model=tag, tau=float(t), jury_test=eff(te, 'jury_p', t) / none(te) - 1,
                               base_test=eff(te, 'base_p', t) / none(te) - 1, jury_dev=eff(dv, 'jury_oof', t) / none(dv) - 1,
                               base_dev=eff(dv, 'base_oof', t) / none(dv) - 1, tau_jury=float(best['jury']), tau_base=float(best['base'])))

df = pd.DataFrame(rows)
# Confirmatory family (Holm): H1 (e4b_jury-base), H2a (e4b_jury-e4b_hol), H2b (e4b_stack_jury-base), H4 (e4b jury vs no abort)
# at k=10 and k=20 -> 8 contrasts.
conf = ((df.kind == 'delta_auroc') & df.prefix.isin(['abs10', 'abs20']) &
        (((df.a == 'e4b_jury') & (df.b == 'base')) | ((df.a == 'e4b_jury') & (df.b == 'e4b_hol')) |
         ((df.a == 'e4b_stack_jury') & (df.b == 'base')))) | \
       ((df.kind == 'h4_rel_gain') & df.prefix.isin(['abs10', 'abs20']) & (df.a == 'e4b_jury') & (df.b == 'no_abort'))
df['confirmatory'] = conf
fam = df[conf].sort_values('p')
m = len(fam)
adj, run = [], 0.0
for i, p in enumerate(fam.p.values):
    run = max(run, min(1.0, (m - i) * p))
    adj.append(run)
df.loc[fam.index, 'p_holm'] = adj
df.to_csv(f'{OUT}/contrasts.csv', index=False)
pd.DataFrame(curves).to_csv(f'{OUT}/tau_curves.csv', index=False)
pd.concat(preds).to_csv(f'{OUT}/test_predictions.csv', index=False)
pd.concat(devpreds).to_csv(f'{OUT}/dev_predictions_all_models.csv', index=False)
pd.set_option('display.width', 250)
print(df[df.kind != 'auroc'].round(4).to_string())
print(df[df.kind == 'auroc'].round(3).to_string())
