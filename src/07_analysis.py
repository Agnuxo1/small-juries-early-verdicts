"""E2-E4: preregistered analysis. AUROC, ECE (10 bins), Brier with repo-cluster bootstrap (2,000, paired), H1-H4.
Usage (server, .venv): python 07_analysis.py <e2_jsonl_dir> <output_dir>
Nothing is tuned on the test set: logistic aggregation, baseline choice and threshold tau come only from the development split."""
import sys, os, json, glob
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

SEED = 20261027
B = 2000
ROOT = os.environ.get('SJEV_ROOT', '/home/ai/proyectos/gemma4-paper')  # data/models/outputs root
IN = sys.argv[1] if len(sys.argv) > 1 else ROOT + '/out/e2'
OUT = sys.argv[2] if len(sys.argv) > 2 else ROOT + '/out/e2_analysis'
os.makedirs(OUT, exist_ok=True)
DIMS = ['localization', 'reproduction', 'coherence', 'test_evidence', 'loop', 'scope']
MODELS = ['e2b', 'e4b']
META = ['instance_id', 'repo', 'model', 'target', 'prefix', 'uid']
ids = {s: pd.read_csv(f'{ROOT}/out/ids_{n}.csv') for s, n in [('test', 'test600'), ('dev', 'dev_all')]}
for _d in ids.values():
    _d['uid'] = _d.instance_id + '|' + _d.target.astype(int).astype(str)
feat = pd.read_csv(ROOT + '/out/features.csv')
feat['uid'] = feat.instance_id + '|' + feat.target.astype(int).astype(str)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ece(y, p, nb=10):
    bins = np.minimum((p * nb).astype(int), nb - 1)
    e = 0.0
    for b in range(nb):
        m = bins == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def load(tag, s, spec, judge):
    fn = f'{IN}/{tag}__{s}__{spec}__{judge}.jsonl'
    if not os.path.exists(fn):
        return None
    rows = [json.loads(l) for l in open(fn)]
    return pd.DataFrame(rows).drop_duplicates('uid', keep='last')


def judged(tag, s, spec):
    """Per-item table with raw scores; invalid JSON outputs are imputed (hol: dev mean... here 0.25; jury: 3 in each dim) and counted."""
    h, j = load(tag, s, spec, 'holistic'), load(tag, s, spec, 'jury')
    if h is None or j is None:
        return None, {}
    d = ids[s][['uid', 'instance_id', 'repo', 'target', 'n_ai_total']].merge(h[['uid', 'parsed']].rename(columns={'parsed': 'hp'}), on='uid') \
        .merge(j[['uid', 'parsed']].rename(columns={'parsed': 'jp'}), on='uid')
    inv = {'hol_invalid': int(d.hp.isna().sum()), 'jury_invalid': int(d.jp.isna().sum()), 'n': len(d)}
    d['hol_p'] = [x['p_success'] if isinstance(x, dict) else 0.25 for x in d.hp]
    for k in DIMS:
        d[k] = [x[k] if isinstance(x, dict) else 3.0 for x in d.jp]
    return d.drop(columns=['hp', 'jp']), inv


def fit_jury(dev, C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(dev[DIMS], dev.target)


def pick_C(dev):
    best = (-1, 1.0)
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        oof = np.zeros(len(dev))
        for tr, te in GroupKFold(5).split(dev, dev.target, dev.repo):
            oof[te] = fit_jury(dev.iloc[tr], C).predict_proba(dev.iloc[te][DIMS])[:, 1]
        a = roc_auc_score(dev.target, oof)
        if a > best[0]:
            best = (a, C)
    return best[1]


def mk(name):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)) if name == 'logreg' else \
        HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)


def baseline(spec, dev, test):
    """F0 baseline without an LLM, trained ONLY on dev; logreg/gbm chosen by OOF AUROC on dev. Returns (p_test, oof_dev, name)."""
    f = feat[feat.prefix == spec].set_index('uid')
    cols = [c for c in f.columns if c not in META]
    Xd, Xt = f.loc[dev.uid, cols].fillna(0), f.loc[test.uid, cols].fillna(0)
    best = None
    for name in ('logreg', 'gbm'):
        oof = np.zeros(len(dev))
        for tr, te in GroupKFold(5).split(Xd, dev.target, dev.repo):
            oof[te] = mk(name).fit(Xd.iloc[tr], dev.target.iloc[tr]).predict_proba(Xd.iloc[te])[:, 1]
        a = roc_auc_score(dev.target, oof)
        if best is None or a > best[0]:
            best = (a, name, oof)
    m = mk(best[1]).fit(Xd, dev.target)
    return m.predict_proba(Xt)[:, 1], best[2], best[1], best[0]


def platt(s_dev, y_dev, s_test):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(np.asarray(s_dev)).reshape(-1, 1), y_dev)
    return m.predict_proba(logit(np.asarray(s_test)).reshape(-1, 1))[:, 1]


def evaluate(spec):
    """Returns a dict with per-item test scores and development metadata."""
    R = {'spec': spec, 'models': {}}
    base_done = False
    for tag in MODELS:
        dv, inv_d = judged(tag, 'dev', spec)
        te, inv_t = judged(tag, 'test', spec)
        if dv is None or te is None:
            continue
        dv = dv.reset_index(drop=True)
        te = te.reset_index(drop=True)
        C = pick_C(dv)
        jm = fit_jury(dv, C)
        oof_j = np.zeros(len(dv))
        for tr, t_ in GroupKFold(5).split(dv, dv.target, dv.repo):
            oof_j[t_] = fit_jury(dv.iloc[tr], C).predict_proba(dv.iloc[t_][DIMS])[:, 1]
        te['jury_p'] = jm.predict_proba(te[DIMS])[:, 1]
        dv['jury_oof'] = oof_j
        te['hol_p_platt'] = platt(dv.hol_p, dv.target, te.hol_p)
        bp, boof, bname, bdev = baseline(spec, dv, te)
        te['base_p'] = bp
        dv['base_oof'] = boof
        st = LogisticRegression(C=1e6, max_iter=1000).fit(np.c_[logit(oof_j), logit(boof)], dv.target)
        te['stack_p'] = st.predict_proba(np.c_[logit(te.jury_p), logit(te.base_p)])[:, 1]
        st2 = LogisticRegression(C=1e6, max_iter=1000).fit(np.c_[logit(dv.hol_p), logit(boof)], dv.target)
        te['stack_hol_p'] = st2.predict_proba(np.c_[logit(te.hol_p), logit(te.base_p)])[:, 1]
        R['models'][tag] = dict(dev=dv, test=te, C=C, base_name=bname, base_dev_auc=bdev, inv_dev=inv_d, inv_test=inv_t,
                                jury_coef=dict(zip(DIMS, [float(x) for x in jm[-1].coef_[0]])))
    return R


def boot_idx(g, rng):
    repos = np.unique(g)
    idx = {r: np.where(g == r)[0] for r in repos}
    out = []
    for _ in range(B):
        pick = rng.choice(repos, len(repos))
        out.append(np.concatenate([idx[r] for r in pick]))
    return out


def auc(y, s, ii):
    yy = y[ii]
    if yy.min() == yy.max():
        return np.nan
    return roc_auc_score(yy, s[ii])


def ci(v):
    v = np.asarray(v)
    v = v[~np.isnan(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def verdict_gt(lo, hi):
    return 'si' if lo > 0 else ('no' if hi <= 0 else 'indeterminado')


def main():
    rng = np.random.default_rng(SEED)
    out = {'tables': [], 'H': {}}
    preds = []
    results = {}
    for spec in ['abs10', 'abs20', 'frac33', 'frac66']:
        R = evaluate(spec)
        if not R['models']:
            continue
        results[spec] = R
        t0 = next(iter(R['models'].values()))['test']
        y = t0.target.values
        g = t0.repo.values
        bidx = boot_idx(g, rng)
        R['bidx'] = bidx
        R['y'] = y
        scores = {'base': t0.base_p.values}
        for tag, M in R['models'].items():
            te = M['test']
            assert (te.uid.values == t0.uid.values).all()
            scores[f'{tag}_hol'] = te.hol_p.values
            scores[f'{tag}_hol_platt'] = te.hol_p_platt.values
            scores[f'{tag}_jury'] = te.jury_p.values
            scores[f'{tag}_stack_jury'] = te.stack_p.values
            scores[f'{tag}_stack_hol'] = te.stack_hol_p.values
            for dname in DIMS:
                scores[f'{tag}_dim_{dname}'] = te[dname].values
        R['scores'] = scores
        R['aucs'] = {k: np.array([auc(y, v, ii) for ii in bidx]) for k, v in scores.items()}
        for k, v in scores.items():
            a = roc_auc_score(y, v)
            lo, hi = ci(R['aucs'][k])
            prob = k in ('base',) or k.endswith(('_hol', '_hol_platt', '_jury', '_stack_jury', '_stack_hol'))
            row = dict(prefix=spec, score=k, n=len(y), n_pos=int(y.sum()), auroc=a, auc_lo=lo, auc_hi=hi)
            if prob:
                bs = [(ece(y[ii], v[ii]), float(np.mean((v[ii] - y[ii]) ** 2))) for ii in bidx[:500]]
                row.update(ece=ece(y, v), ece_lo=ci([b[0] for b in bs])[0], ece_hi=ci([b[0] for b in bs])[1],
                           brier=float(np.mean((v - y) ** 2)), brier_lo=ci([b[1] for b in bs])[0], brier_hi=ci([b[1] for b in bs])[1])
            out['tables'].append(row)
        for tag, M in R['models'].items():
            te = M['test']
            preds.append(te.assign(model=tag, prefix=spec, cohort='test'))
            M['dev'].assign(model=tag, prefix=spec, cohort='dev').to_csv(f'{OUT}/dev_predictions_{tag}_{spec}.csv', index=False)
            out['tables'].append(dict(prefix=spec, score=f'{tag}_META', n=len(te), n_pos=int(te.target.sum()),
                                      C=M['C'], base=M['base_name'], base_dev_auc=M['base_dev_auc'],
                                      inv_test_hol=M['inv_test']['hol_invalid'], inv_test_jury=M['inv_test']['jury_invalid'],
                                      inv_dev_hol=M['inv_dev']['hol_invalid'], inv_dev_jury=M['inv_dev']['jury_invalid'],
                                      coef=json.dumps(M['jury_coef'])))
    pd.concat(preds).to_csv(f'{OUT}/predicciones_test_por_item.csv', index=False)

    def diff(R, a, b, delta=0.0):
        d = R['aucs'][a] - R['aucs'][b]
        lo, hi = ci(d)
        return float(roc_auc_score(R['y'], R['scores'][a]) - roc_auc_score(R['y'], R['scores'][b])), lo, hi

    H = out['H']
    for k in ('abs20', 'abs10'):
        if k not in results or 'e4b' not in results[k]['models']:
            continue
        R = results[k]
        d1 = diff(R, 'e4b_jury', 'base')
        H[f'H1_{k}'] = dict(delta=d1[0], lo=d1[1], hi=d1[2], veredicto=verdict_gt(d1[1], d1[2]))
        for tag in MODELS:
            if tag not in R['models']:
                continue
            da = diff(R, f'{tag}_jury', f'{tag}_hol')
            ds = diff(R, f'{tag}_stack_jury', 'base')
            dsh = diff(R, f'{tag}_stack_hol', 'base')
            H[f'H2_{k}_{tag}'] = dict(jury_vs_hol=dict(delta=da[0], lo=da[1], hi=da[2]),
                                      stack_vs_base=dict(delta=ds[0], lo=ds[1], hi=ds[2]),
                                      stack_hol_vs_base=dict(delta=dsh[0], lo=dsh[1], hi=dsh[2]))
            leg_a = verdict_gt(da[1], da[2])
            leg_b = 'si' if ds[0] >= 0.02 else ('no' if ds[2] < 0.02 else 'indeterminado')
            H[f'H2_{k}_{tag}']['veredicto'] = 'si' if 'si' in (leg_a, leg_b) else ('no' if (leg_a == 'no' and leg_b == 'no') else 'indeterminado')
            H[f'H2_{k}_{tag}']['legs'] = [leg_a, leg_b]
    # H3: E2B >= 90% of E4B AUROC (jury and holistic) ; cost taken from the jsonl files
    cost = cost_table()
    out['cost'] = cost
    for k in ('abs20', 'abs10'):
        if k not in results or len(results[k]['models']) < 2:
            continue
        R = results[k]
        for j in ('jury', 'hol'):
            a2, a4 = R['aucs'][f'e2b_{j}'], R['aucs'][f'e4b_{j}']
            ratio = a2 / a4
            p2, p4 = roc_auc_score(R['y'], R['scores'][f'e2b_{j}']), roc_auc_score(R['y'], R['scores'][f'e4b_{j}'])
            lo, hi = ci(ratio)
            sel = [c for c in cost if c['set'] == 'test' and c['prefix'] == k and c['judge'] == ('jury' if j == 'jury' else 'holistic')]
            c2 = next((c for c in sel if c['model'] == 'e2b'), None)
            c4 = next((c for c in sel if c['model'] == 'e4b'), None)
            tr = (c2['tokens'] / c4['tokens']) if c2 and c4 else None
            sr = (c2['sec'] / c4['sec']) if c2 and c4 else None
            auc_ok = 'si' if lo >= 0.9 else ('no' if hi < 0.9 else 'indeterminado')
            cost_ok = 'si' if (tr is not None and tr < 0.5 and sr < 0.5) else 'no'
            H[f'H3_{k}_{j}'] = dict(auroc_e2b=p2, auroc_e4b=p4, ratio=p2 / p4, ratio_lo=lo, ratio_hi=hi, token_ratio=tr, sec_ratio=sr,
                                    leg_auroc=auc_ok, leg_cost=cost_ok,
                                    veredicto='si' if (auc_ok == 'si' and cost_ok == 'si') else ('no' if (auc_ok == 'no' or cost_ok == 'no') else 'indeterminado'))
    # H4: abort if P(failure) > tau, tau chosen on development with OOF predictions.
    # Simulation assumption: the freed budget is reinvested in other attempts at the mean efficiency (resolved per step).
    # Controls: (a) abort everything beyond k steps (fixed cap, no model); (b) policy using the no-LLM baseline.
    for k, kk in (('abs20', 20), ('abs10', 10)):
        if k not in results:
            continue
        for tag in MODELS:
            if tag not in results[k]['models']:
                continue
            R = results[k]
            M = R['models'][tag]
            cols_d = M['dev'][['uid', 'jury_oof', 'base_oof']].assign(judged=1)
            cols_t = M['test'][['uid', 'jury_p', 'base_p']].assign(judged=1)
            dv = ids['dev'].merge(cols_d, on='uid', how='left').fillna({'jury_oof': 1.0, 'base_oof': 1.0, 'judged': 0})
            te = ids['test'].merge(cols_t, on='uid', how='left').fillna({'jury_p': 1.0, 'base_p': 1.0, 'judged': 0}).reset_index(drop=True)
            bidx_full = boot_idx(te.repo.values, rng)

            def eff(df, pcol, tau, overhead=0.0):
                ab = (1 - df[pcol].values) > tau
                cost_ = np.where(ab, kk, df.n_ai_total.values) + overhead * df.judged.values
                res = np.where(ab, 0, df.target.values)
                return res.sum() / cost_.sum()

            taus = np.linspace(0.30, 0.99, 70)
            for ov in (0.0, 1.0):
                none_t = te.target.sum() / te.n_ai_total.sum()
                rels = {}
                row = dict(dev_eff_none=dv.target.sum() / dv.n_ai_total.sum(), test_eff_none=none_t)
                for name, pc_d, pc_t in (('jury', 'jury_oof', 'jury_p'), ('base', 'base_oof', 'base_p')):
                    best = max(taus, key=lambda t: eff(dv, pc_d, t, ov))
                    rel = [eff(te.iloc[ii], pc_t, best, ov) / (te.iloc[ii].target.sum() / te.iloc[ii].n_ai_total.sum()) - 1 for ii in bidx_full]
                    lo, hi = ci(rel)
                    rels[name] = np.array(rel)
                    row[name] = dict(tau=float(best), dev_eff=eff(dv, pc_d, best, ov), test_eff=eff(te, pc_t, best, ov),
                                     rel_gain=eff(te, pc_t, best, ov) / none_t - 1, lo=lo, hi=hi,
                                     aborted_frac_of_judged=float((((1 - te[pc_t].values) > best) & (te.judged.values == 1)).sum() / max(1, te.judged.sum())),
                                     veredicto=verdict_gt(lo, hi))
                te_cap = te.assign(capp=np.where(te.judged.values == 1, 0.0, 1.0))
                rel = [eff(te_cap.iloc[ii], 'capp', 0.5, ov) / (te.iloc[ii].target.sum() / te.iloc[ii].n_ai_total.sum()) - 1 for ii in bidx_full]
                row['cap_all_beyond_k'] = dict(rel_gain=eff(te_cap, 'capp', 0.5, ov) / none_t - 1, lo=ci(rel)[0], hi=ci(rel)[1])
                for name in ('jury', 'base'):
                    dlo, dhi = ci(rels[name] - np.array(rel))
                    row[name]['minus_cap'] = dict(delta=row[name]['rel_gain'] - row['cap_all_beyond_k']['rel_gain'], lo=dlo, hi=dhi)
                row['veredicto'] = row['jury']['veredicto']
                H[f'H4_{k}_{tag}_ov{int(ov)}'] = row
    json.dump(out, open(f'{OUT}/resultados.json', 'w'), indent=1, default=float)
    pd.DataFrame([r for r in out['tables']]).to_csv(f'{OUT}/tabla_metricas.csv', index=False)
    pd.DataFrame(cost).to_csv(f'{OUT}/coste.csv', index=False)
    print(json.dumps(H, indent=1, default=float))


def cost_table():
    rows = []
    for fn in sorted(glob.glob(IN + '/*__*__*__*.jsonl')):
        tag, s, spec, judge = os.path.basename(fn)[:-6].split('__')
        if tag.startswith('p'):
            continue
        d = pd.DataFrame([json.loads(l) for l in open(fn)])
        rows.append(dict(model=tag, set=s, prefix=spec, judge=judge, n=len(d), tokens=int(d.prompt_tokens.sum() + d.gen_tokens.sum()),
                         prompt_tokens=int(d.prompt_tokens.sum()), cached_tokens=int(d.cached_tokens.sum()), gen_tokens=int(d.gen_tokens.sum()),
                         sec=float(d.sec_est.sum()), invalid=int(d.parsed.isna().sum())))
    return rows


if __name__ == '__main__':
    main()
