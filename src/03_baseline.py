"""E1 baselines: logistic regression and GBM. Split by repo (fixed 70/30) + grouped 5-fold CV.
AUROC with 95% CI from a repo-cluster bootstrap (1000 resamples)."""
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from common import *

rng = np.random.default_rng(SEED)
df = pd.read_csv(ROOT + '/out/features.csv')
META = ['instance_id', 'repo', 'model', 'target', 'prefix']


def models():
    return {'logreg': make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
            'gbm': HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)}


def boot(y, p, g, B=1000):
    repos = np.unique(g)
    idx = {r: np.where(g == r)[0] for r in repos}
    out = []
    for _ in range(B):
        pick = rng.choice(repos, len(repos))
        ii = np.concatenate([idx[r] for r in pick])
        if len(set(y[ii])) < 2:
            continue
        out.append(roc_auc_score(y[ii], p[ii]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


res = []
for pref, d in df.groupby('prefix'):
    d = d.reset_index(drop=True)
    X = d.drop(columns=META).fillna(0)
    y = d.target.values
    g = d.repo.values
    repos = np.array(sorted(set(g)))
    np.random.default_rng(SEED).shuffle(repos)
    test_repos = set(repos[:int(0.3 * len(repos))])
    te = np.array([r in test_repos for r in g])
    assert not (set(g[te]) & set(g[~te])), 'repo in both splits'
    for name in ('logreg', 'gbm'):
        m = models()[name].fit(X[~te], y[~te])
        p = m.predict_proba(X[te])[:, 1]
        a = roc_auc_score(y[te], p)
        lo, hi = boot(y[te], p, g[te])
        oof = np.zeros(len(d))
        for tr_, te_ in GroupKFold(5).split(X, y, g):
            oof[te_] = models()[name].fit(X.iloc[tr_], y[tr_]).predict_proba(X.iloc[te_])[:, 1]
        ao = roc_auc_score(y, oof)
        lo2, hi2 = boot(y, oof, g)
        res.append(dict(prefix=pref, model=name, n=len(d), n_pos=int(y.sum()), repos=len(repos), n_test=int(te.sum()),
                        auroc_holdout=a, ci_lo=lo, ci_hi=hi, auroc_cv5=ao, cv_ci_lo=lo2, cv_ci_hi=hi2))
        print(res[-1], flush=True)
r = pd.DataFrame(res)
r.to_csv(ROOT + '/out/e1_results.csv', index=False)
print(r.round(3).to_string())
