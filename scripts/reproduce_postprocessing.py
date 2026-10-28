import json, hashlib, datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'

def main():
 tp=pd.read_csv(DATA/'test_predictions.csv'); dp=pd.read_csv(DATA/'dev_predictions_all_models.csv')
 t=tp[(tp.model=='e4b')&(tp.prefix=='abs10')].copy().reset_index(drop=True)
 d=dp[(dp.model=='e4b')&(dp.prefix=='abs10')].copy()
 assert not set(d.repo)&set(t.repo),'repository leakage'
 assert t.uid.is_unique and d.uid.is_unique
 iso=IsotonicRegression(out_of_bounds='clip').fit(d.hol_p,d.target)
 preds={'holistic':t.hol_p.to_numpy(),'isotonic':iso.predict(t.hol_p),'jury':t.jury_p.to_numpy(),'features':t.base_p.to_numpy(),'hybrid_50_50':(t.jury_p.to_numpy()+t.base_p.to_numpy())/2}
 y=t.target.to_numpy(); groups=t.repo.to_numpy(); unique=np.unique(groups)
 rng=np.random.default_rng(20261028); by={g:np.flatnonzero(groups==g) for g in unique}
 samples=[np.concatenate([by[g] for g in rng.choice(unique,len(unique),replace=True)]) for _ in range(2000)]
 def met(y,p):return {'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p))}
 out={'kind':'new_exploratory_postprocessing_and_replay','n_judged_test':len(t),'test_repositories':len(unique),'dev_test_repository_overlap':0,'metrics':{k:met(y,p) for k,p in preds.items()},'contrasts':{}}
 for a,b in [('isotonic','holistic'),('hybrid_50_50','jury'),('hybrid_50_50','features')]:
  ds=[roc_auc_score(y[s],preds[a][s])-roc_auc_score(y[s],preds[b][s]) for s in samples if len(np.unique(y[s]))==2]
  out['contrasts'][a+'-vs-'+b]={'delta_auroc':out['metrics'][a]['auroc']-out['metrics'][b]['auroc'],'paired_repository_ci95':np.quantile(ds,[.025,.975]).tolist()}
 tr=pd.read_csv(DATA/'test_runs.csv'); dr=pd.read_csv(DATA/'dev_runs.csv')
 assert not set(tr.repo)&set(dr.repo)
 def attach(r,p,j,b):
  z=r.merge(p[['uid',j,b]],on='uid',how='left',validate='one_to_one')
  z['judged']=z[j].notna().astype(int)
  return z.fillna({j:1.,b:1.})
 T=attach(tr,t,'jury_p','base_p'); D=attach(dr,d,'jury_oof','base_oof')
 def eff(z,col=None,tau=None,overhead=0):
  abort=np.zeros(len(z),bool) if col is None else 1-z[col].to_numpy()>tau
  steps=np.where(abort,10,z.n_ai_total.to_numpy())+overhead*z.judged.to_numpy()
  return float(np.where(abort,0,z.target.to_numpy()).sum()/steps.sum())
 taus=np.round(np.linspace(.30,.99,70),4)
 tau_j=max(taus,key=lambda v:eff(D,'jury_oof',v,1))
 tau_b=max(taus,key=lambda v:eff(D,'base_oof',v,1))
 baseline=eff(T)
 out['G2_replay']={'n_all_test':len(T),'tau_jury_dev':float(tau_j),'tau_features_dev':float(tau_b),'never_abort_efficiency':baseline,'jury_efficiency':eff(T,'jury_p',tau_j,1),'features_efficiency':eff(T,'base_p',tau_b,1),'overhead_steps_per_judged_run':1}
 rr=out['G2_replay'];rr['relative_gain_jury_vs_never_abort']=rr['jury_efficiency']/baseline-1
 rr['relative_gain_features_vs_never_abort']=rr['features_efficiency']/baseline-1
 # Full cohort bootstrap, not just judged cases.
 ug=np.unique(T.repo); idx={g:np.flatnonzero(T.repo.to_numpy()==g) for g in ug}
 gains=[]
 for _ in range(2000):
  ii=np.concatenate([idx[g] for g in rng.choice(ug,len(ug),replace=True)])
  z=T.iloc[ii]; gains.append(eff(z,'jury_p',tau_j,1)/eff(z)-1)
 rr['relative_gain_jury_ci95']=np.quantile(gains,[.025,.975]).tolist()
 out['source_hashes']={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['dev_predictions_all_models.csv','dev_runs.csv','test_predictions.csv','test_runs.csv']}
 out['generated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (ROOT/'results/postprocessing_experiments.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()

