import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import confusion_matrix,balanced_accuracy_score
from compe.data import load_train,feature_groups
from compe.validation import make_folds,best_threshold
R=Path('experiments/seleccion/calibracion/resultados'); O=Path('experiments/seleccion/revision');O.mkdir(exist_ok=True)
df=load_train(); y=df.canceled_job.to_numpy();g=feature_groups(df).to_numpy(); folds=make_folds(df); manifest=json.loads((R/'manifiesto_resultados.json').read_text());assert hashlib.sha256(Path('data/raw/train.csv').read_bytes()).hexdigest()==manifest['hash_train_csv']
p=pd.read_csv(R/'predicciones_externas.csv');ms=pd.read_csv(R/'metricas_externas_globales.csv');aud=[]
for (m,a,s),d in p.groupby(['modelo','assigned','escenario']):
 d=d.sort_values('fila_train');ix=d.fila_train.to_numpy();assert not d.fila_train.duplicated().any();assert np.array_equal(y[ix],d.y_true);assert np.array_equal(d.y_pred,d.score>=d.umbral)
 if s=='cv':expected=np.arange(len(df))
 elif s=='sin_locked':expected=np.flatnonzero(df.reservation_commitment!='locked_allocation')
 elif s=='eval_PRT':expected=np.flatnonzero(df.client_region=='PRT')
 elif s=='eval_resto':expected=np.flatnonzero(df.client_region!='PRT')
 else:expected=np.flatnonzero(df.execution_year==int(s.rsplit('_',1)[1]))
 assert np.array_equal(ix,expected)
 met=ms[(ms.modelo==m)&(ms.assigned==a)&(ms.escenario==s)].iloc[0];cm=confusion_matrix(d.y_true,d.y_pred).ravel();assert np.array_equal(cm,met[['TN','FP','FN','TP']].to_numpy());ba=balanced_accuracy_score(d.y_true,d.y_pred);assert np.isclose(ba,met.balanced_accuracy)
 if s=='cv':
  for k,(tr,va) in enumerate(folds):assert np.array_equal(d[d.fold_externo==k].fila_train,va)
 aud.append(dict(modelo=m,assigned=a,escenario=s,n=len(d),ba=ba,verificado=True))
pd.DataFrame(aud).to_csv(O/'auditoria.csv',index=False)
# Validate actual cached inner split groups and thresholds, not only exported summaries.
for f in Path('experiments/seleccion/calibracion/cache').glob('interno__*.npz'):
 z=np.load(f);idx=z['train_idx'];fid=z['fold_interno'];assert np.isfinite(z['oof_interno']).all();assert len(fid)==len(idx)
 for k in range(3):assert not np.intersect1d(g[idx[fid==k]],g[idx[fid!=k]]).size
 t,_=best_threshold(y[idx],z['oof_interno']);assert np.isclose(t,z['umbral'])
warn={}
for f in Path('experiments/seleccion/calibracion/cache').glob('*.json'):
 j=json.loads(f.read_text())
 for k,v in j.get('advertencias',{}).items():warn[k]=warn.get(k,0)+v
(O/'advertencias.json').write_text(json.dumps(warn,indent=2))
# Pairwise conditional cluster bootstrap of fixed calibrated predictions.
cv=p[p.escenario=='cv'];keys=list(cv.groupby(['modelo','assigned']).groups);win=('hist_gradient_boosting',False);w=cv[(cv.modelo==win[0])&(~cv.assigned)].sort_values('fila_train');wp=w.y_pred.to_numpy();G=g.max()+1;counts=np.stack([np.bincount(g,weights=y==c,minlength=G) for c in [0,1]],axis=1);paired=[];boot=[]
for m,a in keys:
 d=cv[(cv.modelo==m)&(cv.assigned==a)].sort_values('fila_train');pred=d.y_pred.to_numpy();delta=np.stack([np.bincount(g,weights=((wp==c).astype(float)-(pred==c))*(y==c),minlength=G) for c in [0,1]],axis=1);rng=np.random.default_rng(42);v=[]
 for _ in range(2000):
  ix=rng.integers(G,size=G);v.append(.5*np.sum(delta[ix].sum(axis=0)/counts[ix].sum(axis=0)))
 for k,(_,va) in enumerate(folds):paired.append(dict(comparador=m,assigned=a,fold=k,diferencia=balanced_accuracy_score(y[va],wp[va])-balanced_accuracy_score(y[va],pred[va])))
 boot.append(dict(comparador=m,assigned=a,diferencia=balanced_accuracy_score(y,wp)-balanced_accuracy_score(y,pred),ic025=np.quantile(v,.025),ic975=np.quantile(v,.975)))
pd.DataFrame(paired).to_csv(O/'diferencias_folds.csv',index=False);pd.DataFrame(boot).to_csv(O/'bootstrap_grupos.csv',index=False)
print('AUDITORIA OK',len(aud),'grupos modelo/variante/escenario');print(pd.DataFrame(boot).to_string(index=False));print('WARNINGS',warn)
# Threshold sensitivity is descriptive, no threshold selection using external labels.
sens=[]
for (m,a),d in cv.groupby(['modelo','assigned']):
 d=d.sort_values('fila_train');s=d.score.to_numpy();t=d.umbral.to_numpy();base=balanced_accuracy_score(y,s>=t)
 for shift in [-.05,-.025,0,.025,.05]:
  th=t.copy()
  for k,(_,va) in enumerate(folds):
   q=np.mean(s[va]<t[va][0]);th[va]=np.quantile(s[va],np.clip(q+shift,0,1)) if shift else t[va]
  sens.append(dict(modelo=m,assigned=a,desplazamiento_cuantil=shift,ba=balanced_accuracy_score(y,s>=th),caida=base-balanced_accuracy_score(y,s>=th)))
pd.DataFrame(sens).to_csv(O/'sensibilidad_descriptiva.csv',index=False)
print('LOCKED BA',balanced_accuracy_score(y,df.reservation_commitment=='locked_allocation'))
print(ms[(ms.escenario=='cv')&(~ms.assigned)].to_string(index=False))
