"""Read-only audit of the completed stacking experiment; no model fitting or test.csv."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import stacking as S
from compe.data import load_train
from compe.validation import best_threshold

ROOT=Path('experiments/seleccion/stacking_resultados/a92fbb4e4778849e');R=ROOT/'resultados'
OUT=Path('experiments/seleccion/revision_stacking');OUT.mkdir(exist_ok=True)
df=load_train();params=json.loads(S.C.PARAMETROS.read_text());e=S.Experiment(df,params,ROOT.parent,S.C._verificar_train(S.C.cfg.DATA_RAW/'train.csv'))
assert e.run==ROOT.resolve() or e.run.resolve()==ROOT.resolve()
assert e.identity==json.loads((ROOT/'manifiesto.json').read_text())
p=pd.read_csv(R/'predicciones_externas.csv');status=pd.read_csv(R/'estado.csv');assert len(status)==15 and status.completo.all()
audit=[];base=np.zeros((len(df),3));warn=Counter();sources=Counter()
for scenario,k,tr,va in e.tasks('todo'):
 e.check_split(tr,va);z=e.cached(e.key('result','three',tr,va));assert z is not None
 assert np.array_equal(z['train_idx'],tr) and np.array_equal(z['eval_idx'],va)
 assert z['inner_score'].shape==(len(tr),3)
 np.testing.assert_allclose(z['score'][:,0],z['base_external'][:,1])
 np.testing.assert_allclose(z['score'][:,1],z['base_external'].mean(axis=1))
 logits=z['base_external']@z['meta_coef'][0]+z['meta_intercept'][0]
 np.testing.assert_allclose(z['score'][:,2],1/(1+np.exp(-logits)))
 if scenario=='cv':base[va]=z['base_external']
 for j,name in enumerate(S.STRATEGIES):
  t,_=best_threshold(e.y[tr],z['inner_score'][:,j]);assert np.isclose(t,z['threshold'][j])
  d=p[(p.modelo==name)&(p.escenario==scenario)]
  if k is not None:d=d[d.fold_externo==k]
  d=d.sort_values('fila_train');order=np.argsort(va)
  assert np.array_equal(d.fila_train,va[order]);assert np.array_equal(d.y_true,e.y[va][order])
  np.testing.assert_allclose(d.score,z['score'][order,j]);np.testing.assert_allclose(d.umbral,t)
 audit.append(dict(escenario=scenario,fold=k,filas=len(va),verificado=True))
mat,met,folds=S.M.calcular(p);old=pd.read_csv(R/'metricas_externas_globales.csv')
for _,r in met.iterrows():
 v=old[(old.modelo==r.modelo)&(old.escenario==r.escenario)].iloc[0]
 for col in S.M.METRICAS+['TN','FP','FN','TP','n']:assert np.isclose(r[col],v[col],equal_nan=True)
# HGB baseline exactly agrees with the previous independent calibration run, every scenario.
prev=pd.read_csv('experiments/seleccion/calibracion/resultados/predicciones_externas.csv')
prev=prev[(prev.modelo=='hist_gradient_boosting')&(~prev.assigned)]
for scenario,d in p[p.modelo=='histgb'].groupby('escenario'):
 a=d.sort_values('fila_train');b=prev[prev.escenario==scenario].sort_values('fila_train')
 assert np.array_equal(a.fila_train,b.fila_train) and np.array_equal(a.y_pred,b.y_pred)
 np.testing.assert_allclose(a.score,b.score,rtol=0,atol=1e-14)
 np.testing.assert_allclose(a.umbral,b.umbral,rtol=0,atol=1e-14)
for path in (ROOT/'cache').glob('*.json'):
 info=json.loads(path.read_text());assert info['sha256_npz']==S.C.hash_archivo(path.with_suffix('.npz'))
 sources[info['source']]+=1;warn.update(info.get('warnings',{}))
pd.DataFrame(audit).to_csv(OUT/'auditoria.csv',index=False)
pd.DataFrame(np.corrcoef(base.T),index=S.BASES,columns=S.BASES).to_csv(OUT/'correlacion_bases.csv')
# Descriptive score sensitivity: shift training-derived quantile cutoffs, never choose new ones.
sens=[]
for scenario,k,tr,va in e.tasks('cv'):
 z=e.cached(e.key('result','three',tr,va))
 for j,name in enumerate(S.STRATEGIES):
  inner=z['inner_score'][:,j];q=np.mean(inner<z['threshold'][j])
  for dq in [-.05,-.025,0,.025,.05]:
   t=z['threshold'][j] if dq==0 else np.quantile(inner,np.clip(q+dq,0,1))
   sens.append(dict(modelo=name,fold=k,desplazamiento_cuantil=dq,umbral=t,ba=S.C.ba_segura(e.y[va],z['score'][:,j]>=t)))
pd.DataFrame(sens).to_csv(OUT/'sensibilidad_folds.csv',index=False)
ss=pd.DataFrame(sens).groupby(['modelo','desplazamiento_cuantil']).ba.mean().reset_index();ss.to_csv(OUT/'sensibilidad_media.csv',index=False)
# Error transitions relative to HGB, same observations, threshold policy unchanged.
cv=p[p.escenario=='cv'];ref=cv[cv.modelo=='histgb'].sort_values('fila_train');yy=ref.y_true.to_numpy();pp=ref.y_pred.to_numpy();errs=[]
for name in ['promedio','stacking']:
 d=cv[cv.modelo==name].sort_values('fila_train');qq=d.y_pred.to_numpy()
 errs.append(dict(modelo=name,corrige_error_histgb=int(((pp!=yy)&(qq==yy)).sum()),
                  rompe_acierto_histgb=int(((pp==yy)&(qq!=yy)).sum()),
                  desacuerdo=float(np.mean(pp!=qq))))
pd.DataFrame(errs).to_csv(OUT/'cambios_errores.csv',index=False)
info=dict(auditoria='OK',tareas=15,conjuntos_metricas=len(met),filas_predicciones=len(p),fuentes=dict(sources),advertencias=dict(warn))
(OUT/'auditoria.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2));print(pd.DataFrame(errs).to_string(index=False));print(ss.to_string(index=False));print(np.corrcoef(base.T))
