import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json,time,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from compe.data import load_train,feature_groups
from compe.config import FEATURE_COLS
from compe.models import MODELS
from compe.pipeline import make_pipeline
from compe.validation import make_folds,cv_scores,metrics,best_threshold,_ba,model_scores,subgroup_table
warnings.filterwarnings('ignore')
OUT=Path('experiments/seleccion'); df=load_train(); y=df.canceled_job.to_numpy(); folds=make_folds(df); names=['logistic','ridge','linear_svc','knn','random_forest','bagging','gradient_boosting','hist_gradient_boosting']; finals=['logistic','gradient_boosting','hist_gradient_boosting']
def save(rows,name): pd.DataFrame(rows).to_csv(OUT/name,index=False)
def pipe(n,p,a): return make_pipeline(MODELS[n].build(p),MODELS[n].variante,use_assigned=a)
params={}; audit=[]; original=[]; curves=[]; confirmations=[]; limits=[]
for n in names:
 d=Path('experiments')/n;m=json.loads((d/'metricas.json').read_text());b=json.loads((d/'base.json').read_text());p=json.loads((d/'mejores_parametros.json').read_text())['params']; rows=pd.concat([pd.read_csv(d/f) for f in ['busqueda.csv','refinamiento.csv']],ignore_index=True).drop_duplicates('params'); c=rows[rows.ba_val>=m['ba_val']-m['ba_std']/np.sqrt(5)].sort_values('gap').iloc[0]; chosen=p;reason='Se conserva: reducción de gap alternativa menor a 0,005.'
 if m['gap']-c.gap>=.005: chosen=json.loads(c.params);reason='Menor gap dentro de un error estándar de la elegida original.'
 if n=='hist_gradient_boosting': chosen={};reason='Default más simple (100 iteraciones); mejora ajustada inferior a dispersión, gap similar.'
 params[n]=chosen
 audit.append(dict(modelo=n,params=json.dumps(chosen),motivo=reason,ba_original=m['ba_val'],gap_original=m['gap'],ba_base=b['ba_val'],gap_base=b['gap'],ba_alternativa=c.ba_val,gap_alternativa=c.gap,params_alternativa=c.params))
 original.append(m)
 for f,target in [('curva_validacion.csv',curves),('confirmacion.csv',confirmations)]:
  for r in pd.read_csv(d/f).to_dict('records'): target.append(dict(modelo=n,**r))
 for lim in [.02,.03,.04,.05]:
  ok=rows[rows.gap<=lim];limits.append(dict(modelo=n,limite=lim,n=len(ok),mejor_ba=ok.ba_val.max(),perdida_vs_original=m['ba_val']-ok.ba_val.max()))
(OUT/'parametros.json').write_text(json.dumps(params,indent=2));save(audit,'seleccion_configuraciones.csv');save(original,'originales.csv');save(curves,'curvas.csv');save(confirmations,'confirmaciones.csv');save(limits,'limites_gap.csv')
summary=[];subs=[];sensitivity=[];scores={};preds={}
for n in names:
 for a in [True,False]:
  path=OUT/f'{n}_{a}.npz';mp=OUT/f'{n}_{a}.json'
  if path.exists(): s=np.load(path)['oof'];m=json.loads(mp.read_text())
  else:
   res=cv_scores(pipe(n,params[n],a),df,folds,MODELS[n].score);s=res.oof;m=metrics(df,res,folds);np.savez(path,oof=s);mp.write_text(json.dumps(m))
  scores[n,a]=s;preds[n,a]=s>=m['umbral'];summary.append(dict(modelo=n,assigned=a,**m));subs.extend(dict(modelo=n,assigned=a,**r) for r in subgroup_table(df,s,m['umbral']).to_dict('records'))
  q=np.mean(s<m['umbral'])
  for dq in np.linspace(-.10,.10,21):
   t=np.quantile(s,np.clip(q+dq,0,1));sensitivity.append(dict(modelo=n,assigned=a,desplazamiento_cuantil=dq,umbral=t,ba=_ba(y,s>=t)))
  print('OOF',n,a,m['ba_val'],flush=True)
 # common-fold control of recommended config
 dest=OUT/f'{n}_comun.json'
 if not dest.exists():
  cf=make_folds(df,grouped=False);res=cv_scores(pipe(n,params[n],True),df,cf,MODELS[n].score);dest.write_text(json.dumps(metrics(df,res,cf)))
save(summary,'metricas_recomendadas.csv');save(subs,'subgrupos.csv');save(sensitivity,'sensibilidad.csv')
# References: decision stump per raw input, selected on training BA only. Balanced weights.
ref=[];rp=np.zeros(len(df),bool)
for k,(tr,va) in enumerate(folds):
 best=(-1,None,None)
 for col in FEATURE_COLS:
  enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1); x=df[[col]]
  if not pd.api.types.is_numeric_dtype(x[col]): xt=enc.fit_transform(x.iloc[tr].fillna('(nulo)').astype(str));xv=enc.transform(x.iloc[va].fillna('(nulo)').astype(str))
  else: med=x.iloc[tr].median();xt=x.iloc[tr].fillna(med).to_numpy();xv=x.iloc[va].fillna(med).to_numpy()
  tree=DecisionTreeClassifier(max_depth=1,class_weight='balanced',random_state=42).fit(xt,y[tr]);ba=_ba(y[tr],tree.predict(xt).astype(bool))
  if ba>best[0]: best=(ba,col,tree.predict(xv).astype(bool))
 rp[va]=best[2];ref.append(dict(fold=k,variable=best[1],ba=_ba(y[va],best[2]),ba_locked=_ba(y[va],(df.reservation_commitment.iloc[va]=='locked_allocation').to_numpy())))
save(ref,'referencias.csv')
# Paired folds and cluster bootstrap, row-weighted BA with groups sampled equally.
best=max(names,key=lambda n:_ba(y,preds[n,True])); paired=[];boot=[];g=feature_groups(df).to_numpy();G=g.max()+1;rng=np.random.default_rng(42)
counts=np.stack([np.bincount(g,weights=(y==c),minlength=G) for c in [0,1]],axis=1)
for n in names:
 for k,(_,va) in enumerate(folds):paired.append(dict(modelo=n,referente=best,fold=k,ba=_ba(y[va],preds[n,True][va]),diferencia=_ba(y[va],preds[n,True][va])-_ba(y[va],preds[best,True][va])))
 delta=np.stack([np.bincount(g,weights=((preds[n,True]==c).astype(float)-(preds[best,True]==c))*(y==c),minlength=G) for c in [0,1]],axis=1);vals=[]
 for _ in range(1000):
  ix=rng.integers(G,size=G);vals.append(.5*np.sum(delta[ix].sum(axis=0)/counts[ix].sum(axis=0)))
 boot.append(dict(modelo=n,referente=best,diferencia=_ba(y,preds[n,True])-_ba(y,preds[best,True]),inferior=np.quantile(vals,.025),superior=np.quantile(vals,.975)))
save(paired,'pareadas.csv');save(boot,'bootstrap.csv')
# External assessment with threshold fitted strictly inside training partition.
def isolated(n,a,train,val):
 train=train.reset_index(drop=True); inner=make_folds(train,n_splits=3);est=pipe(n,params[n],a);r=cv_scores(est,train,inner,MODELS[n].score);t,_=best_threshold(train.canceled_job.to_numpy(),r.oof);est.fit(train,train.canceled_job);s=model_scores(est,val,MODELS[n].score);return s>=t,t
nested=[];robust=[]
for n in names:
 for a in ([True,False] if n in finals else [True]):
  pred=np.zeros(len(df),bool)
  for k,(tr,va) in enumerate(folds):
   dest=OUT/f'externo_{n}_{a}_{k}.npz'
   if dest.exists(): z=np.load(dest);pp=z['pred'];t=float(z['umbral'])
   else: pp,t=isolated(n,a,df.iloc[tr],df.iloc[va]);np.savez(dest,pred=pp,umbral=t)
   pred[va]=pp;nested.append(dict(modelo=n,assigned=a,fold=k,ba=_ba(y[va],pp),umbral=t));print('EXTERNO',n,a,k,flush=True)
  nested.append(dict(modelo=n,assigned=a,fold='global',ba=_ba(y,pred),umbral=np.nan));save(nested,'umbrales_internos.csv')
  if n not in finals: continue
  masks={f'anio_{v}':(df.execution_year==v).to_numpy() for v in sorted(df.execution_year.unique())};masks.update(region_PRT=(df.client_region=='PRT').to_numpy(),region_resto=(df.client_region!='PRT').to_numpy())
  for label,mask in masks.items():
   dest=OUT/f'robustez_{n}_{a}_{label}.npz'
   if dest.exists(): z=np.load(dest);pp=z['pred'];t=float(z['umbral'])
   else: pp,t=isolated(n,a,df.loc[~mask],df.loc[mask]);np.savez(dest,pred=pp,umbral=t)
   robust.append(dict(modelo=n,assigned=a,escenario=label,ba=_ba(y[mask],pp),filas=int(mask.sum()),umbral=t));print('ROBUSTEZ',n,a,label,flush=True);save(robust,'robustez.csv')
  sub=df[df.reservation_commitment!='locked_allocation'].reset_index(drop=True);sy=sub.canceled_job.to_numpy();sp=np.zeros(len(sub),bool)
  for k,(tr,va) in enumerate(make_folds(sub)):
   dest=OUT/f'sin_locked_{n}_{a}_{k}.npz'
   if dest.exists(): pp=np.load(dest)['pred']
   else: pp,t=isolated(n,a,sub.iloc[tr],sub.iloc[va]);np.savez(dest,pred=pp,umbral=t)
   sp[va]=pp
  robust.append(dict(modelo=n,assigned=a,escenario='entrenado_sin_locked',ba=_ba(sy,sp),filas=len(sub),umbral=np.nan));save(robust,'robustez.csv')
print('COMPLETO',flush=True)
