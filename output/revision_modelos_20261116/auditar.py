"""Audita modelos existentes sin fit, predict ni lectura de test.csv."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from compe.data import feature_groups
from compe.config import FEATURE_COLS
from compe.validation import make_folds,best_threshold

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def idxhash(a):return hashlib.sha256(np.asarray(a,dtype=np.int64).tobytes()).hexdigest()[:16]
def ba(y,p):return balanced_accuracy_score(y,p)

df=pd.read_csv(ROOT/'data/raw/train.csv');y=df.canceled_job.to_numpy();folds=make_folds(df);groups=feature_groups(df).to_numpy();N=len(df)
pred=pd.read_csv(ROOT/'experiments/seleccion/calibracion/resultados/predicciones_externas.csv')
metrics=pd.read_csv(ROOT/'experiments/seleccion/calibracion/resultados/metricas_externas_globales.csv')
selection=pd.read_csv(ROOT/'experiments/seleccion/seleccion_configuraciones.csv').set_index('modelo')
controls=pd.read_csv(ROOT/'experiments/seleccion/revision_stacking/comparacion_gap_sin_assigned.csv').set_index('modelo')
rows=[];checks=[];cv_predictions={}
for (model,assigned,scenario),q in pred.groupby(['modelo','assigned','escenario']):
 ix=q.fila_train.to_numpy();assert len(np.unique(ix))==len(q);assert np.array_equal(q.y_true,y[ix]);assert np.array_equal(q.y_pred,q.score.ge(q.umbral).astype(int));value=ba(q.y_true,q.y_pred)
 reference=metrics[metrics.modelo.eq(model)&metrics.assigned.eq(assigned)&metrics.escenario.eq(scenario)].iloc[0];assert abs(value-reference.balanced_accuracy)<1e-12
 checks.append({'modelo':model,'assigned':bool(assigned),'escenario':scenario,'n':len(q),'ba':value,'verificado':True})
 if scenario!='cv':continue
 q=q.sort_values('fila_train');assert q.fila_train.tolist()==list(range(N));cv_predictions[(model,bool(assigned))]=q
 gap=None;tr=None;global_ba=None
 if assigned:
  r=selection.loc[model];gap=r.gap_original if model=='logistic' else r.gap_base if model=='hist_gradient_boosting' else r.gap_alternativa
 elif model in controls.index:
  r=controls.loc[model];gap=r.gap;tr=r.ba_train;global_ba=r.ba_val
 rows.append({'modelo':model,'assigned':bool(assigned),'ba_externa':value,'std_folds':reference.balanced_accuracy_std_folds_ddof0,'roc_auc':roc_auc_score(q.y_true,q.score),'gap_global_oof':gap,'ba_train_diagnostica':tr,'ba_global_oof_control':global_ba,'protocolo':'5 externos / corte OOF 3 internos'})
pd.DataFrame(checks).to_csv(OUT/'verificacion_predicciones.csv',index=False)
# RF sin assigned: reconstruir exclusivamente con caché ya calculado del experimento de ensembles.
run=ROOT/'experiments/seleccion/stacking_resultados/a92fbb4e4778849e';manifest=json.loads((run/'manifiesto.json').read_text());assert manifest['train_hash']==sha(ROOT/'data/raw/train.csv');assert digest(manifest)[:16]==run.name
fixed=json.loads((ROOT/'experiments/seleccion/parametros.json').read_text());assert manifest['params']['random_forest']==fixed['random_forest']
def cached(kind,tr,va=None):
 key={'identity':run.name,'kind':kind,'model':'random_forest','train':idxhash(tr),'evaluate':None if va is None else idxhash(va)}
 p=run/'cache'/f'{kind}_random_forest_{digest(key)[:24]}'
 info=json.loads(p.with_suffix('.json').read_text());assert info['key']==key;assert info['sha256_npz']==sha(p.with_suffix('.npz'))
 with np.load(p.with_suffix('.npz')) as z:return {k:z[k] for k in z.files}
rfscore=np.full(N,np.nan);rfpred=np.zeros(N,int);rf_folds=[]
for k,(tr,va) in enumerate(folds):
 a=cached('base',tr,va);b=cached('oof',tr);assert np.array_equal(a['eval_idx'],va);assert np.array_equal(b['train_idx'],tr);assert len(b['score'])==len(tr);assert not set(groups[tr])&set(groups[va]);threshold,innerba=best_threshold(y[tr],b['score']);rfscore[va]=a['score'];rfpred[va]=a['score']>=threshold;rf_folds.append({'fold':k,'threshold':threshold,'ba':ba(y[va],rfpred[va]),'n':len(va)})
assert np.isfinite(rfscore).all();rfglobal=ba(y,rfpred);rows.append({'modelo':'random_forest','assigned':False,'ba_externa':rfglobal,'std_folds':float(np.std([r['ba'] for r in rf_folds])),'roc_auc':roc_auc_score(y,rfscore),'gap_global_oof':None,'ba_train_diagnostica':None,'ba_global_oof_control':None,'protocolo':'5 externos / corte OOF 3 internos; caché ensembles'})
pd.DataFrame(rf_folds).to_csv(OUT/'rf_sin_assigned_folds.csv',index=False);pd.DataFrame({'fila_train':np.arange(N),'y_true':y,'score':rfscore,'y_pred':rfpred}).to_csv(OUT/'rf_sin_assigned_predicciones.csv',index=False)
# Ensembles auditados, solo CV.
e=pd.read_csv(run/'resultados/predicciones_externas.csv')
em=pd.read_csv(run/'resultados/metricas_externas_globales.csv')
for model in ['promedio','stacking']:
 q=e[e.modelo.eq(model)&e.escenario.eq('cv')].sort_values('fila_train');assert q.fila_train.tolist()==list(range(N));assert np.array_equal(q.y_true,y);assert np.array_equal(q.y_pred,q.score.ge(q.umbral).astype(int));m=em[em.modelo.eq(model)&em.escenario.eq('cv')].iloc[0];assert abs(ba(y,q.y_pred)-m.balanced_accuracy)<1e-12
 rows.append({'modelo':model,'assigned':False,'ba_externa':ba(y,q.y_pred),'std_folds':m.balanced_accuracy_std_folds_ddof0,'roc_auc':roc_auc_score(y,q.score),'gap_global_oof':None,'ba_train_diagnostica':None,'ba_global_oof_control':None,'protocolo':'umbrales internos; stacking con aislamiento adicional'})
# Comparación condicional pareada entre RF sin assigned e HistGB.
hgb=cv_predictions[('hist_gradient_boosting',False)];hp=hgb.y_pred.to_numpy();gframe=pd.DataFrame({'g':groups,'n0':y==0,'n1':y==1,'d0':((hp==0)&(y==0)).astype(int)-((rfpred==0)&(y==0)).astype(int),'d1':((hp==1)&(y==1)).astype(int)-((rfpred==1)&(y==1)).astype(int)}).groupby('g',sort=False).sum().to_numpy(dtype=float)
rng=np.random.default_rng(42);boot=[]
for _ in range(2000):
 sums=gframe[rng.integers(0,len(gframe),len(gframe))].sum(axis=0)
 if sums[0]>0 and sums[1]>0:boot.append(.5*(sums[2]/sums[0]+sums[3]/sums[1]))
rf_comp={'diferencia_hgb_menos_rf':ba(y,hp)-rfglobal,'ic_condicional025':float(np.quantile(boot,.025)),'ic_condicional975':float(np.quantile(boot,.975)),'folds_ganados_hgb':sum(ba(y[va],hp[va])>ba(y[va],rfpred[va]) for _,va in folds),'n_replicas':len(boot)}
# Primeras configuraciones: las variantes sin assigned de TODOS los modelos existen como OOF.
original=[];search=[];pareto=[]
for folder in sorted((ROOT/'experiments').iterdir()):
 if not (folder/'metricas.json').exists():continue
 m=json.loads((folder/'metricas.json').read_text());scores=np.load(folder/'oof_sin_assigned.npy');assert scores.shape==(N,);t,v=best_threshold(y,scores);assert abs(v-m['ba_val_sin_assigned'])<1e-12
 original.append({'modelo':folder.name,'ba_original_con_assigned':m['ba_val'],'gap_original_con_assigned':m['gap'],'ba_original_sin_assigned_oof':v,'auc_original_sin_assigned':roc_auc_score(y,scores),'umbral_sin_assigned':t,'n_busqueda':m['combinaciones_probadas'],'params':(folder/'mejores_parametros.json').read_text()})
 a=pd.concat([pd.read_csv(folder/'busqueda.csv'),pd.read_csv(folder/'refinamiento.csv')],ignore_index=True).drop_duplicates('params');assert len(a)==m['combinaciones_probadas'];a['modelo']=folder.name;search.append(a)
 for cap in [.02,.03,.04,.05]:
  q=a[a.gap.le(cap)]
  if len(q):
   r=q.loc[q.ba_val.idxmax()];pareto.append({'modelo':folder.name,'limite_gap':cap,'n_candidatos':len(q),'ba_max_oof':r.ba_val,'gap':r.gap,'roc_auc':r.roc_auc,'params':r.params,'assigned':True,'nota':'máximo retrospectivo de búsqueda; no evaluación independiente'})
search=pd.concat(search,ignore_index=True);assert len(search)==669
pd.DataFrame(original).to_csv(OUT/'configuraciones_originales.csv',index=False);pd.DataFrame(pareto).to_csv(OUT/'frontera_ba_gap_busqueda.csv',index=False)
# Alcance preciso del agrupamiento: comparar filas sobre las entradas activas crudas.
active=[c for c in FEATURE_COLS if c not in ['assigned_gpu_type','execution_week']];activegroups=df.groupby(active,dropna=False,sort=False).ngroup().to_numpy();overlaps=[]
for k,(tr,va) in enumerate(folds):
 n=int(np.isin(activegroups[va],np.unique(activegroups[tr])).sum());overlaps.append({'fold':k,'n_validacion':len(va),'coincidencias_entradas_activas_crudas':n})
pd.DataFrame(overlaps).to_csv(OUT/'alcance_agrupamiento.csv',index=False)
summary=pd.DataFrame(rows).sort_values(['assigned','ba_externa'],ascending=[True,False]);summary.to_csv(OUT/'comparacion_protocolos.csv',index=False)
result={'familias_originales':len(original),'configuraciones_buscadas':len(search),'conjuntos_predicciones_verificados':len(checks),'rf_sin_assigned':{'ba':rfglobal,'auc':roc_auc_score(y,rfscore),'std':float(np.std([r['ba'] for r in rf_folds]))},'rf_comparacion':rf_comp,'alcance_grupos':overlaps,'sin_fit_ni_test':True}
(OUT/'resumen.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(summary.to_string(index=False));print(json.dumps(result,ensure_ascii=False))
