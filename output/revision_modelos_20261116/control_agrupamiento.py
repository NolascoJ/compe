"""Control acotado en TRAIN: grupos de entradas efectivas, 3 candidatos fijos.
No abre TEST ni modifica código o experimentos previos. No es selección HP anidada completa.
"""
from pathlib import Path
import sys,os,json,hashlib,time,warnings,platform
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd,sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from joblib import Parallel,delayed,parallel_config
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'control_activo';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
from compe.config import FEATURE_COLS
from compe.models import MODELS
from compe.pipeline import make_pipeline
from compe.validation import best_threshold

ACTIVE=[c for c in FEATURE_COLS if c not in ['assigned_gpu_type','execution_week']]
DF=pd.read_csv(ROOT/'data/raw/train.csv');Y=DF.canceled_job.to_numpy();G=DF.groupby(ACTIVE,dropna=False,sort=False).ngroup().to_numpy()
FOLDS=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(DF,Y,G))
PARS=json.loads((ROOT/'experiments/seleccion/parametros.json').read_text())
CANDIDATES=[('HistGB_31','hist_gradient_boosting',{}),('HistGB_15','hist_gradient_boosting',{'max_leaf_nodes':15}),('GradientBoosting','gradient_boosting',PARS['gradient_boosting'])]
identity={'train_sha256':hashlib.sha256((ROOT/'data/raw/train.csv').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':42,'n_externos':5,'n_internos':3,'active_columns':ACTIVE,'candidates':CANDIDATES,'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__}}
(OUT/'manifest.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2))

def fit_candidate(name,model,params,k):
 tr,va=FOLDS[k];assert not set(G[tr])&set(G[va]);stamp=time.time();inner=np.full(len(tr),np.nan);notice=[]
 with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always')
  for itr,iva in StratifiedGroupKFold(3,shuffle=True,random_state=42).split(DF.iloc[tr],Y[tr],G[tr]):
   assert not set(G[tr[itr]])&set(G[tr[iva]])
   pipe=make_pipeline(MODELS[model].build(params),MODELS[model].variante,use_assigned=False);pipe.fit(DF.iloc[tr[itr]],Y[tr[itr]]);inner[iva]=pipe.predict_proba(DF.iloc[tr[iva]])[:,1]
  threshold,innerba=best_threshold(Y[tr],inner)
  pipe=make_pipeline(MODELS[model].build(params),MODELS[model].variante,use_assigned=False);pipe.fit(DF.iloc[tr],Y[tr]);score=pipe.predict_proba(DF.iloc[va])[:,1];score_train=pipe.predict_proba(DF.iloc[tr])[:,1];notice=[str(w.message) for w in ws]
 result={'candidate':name,'model':model,'fold':k,'n_train':len(tr),'n_val':len(va),'threshold_internal':threshold,'ba_external':float(balanced_accuracy_score(Y[va],score>=threshold)),'roc_auc':float(roc_auc_score(Y[va],score)),'seconds':time.time()-stamp,'warnings':notice}
 np.savez_compressed(OUT/f'{name}_fold{k}.npz',train_idx=tr,eval_idx=va,score_external=score,score_train=score_train,score_inner=inner)
 (OUT/f'{name}_fold{k}.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
 tasks=[(name,model,params,k) for name,model,params in CANDIDATES for k in range(5)]
 print(f'CONTROL: {len(tasks)} folds externos, 60 ajustes; solo TRAIN; grupos={len(np.unique(G))}',flush=True)
 with parallel_config(backend='loky',inner_max_num_threads=1):
  for r in Parallel(n_jobs=2,return_as='generator_unordered')(delayed(fit_candidate)(*t) for t in tasks): print(json.dumps(r,ensure_ascii=False),flush=True)
 summaries=[];predictions=[]
 for name,model,params in CANDIDATES:
  score=np.full(len(Y),np.nan);pred=np.zeros(len(Y),int);foldresults=[];blocks=[]
  for k in range(5):
   info=json.loads((OUT/f'{name}_fold{k}.json').read_text());z=np.load(OUT/f'{name}_fold{k}.npz');va=z['eval_idx'];score[va]=z['score_external'];pred[va]=z['score_external']>=info['threshold_internal'];foldresults.append(info);blocks.append({k:z[k] for k in z.files})
  threshold,ba_oof=best_threshold(Y,score);ba_train=np.mean([balanced_accuracy_score(Y[z['train_idx']],z['score_train']>=threshold) for z in blocks]);summaries.append({'candidato':name,'modelo':model,'ba_externa':balanced_accuracy_score(Y,pred),'ba_externa_media':np.mean([r['ba_external'] for r in foldresults]),'std_folds':np.std([r['ba_external'] for r in foldresults]),'roc_auc':roc_auc_score(Y,score),'ba_train_global':ba_train,'ba_oof_global':ba_oof,'gap_global_oof':ba_train-ba_oof,'umbral_global_oof':threshold,'params':json.dumps(params,sort_keys=True),'warnings':sum(len(r['warnings']) for r in foldresults)})
  predictions.append(pd.DataFrame({'candidato':name,'fila_train':np.arange(len(Y)),'y_true':Y,'score':score,'pred_internal':pred}))
 pd.DataFrame(summaries).to_csv(OUT/'resumen.csv',index=False);pd.concat(predictions).to_csv(OUT/'predicciones.csv',index=False);print('COMPLETO\n'+pd.DataFrame(summaries).to_string(index=False),flush=True)
