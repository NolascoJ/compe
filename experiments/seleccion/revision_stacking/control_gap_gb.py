import json
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from compe.data import load_train
from compe.models import MODELS
from compe.pipeline import make_pipeline
from compe.validation import make_folds,cv_scores,metrics
out=Path('experiments/seleccion/revision_stacking');df=load_train();folds=make_folds(df)
params=json.loads(Path('experiments/seleccion/parametros.json').read_text())['gradient_boosting'];spec=MODELS['gradient_boosting']
with threadpool_limits(limits=2):
 res=cv_scores(make_pipeline(spec.build(params),spec.variante,use_assigned=False),df,folds,spec.score,n_jobs=2)
m=metrics(df,res,folds)
pred=pd.read_csv('experiments/seleccion/calibracion/resultados/predicciones_externas.csv');old=pred[(pred.modelo=='gradient_boosting')&(~pred.assigned)&(pred.escenario=='cv')].sort_values('fila_train')
assert np.allclose(res.oof,old.score,rtol=0,atol=1e-12)
(out/'control_gap_gb_sin_assigned.json').write_text(json.dumps(dict(params=params,use_assigned=False,**m),indent=2))
np.savez_compressed(out/'control_gap_gb_sin_assigned.npz',oof=res.oof,**{f'train_{i}':s for i,s in enumerate(res.train)})
h=pd.read_csv('experiments/seleccion/revision/control_hgb_sin_assigned.csv');h=h[h.grouped].iloc[0]
t=pd.DataFrame([dict(modelo='gradient_boosting',**m),dict(modelo='hist_gradient_boosting',**h.to_dict())]);t.to_csv(out/'comparacion_gap_sin_assigned.csv',index=False)
print(t[['modelo','ba_train','ba_val','gap','ba_std']].to_string(index=False),flush=True)
