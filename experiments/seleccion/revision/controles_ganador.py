import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from compe.data import load_train
from compe.config import FEATURE_COLS,CAT_COLS,ID_COLS
from compe.validation import make_folds,cv_scores,metrics
from compe.models import MODELS
from compe.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score as ba
out=Path('experiments/seleccion/revision');df=load_train();y=df.canceled_job.to_numpy();spec=MODELS['hist_gradient_boosting'];rows=[]
for grouped in [True,False]:
 f=make_folds(df,grouped=grouped);res=cv_scores(make_pipeline(spec.build({}),spec.variante,use_assigned=False),df,f,spec.score);m=metrics(df,res,f);rows.append(dict(grouped=grouped,**m))
pd.DataFrame(rows).to_csv(out/'control_hgb_sin_assigned.csv',index=False)
# Best single-variable stump picked only on outer training BA, no assigned.
pred=np.zeros(len(df),int);refs=[]
for k,(tr,va) in enumerate(make_folds(df)):
 best=(-1,None,None)
 for col in FEATURE_COLS:
  if col=='assigned_gpu_type':continue
  x=df[[col]]
  if col in CAT_COLS+ID_COLS:
   enc=OneHotEncoder(handle_unknown='ignore');xt=enc.fit_transform(x.iloc[tr].astype('string').fillna('(nulo)'));xv=enc.transform(x.iloc[va].astype('string').fillna('(nulo)'))
  else:
   med=x.iloc[tr].median();xt=x.iloc[tr].fillna(med);xv=x.iloc[va].fillna(med)
  model=DecisionTreeClassifier(max_depth=1,class_weight='balanced',random_state=42).fit(xt,y[tr]);b=ba(y[tr],model.predict(xt))
  if b>best[0]:best=b,col,model.predict(xv)
 pred[va]=best[2];refs.append(dict(fold=k,variable=best[1],ba=ba(y[va],best[2])))
refs.append(dict(fold='global',variable='seleccionada_en_train',ba=ba(y,pred)));pd.DataFrame(refs).to_csv(out/'referencia_univariable.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False));print(refs)
