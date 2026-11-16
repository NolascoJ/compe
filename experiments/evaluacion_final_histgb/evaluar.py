"""PYTHONPATH=src .venv/bin/python experiments/evaluacion_final_histgb/evaluar.py
Una ejecución; las siguientes verifican artefactos y regeneran el informe.
"""
import hashlib
import json
import platform
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from compe.config import ROOT, FEATURE_COLS, TARGET
from compe.models import MODELS
from compe.pipeline import make_pipeline

OUT = Path(__file__).resolve().parent
THRESHOLD = 0.38294747099493937
TRAIN = ROOT / 'data/raw/train.csv'
TEST = ROOT / 'data/raw/test.csv'
CANDIDATE = ROOT / 'experiments/seleccion/revision_stacking/candidato_para_test.json'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, obj):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)

def check(condition, message):
    if not condition:
        raise RuntimeError(message + '; no se repetirá la evaluación automáticamente.')

def metrics(d):
    y, p = d.y_true.to_numpy(), d.prediction.to_numpy()
    tn, fp, fn, tp = [int(((y == a) & (p == b)).sum()) for a,b in [(0,0),(0,1),(1,0),(1,1)]]
    div = lambda a,b: a / b if b else None
    recall, spec = div(tp,tp+fn), div(tn,tn+fp)
    return dict(n=len(d), n0=tn+fp, n1=fn+tp, TN=tn, FP=fp, FN=fn, TP=tp,
                balanced_accuracy=(recall+spec)/2 if recall is not None and spec is not None else None,
                recall=recall, specificity=spec, precision=div(tp,tp+fp),
                f1=div(2*tp,2*tp+fp+fn), accuracy=div(tp+tn,len(d)),
                roc_auc=float(roc_auc_score(y,d.score)) if len(set(y))==2 else None)

def main():
    candidate = json.loads(CANDIDATE.read_text())
    model = make_pipeline(MODELS['hist_gradient_boosting'].build({}), 'E', use_assigned=False)
    params = model.named_steps['model'].get_params()
    check(all(params[k] == v for k,v in candidate['params'].items()), 'Constructor distinto')
    check(candidate['umbral_propuesto'] == THRESHOLD, 'Umbral distinto')
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__)
    code = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'src/compe').glob('*.py'))}
    code[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    identity = dict(parameters=params, threshold=THRESHOLD, versions=versions, code_sha256=code,
                    train_sha256=sha(TRAIN), candidate_sha256=sha(CANDIDATE),
                    preprocessing=model.named_steps['prep'].get_params(), feature_cols=FEATURE_COLS)
    check(identity['train_sha256']==candidate['hash_train'], 'Train distinto al de selección')
    manifest_path = OUT/'manifest.json'
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        check(m['identity']==identity, 'Manifiesto incompatible con código, datos o versiones actuales')
        check(m['state']=='complete', 'Hay una evaluación parcial: '+m['state'])
        for name, digest in m['artifacts'].items():
            check((OUT/name).exists() and sha(OUT/name)==digest, 'Artefacto inconsistente: '+name)
        check(sha(TEST)==m['test_sha256'], 'Test modificado')
    else:
        check(not [p for p in OUT.iterdir() if p.name != 'evaluar.py'], 'Hay archivos parciales sin manifiesto')
        check(not (ROOT/'reports/evaluacion_final_histgb.md').exists(), 'Existe un informe final previo')
        m = dict(identity=identity, state='prepared_before_test', created_utc=datetime.now(timezone.utc).isoformat(),
                 prior_evaluation_check='Revisión de archivos y referencias del repositorio: sin evaluación final previa; candidato registrado como no ejecutado.',
                 command='PYTHONPATH=src .venv/bin/python experiments/evaluacion_final_histgb/evaluar.py')
        save('manifest.json',m)
        train = pd.read_csv(TRAIN)
        model.fit(train[FEATURE_COLS], train[TARGET])
        check(not {'assigned_gpu_type','gpu_changed',TARGET,'execution_status','execution_status_date'} & set(model.named_steps['prep'].transform(train.iloc[:1][FEATURE_COLS]).columns), 'Features prohibidas')
        joblib.dump(dict(pipeline=model, threshold=THRESHOLD), OUT/'predictor.joblib')
        save('threshold.json',dict(threshold=THRESHOLD, comparison='score >= threshold'))
        m.update(state='trained_before_test', train_rows=len(train), predictor_sha256=sha(OUT/'predictor.joblib'))
        save('manifest.json',m)
        # Primera y única apertura para evaluación; ningún fit después de esta línea.
        test = pd.read_csv(TEST)
        check(set(test[TARGET].unique()) <= {0,1}, 'Etiquetas inválidas')
        scores = model.predict_proba(test[FEATURE_COLS])[:,list(model.classes_).index(1)]
        check(np.isfinite(scores).all(), 'Scores inválidos')
        # Igualdad exacta de las 31 entradas originales, nulos iguales (sin hashes aproximados).
        joined = test[FEATURE_COLS].merge(train[FEATURE_COLS].drop_duplicates().assign(_match=True), on=FEATURE_COLS, how='left', sort=False, validate='many_to_one')
        p = pd.DataFrame(dict(test_index=test.index, y_true=test[TARGET], score=scores,
                              prediction=(scores>=THRESHOLD).astype(int), threshold=THRESHOLD,
                              client_region=test.client_region, execution_year=test.execution_year,
                              reservation_commitment=test.reservation_commitment,
                              execution_status=test.execution_status, matches_train=joined['_match'].fillna(False).astype(bool)))
        p.to_csv(OUT/'predictions.csv',index=False)
        m.update(state='complete', test_sha256=sha(TEST), test_rows=len(test),
                 artifacts={name:sha(OUT/name) for name in ['predictor.joblib','threshold.json','predictions.csv']})
        save('manifest.json',m)
    p = pd.read_csv(OUT/'predictions.csv',float_precision='round_trip')
    check(len(p)==m['test_rows'] and np.array_equal(p.test_index,np.arange(len(p))), 'Índices incompletos')
    check((p.threshold==THRESHOLD).all() and np.array_equal(p.prediction,(p.score>=THRESHOLD).astype(int)), 'Predicciones inconsistentes')
    masks = dict(global_=np.ones(len(p),bool), sin_coincidencias=~p.matches_train,
                 PRT=p.client_region.eq('PRT'), resto=~p.client_region.eq('PRT'),
                 sin_locked_allocation=~p.reservation_commitment.eq('locked_allocation'),
                 con_locked_allocation=p.reservation_commitment.eq('locked_allocation'),
                 no_show=p.execution_status.eq('no_show'))
    masks.update({f'anio_{a}':p.execution_year.eq(a) for a in sorted(p.execution_year.unique())})
    result = {k:metrics(p.loc[v]) for k,v in masks.items()}
    save('metrics.json',result)
    columns = list(result['global_'])
    fmt = lambda v: 'indefinida' if v is None else (f'{v:.6f}' if isinstance(v,float) else str(v))
    table = '| Grupo | '+' | '.join(columns)+' |\n|---|'+'---|'*len(columns)+'\n'
    table += '\n'.join('| '+k+' | '+' | '.join(fmt(v[c]) for c in columns)+' |' for k,v in result.items())
    ba=result['global_']['balanced_accuracy']
    report = f'''# Evaluación final de HistGradientBoosting en holdout interno

Modelo fijo sin assigned_gpu_type ni gpu_changed; pipeline original E, ajustado exclusivamente con las {m['train_rows']:,} filas de train. Target, execution_status y execution_status_date excluidos de las entradas. El estado se utiliza exclusivamente para análisis. Parámetros completos, versiones y hashes: [manifest.json](../experiments/evaluacion_final_histgb/manifest.json).

Umbral fijo **{THRESHOLD}**; cancelación si score >= umbral. Una única llamada a predict_proba sobre test; sin selección ni ajustes posteriores. Índice test_index: posición original en test.csv, base cero, sin reordenamiento. Predictor y umbral guardados antes de abrir test.

**Balanced accuracy global: {ba:.4%}.**

{table}

Las cantidades n0 y n1 corresponden a completados y cancelados. BA y ROC-AUC son indefinidas si falta alguna clase; denominadores nulos también quedan indefinidos. no_show cuenta como cancelación; su recall es la proporción detectada por este mismo predictor.

Solapamiento conocido: previamente se encontraron 281/9.568 filas (2,94%) con entradas idénticas a train. Reproducción actual: **{int(p.matches_train.sum())}/{len(p)} ({p.matches_train.mean():.4%})**, usando exclusivamente las 31 FEATURE_COLS originales (incluye assigned_gpu_type para esta definición), nulos iguales. No se usa la coincidencia como predictor ni se excluye del resultado global. El subconjunto sin coincidencias se informa adicionalmente; tampoco garantiza independencia de clientes ni generalización temporal.

La CV previa obtuvo aproximadamente 84,829% de BA externa con cinco umbrales internos diferentes, entrenamientos parciales y folds agrupados sin entradas idénticas entre entrenamiento y validación. **No corresponde atribuir esa cifra al umbral global fijo de esta evaluación.** La diferencia descriptiva del holdout es {(ba-0.84829)*100:+.3f} puntos porcentuales. El modelo actual usa todo train y otra población, con el solapamiento declarado. La BA OOF global de 84,84% del control de gap utilizó un umbral elegido sobre esos mismos OOF y es levemente optimista. Estas diferencias no constituyen una prueba estadística ni justifican cambiar el predictor. No se probaron otros candidatos ni umbrales en test. Este holdout interno no es el test oculto de los profesores.

Reproducir/verificar: `PYTHONPATH=src .venv/bin/python experiments/evaluacion_final_histgb/evaluar.py`. Si existen resultados completos, verifica hashes y consistencia y regenera métricas e informe desde predictions.csv, sin entrenar ni predecir. Si hay archivos parciales o inconsistentes, aborta. Los experimentos anteriores permanecen intactos.
'''
    (ROOT/'reports/evaluacion_final_histgb.md').write_text(report)
    print(json.dumps(result['global_'],indent=2))

if __name__=='__main__':
    main()
