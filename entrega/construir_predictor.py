"""Package the already evaluated train-only model without retraining or opening test.csv."""
import ast
import base64
import hashlib
import json
import pickle
import zlib
from pathlib import Path
from importlib.metadata import version
import joblib
from compe import config as cfg

def main():
    ROOT = Path(__file__).resolve().parents[1]
    source = ROOT / 'experiments/evaluacion_final_histgb/predictor.joblib'
    bundle = joblib.load(source)
    pipe = bundle['pipeline']
    thr = float(bundle['threshold'])
    assert not pipe.named_steps['prep'].use_assigned
    assert pipe.named_steps['model'].max_iter == 100
    src = (ROOT / 'src/compe/features.py').read_text()
    tree = ast.parse(src)
    klass = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Preprocessor'))
    for method in klass.body:
        if isinstance(method, ast.FunctionDef) and method.name == 'fit':
            method.body = ast.parse("raise RuntimeError('Este predictor solo permite inferencia.')").body
    klass = ast.unparse(klass)
    helper = ast.unparse(next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_as_label')))
    required = [c for c in cfg.FEATURE_COLS if c not in ['assigned_gpu_type', 'execution_week']]
    versions = {p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl']}
    payload = zlib.compress(pickle.dumps(bundle, protocol=5), 9)
    header = '"""HistGB Grupo 9 autocontenido; no entrena ni necesita archivos externos.\nFunción: predictor_v2_Grupo_9(X) -> ndarray 0/1, en orden de entrada.\nArtefacto evaluado: entrenado SOLO con train (38.274 filas), no train+test.\nDependencias verificadas al empaquetar: VERSIONS.\nUsar esas versiones para cargar el modelo serializado; Colab pendiente de prueba.\n"""\n# Ejemplo de uso (con las dependencias indicadas instaladas previamente):\n# import pandas as pd\n# from predictor_v2_Grupo_9 import predictor_v2_Grupo_9\n#\n# X = pd.read_csv("reservas_activas.csv")  # Columnas originales requeridas.\n# y_pred = predictor_v2_Grupo_9(X)\n# assert len(y_pred) == len(X)\n# # y_pred[i]: 1 = cancelación, 0 = no cancelación, en el orden de X.\n# # El modelo ya está entrenado; esta llamada solo realiza inferencia.\n\nimport base64\nimport hashlib\nimport io\nimport pickle\nimport zlib\nfrom functools import lru_cache\nfrom types import SimpleNamespace\nimport numpy as np\nimport pandas as pd\nfrom sklearn.base import BaseEstimator, TransformerMixin\n'.replace('VERSIONS', str(versions))
    installer = f"\nimport sys as _sys\nif __name__ == '__main__' and '--instalar-dependencias' in _sys.argv:\n    import subprocess as _subprocess\n    _subprocess.check_call([_sys.executable, '-m', 'pip', 'install',\n                           *{[f'{k}=={v}' for k, v in versions.items()]!r}])\n    print('Instalación completada. En Colab, reiniciá la sesión antes de importar el predictor.')\n    raise SystemExit(0)\n"
    header = header.replace('import base64\n', installer + '\nimport base64\n', 1)
    text = header + f'\nDEPENDENCIAS = {versions!r}\nUMBRAL = {thr!r}\nCOLUMNAS_REQUERIDAS = {required!r}\n'
    text += f"cfg = SimpleNamespace(TARGET='canceled_job', LEAK_COLS={cfg.LEAK_COLS!r}, CAT_COLS={cfg.CAT_COLS!r}, ID_COLS={cfg.ID_COLS!r})\n"
    text += "DROP_ALWAYS = [cfg.TARGET, *cfg.LEAK_COLS, 'execution_week']\nOTROS = 'otros'\nNULL_LABELS = {'orchestration_agent_id': 'sin_agente', 'organization_id': 'sin_organizacion', 'client_region': 'region_desconocida'}\nNULL_DEFAULT = 'desconocido'\n" + helper + '\n\n' + klass + '\n\nclass _Unpickler(pickle.Unpickler):\n    def find_class(self, module, name):\n        if module == \'compe.features\' and name == \'Preprocessor\':\n            return Preprocessor\n        return super().find_class(module, name)\n\n@lru_cache(maxsize=1)\ndef _modelo():\n    packed = base64.b64decode(_MODEL_DATA)\n    if hashlib.sha256(packed).hexdigest() != _PAYLOAD_SHA256:\n        raise RuntimeError(\'El modelo incluido está incompleto o modificado.\')\n    return _Unpickler(io.BytesIO(zlib.decompress(packed))).load()[\'pipeline\']\n\ndef _entrada(X):\n    if not isinstance(X, pd.DataFrame):\n        raise TypeError(\'X debe ser un pandas.DataFrame con columnas nombradas.\')\n    if X.columns.has_duplicates:\n        raise ValueError(\'X contiene nombres de columnas duplicados.\')\n    missing = [c for c in COLUMNAS_REQUERIDAS if c not in X.columns]\n    if missing:\n        raise ValueError(\'Faltan entradas utilizadas por el modelo: \' + \', \'.join(missing))\n    # Exact order of original training columns; extra/target/leak columns ignored.\n    out = X.loc[:, COLUMNAS_REQUERIDAS].copy()\n    # Historical preprocessing drops this column even though it is unused.\n    out[\'assigned_gpu_type\'] = \'(no_utilizado)\'\n    return out\n\ndef probabilidades_v2_Grupo_9(X):\n    """Probabilidad de canceled_job=1 por fila; no ajusta umbrales."""\n    frame = _entrada(X)\n    if len(frame) == 0:\n        return np.empty(0, dtype=float)\n    model = _modelo()\n    position = int(np.flatnonzero(model.classes_ == 1)[0])\n    scores = np.asarray(model.predict_proba(frame)[:, position], dtype=float)\n    if scores.shape != (len(frame),) or not np.isfinite(scores).all():\n        raise RuntimeError(\'Salida no válida del modelo.\')\n    return scores\n\ndef predictor_v2_Grupo_9(X):\n    """1=cancelación, 0=no cancelación; conserva el orden; no modifica X."""\n    return (probabilidades_v2_Grupo_9(X) >= UMBRAL).astype(np.int8)\n\n# Alias para evaluadores que requieren exactamente predictor(X).\npredictor = predictor_v2_Grupo_9\n'
    text += '\n\ndef predictor_v1_Grupo_9(X):\n    """Nombre solicitado por el enunciado; mismo modelo, corte y salida que v2."""\n    return predictor_v2_Grupo_9(X)\n'
    text += f'\n_PAYLOAD_SHA256 = {hashlib.sha256(payload).hexdigest()!r}\n_MODEL_DATA = b"""\n'
    b64 = base64.b64encode(payload).decode()
    text += '\n'.join((b64[i:i + 120] for i in range(0, len(b64), 120))) + '\n"""\n'
    # Normalizar encabezado y comentarios del archivo generado.
    replacements = {'HistGB Grupo 9 autocontenido; no entrena ni necesita archivos externos.': 'Predictor de cancelaciones GPU · Grupo 9.', 'Artefacto evaluado: entrenado SOLO con train (38.274 filas), no train+test.': 'Entrenado con 38.274 filas de train; umbral fijo.', 'Usar esas versiones para cargar el modelo serializado; Colab pendiente de prueba.': 'Carga entre versiones de sklearn no soportada; usar las dependencias indicadas.', '# Ejemplo de uso (con las dependencias indicadas instaladas previamente):': '# Ejemplo de uso:', '# X = pd.read_csv("reservas_activas.csv")  # Columnas originales requeridas.': '# X = pd.read_csv("reservas_activas.csv")', '# # y_pred[i]: 1 = cancelación, 0 = no cancelación, en el orden de X.': '# # 1 = cancelación; 0 = no cancelación. Se conserva el orden.', '# # El modelo ya está entrenado; esta llamada solo realiza inferencia.\n': '', '# Exact order of original training columns; extra/target/leak columns ignored.': '# Seleccionar entradas en el orden usado al entrenar.', '# Historical preprocessing drops this column even though it is unused.': '# La transformación original exige esta columna para descartarla.', '# Alias para evaluadores que requieren exactamente predictor(X).\n': '', '"""Probabilidad de canceled_job=1 por fila; no ajusta umbrales."""': '"""Probabilidad de cancelación por fila."""'}
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    text = text.replace('\"\"\"\n# Ejemplo de uso', '\"\"\"\n\n# Ejemplo de uso', 1)
    text = text.replace('Se conserva el orden.\n\n\nimport sys', 'Se conserva el orden.\n\nimport sys', 1)
    path = ROOT / 'entrega/predictor_v2_Grupo_9.py'
    path.write_text(text)
    manifest = dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), threshold=thr, training_rows=38274, training_scope='train_only', dependencies=versions, required_columns=required, entry_points=['predictor_v1_Grupo_9', 'predictor_v2_Grupo_9', 'predictor'], retrained=False, colab_verified=False)
    (ROOT / 'entrega/manifiesto.json').write_text(json.dumps(manifest, indent=2))
    print(path, 'bytes', path.stat().st_size)
if __name__ == '__main__':
    main()
