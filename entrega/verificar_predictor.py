"""Verify packaging on train only. No holdout evaluation or retraining."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import joblib, numpy as np, pandas as pd

def main():
    ROOT = Path(__file__).resolve().parents[1]
    path = ROOT / 'entrega/predictor_v2_Grupo_9.py'
    s = importlib.util.spec_from_file_location('packed', path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    x = pd.read_csv(ROOT / 'data/raw/train.csv').sample(600, random_state=42)
    before = x.copy(deep=True)
    b = joblib.load(ROOT / 'experiments/evaluacion_final_histgb/predictor.joblib')
    p = b['pipeline'].predict_proba(x)[:, 1]
    q = m.probabilidades_v2_Grupo_9(x)
    np.testing.assert_array_equal(p, q)
    np.testing.assert_array_equal(m.predictor(x), p >= b['threshold'])
    np.testing.assert_array_equal(m.predictor_v1_Grupo_9(x), m.predictor_v2_Grupo_9(x))
    pd.testing.assert_frame_equal(x, before)
    xx = x.drop(columns=['assigned_gpu_type', 'canceled_job', 'execution_status', 'execution_status_date'])
    np.testing.assert_array_equal(m.probabilidades_v2_Grupo_9(xx), p)
    np.testing.assert_array_equal(m.probabilidades_v2_Grupo_9(x.iloc[::-1]), p[::-1])
    np.testing.assert_array_equal(m.probabilidades_v2_Grupo_9(x[x.columns[::-1]].assign(extra_ignored=123)), p)
    assert m.predictor(x.iloc[:0]).shape == (0,)
    new = xx.head(8).copy()
    new['client_region'] = 'REGION_NUEVA'
    assert np.isfinite(m.probabilidades_v2_Grupo_9(new)).all()
    np.testing.assert_allclose(m.probabilidades_v2_Grupo_9(xx.iloc[:8]), np.concatenate([m.probabilidades_v2_Grupo_9(xx.iloc[i:i + 1]) for i in range(8)]), rtol=0, atol=1e-15)
    try:
        m.predictor(xx.drop(columns=['queue_wait_time']))
    except ValueError:
        pass
    else:
        raise AssertionError('missing active input accepted')
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(path, Path(tmp) / path.name)
        x.head(8).to_pickle(Path(tmp) / 'input.pkl')
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        code = "import sys;sys.path.insert(0,'.');import pandas as pd;from predictor_v2_Grupo_9 import predictor;print(predictor(pd.read_pickle('input.pkl')).tolist());assert not any(k=='compe' or k.startswith('compe.') for k in sys.modules)"
        subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp, env=env, check=True)
    report = dict(status='OK', rows_compared=600, exact_scores=True, v1_v2_equivalent=True, standalone_isolated_process=True, order_preserved=True, input_unchanged=True, assigned_optional=True, unknown_region_supported=True, missing_active_features_rejected=True, colab_verified=False, test_csv_accessed=False)
    (ROOT / 'entrega/verificacion.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
