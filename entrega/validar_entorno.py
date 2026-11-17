"""Prueba autónoma de versiones e inferencia sobre el ejemplo, sin acceder a train/test.

Se puede copiar con el predictor y los dos CSV a una carpeta fuera del repo.
No instala dependencias ni modifica el modelo.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
import warnings


def validate(directory: Path):
    predictor = directory / "predictor_v2_Grupo_9.py"
    tree = ast.parse(predictor.read_text())
    dependencies = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "DEPENDENCIAS" for t in n.targets))
    installed = {}
    for package in dependencies:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = None
    report = dict(python=platform.python_version(), platform=platform.platform(),
                  runtime="google_colab" if (importlib.util.find_spec("google") is not None
                          and importlib.util.find_spec("google.colab") is not None) else "local",
                  dependencies=installed, expected_dependencies=dependencies,
                  predictor_sha256=hashlib.sha256(predictor.read_bytes()).hexdigest(),
                  test_csv_accessed=False, retrained=False)
    if installed != dependencies:
        return {**report, "status": "ERROR", "error": "Versiones diferentes a las del artefacto; instalar requirements.txt y reiniciar el kernel."}

    import numpy as np
    import pandas as pd
    from sklearn.exceptions import InconsistentVersionWarning

    spec = importlib.util.spec_from_file_location("predictor_prueba", predictor)
    module = importlib.util.module_from_spec(spec)
    data = pd.read_csv(directory / "dataset_prueba_original_300.csv")
    expected = pd.read_csv(directory / "predicciones_esperadas_prueba_300.csv")
    original = data.copy(deep=True)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        warnings.simplefilter("error", InconsistentVersionWarning)
        spec.loader.exec_module(module)
        scores = module.probabilidades_v2_Grupo_9(data)
        labels = module.predictor_v2_Grupo_9(data)
        np.testing.assert_array_equal(labels, expected.prediccion.to_numpy())
        np.testing.assert_allclose(scores, expected.probabilidad_cancelacion.to_numpy(), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(module.predictor_v1_Grupo_9(data), labels)
        np.testing.assert_array_equal(module.predictor(data), labels)
        np.testing.assert_array_equal(module.predictor(data.iloc[::-1]), labels[::-1])
        pd.testing.assert_frame_equal(data, original)
        assert labels.shape == (len(data),) and np.isin(labels, [0, 1]).all()
        assert module.predictor(data.iloc[:0]).shape == (0,)
        optional = [c for c in ["assigned_gpu_type", "execution_week", "canceled_job",
                               "execution_status", "execution_status_date"] if c in data]
        np.testing.assert_array_equal(module.predictor(data.drop(columns=optional)), labels)
        for c in ["high_value_client", "total_processes", "total_gpu_hours"]:
            try:
                module.predictor(data.drop(columns=[c]))
            except ValueError as exc:
                assert c in str(exc)
            else:
                raise AssertionError(f"No rechazó la entrada sin {c}")
    return {**report, "status": "OK", "rows_compared": len(data), "labels_match": True,
            "max_score_error": float(np.max(np.abs(scores - expected.probabilidad_cancelacion.to_numpy()))),
            "v1_v2_equivalent": True, "order_preserved": True, "input_unchanged": True,
            "optional_columns_supported": True, "missing_active_features_rejected": True,
            "warnings": [str(w.message) for w in observed]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--report", type=Path, help="Guardar resultado JSON en esta ruta.")
    args = parser.parse_args()
    try:
        report = validate(args.directory)
    except Exception as exc:
        report = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
