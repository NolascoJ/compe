"""Matriz de confusion y metricas calculadas SOLO desde `predicciones_externas.csv`.

No carga train, no carga ni ajusta modelos y no vuelve a optimizar ningun punto de corte:
usa el `y_pred` guardado (cada fold con su propio punto de corte interno).

Uso:
  .venv/bin/python experiments/seleccion/metricas_externas.py [predicciones_externas.csv] [carpeta_salida]

Convenciones:
  - clase positiva: canceled_job = 1.
  - global = concatenando las predicciones externas del grupo (no se promedian matrices ni metricas).
  - por fold: misma cuenta dentro de cada fold externo; media y desvio con ddof=0 (igual que validation.py).
  - si un denominador es cero (por ejemplo, ningun positivo predicho para la precision), la metrica queda NA;
    la balanced accuracy queda NA si falta alguna de las dos clases reales.
  - ROC-AUC global mezcla puntuaciones de modelos distintos (uno por fold); por eso se informa tambien por fold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CLAVES = ["modelo", "assigned", "escenario"]
OBLIGATORIAS = ["modelo", "assigned", "escenario", "fold_externo", "fila_train", "y_true", "score", "y_pred", "umbral"]
METRICAS = ["sensibilidad", "especificidad", "balanced_accuracy", "accuracy", "precision", "f1", "roc_auc"]
POR_DEFECTO = Path(__file__).resolve().parent / "calibracion" / "resultados" / "predicciones_externas.csv"


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else float("nan")


def metricas(y_true, y_pred, score) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    sens, esp = _div(tp, tp + fn), _div(tn, tn + fp)
    dos_clases = (y == 1).any() and (y == 0).any()
    return {
        "n": len(y), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "sensibilidad": sens, "especificidad": esp,
        "balanced_accuracy": (sens + esp) / 2 if dos_clases else float("nan"),
        "accuracy": _div(tp + tn, len(y)), "precision": _div(tp, tp + fp), "f1": _div(2 * tp, 2 * tp + fp + fn),
        "roc_auc": float(roc_auc_score(y, score)) if dos_clases else float("nan"),
    }


def calcular(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    faltan = [c for c in OBLIGATORIAS if c not in pred.columns]
    if faltan:
        raise ValueError(f"faltan columnas en las predicciones: {faltan}")
    if not pred["y_pred"].isin([0, 1]).all() or not pred["y_true"].isin([0, 1]).all():
        raise ValueError("y_pred e y_true tienen que ser 0/1")
    if not ((pred["score"] >= pred["umbral"]).astype(int) == pred["y_pred"]).all():
        raise ValueError("y_pred no coincide con score >= umbral")
    matriz, globales, por_fold = [], [], []
    for clave, g in pred.groupby(CLAVES, sort=False):
        ident = dict(zip(CLAVES, clave))
        if not g["fila_train"].is_unique:
            raise ValueError(f"{ident}: una observacion aparece mas de una vez")
        m = metricas(g["y_true"], g["y_pred"], g["score"])
        if m["TN"] + m["FP"] + m["FN"] + m["TP"] != len(g):
            raise AssertionError(f"{ident}: la matriz no suma la cantidad de observaciones")
        matriz.append({**ident, **{k: m[k] for k in ["n", "TN", "FP", "FN", "TP"]}})
        filas_fold = []
        for f, gf in g.groupby("fold_externo", dropna=False, sort=True):
            mf = metricas(gf["y_true"], gf["y_pred"], gf["score"])
            filas_fold.append({**ident, "fold_externo": f, "umbral": float(gf["umbral"].iloc[0]), **mf})
        por_fold += filas_fold
        tf = pd.DataFrame(filas_fold)
        resumen = {**ident, **m, "n_folds": len(tf)}
        for k in METRICAS:
            resumen[f"{k}_media_folds"] = float(tf[k].mean())
            resumen[f"{k}_std_folds_ddof0"] = float(tf[k].std(ddof=0))
        globales.append(resumen)
    return pd.DataFrame(matriz), pd.DataFrame(globales), pd.DataFrame(por_fold)


def escribir(pred: pd.DataFrame, salida: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matriz, globales, por_fold = calcular(pred)
    salida.mkdir(parents=True, exist_ok=True)
    matriz.to_csv(salida / "matriz_confusion_global.csv", index=False)
    globales.to_csv(salida / "metricas_externas_globales.csv", index=False)
    por_fold.to_csv(salida / "metricas_externas_por_fold.csv", index=False)
    return matriz, globales, por_fold


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ruta = Path(argv[0]) if argv else POR_DEFECTO
    salida = Path(argv[1]) if len(argv) > 1 else ruta.parent
    pred = pd.read_csv(ruta, dtype={"fold_externo": "Int64"})
    matriz, globales, _ = escribir(pred, salida)
    print(f"{len(pred)} predicciones, {len(matriz)} grupos (modelo x assigned x escenario) -> {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
