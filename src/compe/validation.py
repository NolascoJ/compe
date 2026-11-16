"""Validacion cruzada agrupada y metricas. Toda comparacion de modelos pasa por aca."""
from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import config as cfg
from .data import feature_groups


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = cfg.SEED, grouped: bool = True):
    """Folds estratificados.

    grouped=True (principal): filas con el mismo X caen siempre en el mismo fold.
    grouped=False (control): Stratified K-Fold comun; las copias se reparten entre folds.
    """
    y = df[cfg.TARGET]
    if grouped:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(df, y, groups=feature_groups(df)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(df, y))


def model_scores(model, X, kind: str) -> np.ndarray:
    """Puntuacion continua: probabilidad de cancelar o funcion de decision."""
    if kind == "proba":
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


@dataclass
class CVResult:
    oof: np.ndarray                                   # puntuacion fuera de fold por fila
    train: list = field(default_factory=list)         # puntuacion sobre el entrenamiento de cada fold
    fit_seconds: float = 0.0


def _one_fold(model, df, y, tr, va, kind):
    t = time.time()
    m = clone(model).fit(df.iloc[tr], y[tr])
    secs = time.time() - t
    return model_scores(m, df.iloc[va], kind), model_scores(m, df.iloc[tr], kind), secs


def cv_scores(model, df: pd.DataFrame, folds, kind: str = "proba", n_jobs: int = 1) -> CVResult:
    """Ajusta el pipeline en cada fold y devuelve puntuaciones de validacion y de entrenamiento."""
    y = df[cfg.TARGET].to_numpy()
    out = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(model, df, y, tr, va, kind) for tr, va in folds)
    oof = np.zeros(len(df))
    for (tr, va), (sv, _, _) in zip(folds, out):
        oof[va] = sv
    return CVResult(oof=oof, train=[o[1] for o in out], fit_seconds=float(sum(o[2] for o in out)))


def _ba(y, pred) -> float:
    return 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())


def best_threshold(y, s) -> tuple[float, float]:
    """Punto de corte que maximiza balanced accuracy. Candidatos: cuantiles de la puntuacion."""
    cands = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 193)))
    scores = [_ba(y, s >= t) for t in cands]
    i = int(np.argmax(scores))
    return float(cands[i]), float(scores[i])


def metrics(df: pd.DataFrame, res: CVResult, folds) -> dict:
    """Metricas con el punto de corte elegido sobre las predicciones fuera de fold."""
    y = df[cfg.TARGET].to_numpy()
    thr, ba = best_threshold(y, res.oof)
    pred = res.oof >= thr
    por_fold = [_ba(y[va], pred[va]) for _, va in folds]
    ba_train = float(np.mean([_ba(y[tr], s >= thr) for (tr, _), s in zip(folds, res.train)]))
    tp = float((pred & (y == 1)).sum())
    precision = tp / max(pred.sum(), 1)
    recall = tp / (y == 1).sum()
    return {
        "ba_val": ba,
        "ba_train": ba_train,
        "gap": ba_train - ba,
        "ba_std": float(np.std(por_fold)),
        "umbral": thr,
        "roc_auc": float(roc_auc_score(y, res.oof)),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "precision": float(precision),
        "recall": float(recall),
        "completados_detectados": float((~pred[y == 0]).mean()),
        "fit_seconds": res.fit_seconds,
    }


def subgroup_table(df: pd.DataFrame, oof: np.ndarray, thr: float) -> pd.DataFrame:
    """Balanced accuracy por subgrupo (test desconocido) y deteccion de no_show.

    execution_status se usa solo para este desglose, nunca como entrada del modelo.
    """
    y = df[cfg.TARGET].to_numpy()
    pred = oof >= thr
    masks = {
        "todo": np.ones(len(df), bool),
        "sin_locked_allocation": (df["reservation_commitment"] != "locked_allocation").to_numpy(),
        "region_PRT": (df["client_region"] == "PRT").to_numpy(),
        "region_resto": (df["client_region"] != "PRT").to_numpy(),
        **{f"anio_{a}": (df["execution_year"] == a).to_numpy() for a in sorted(df["execution_year"].unique())},
    }
    rows = []
    for name, m in masks.items():
        yy, pp = y[m], pred[m]
        rows.append({
            "subgrupo": name, "filas": int(m.sum()), "tasa_cancelacion": float(yy.mean()),
            "ba": float(_ba(yy, pp)), "cancelados_detectados": float(pp[yy == 1].mean()),
            "completados_detectados": float((~pp[yy == 0]).mean()),
        })
    ns = (df["execution_status"] == "no_show").to_numpy()
    rows.append({"subgrupo": "no_show", "filas": int(ns.sum()), "tasa_cancelacion": 1.0, "ba": float("nan"),
                 "cancelados_detectados": float(pred[ns].mean()), "completados_detectados": float("nan")})
    return pd.DataFrame(rows)
