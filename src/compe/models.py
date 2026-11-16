"""Registro de modelos: estimador, preprocesamiento, espacio de busqueda y curva de validacion.

Espacios de busqueda (TODO.md, seccion 4.2). Cada parametro es una tupla:
  ("log", lo, hi)    real, escala logaritmica
  ("float", lo, hi)  real, escala lineal
  ("int", lo, hi)    entero, extremos incluidos
  ("cat", [...])     lista de valores
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import SEED


class EarlyStopping(BaseEstimator, ClassifierMixin):
    """Parada temprana para XGBoost / LightGBM sin tocar el fold de validacion.

    Aparta una parte del entrenamiento (filas identicas siempre del mismo lado),
    busca la cantidad de arboles y reentrena con todo el entrenamiento.
    """

    def __init__(self, estimator, rounds: int = 50, val_frac: float = 0.15, seed: int = SEED):
        self.estimator = estimator
        self.rounds = rounds
        self.val_frac = val_frac
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        grupos = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()
        tr, va = next(GroupShuffleSplit(1, test_size=self.val_frac, random_state=self.seed).split(X, y, grupos))
        take = (lambda A, i: A.iloc[i]) if hasattr(X, "iloc") else (lambda A, i: A[i])
        est = clone(self.estimator)
        if type(est).__module__.startswith("xgboost"):
            est.set_params(early_stopping_rounds=self.rounds)
            est.fit(take(X, tr), y[tr], eval_set=[(take(X, va), y[va])], verbose=False)
            best = est.best_iteration + 1
        else:
            import lightgbm
            est.fit(take(X, tr), y[tr], eval_set=[(take(X, va), y[va])],
                    callbacks=[lightgbm.early_stopping(self.rounds, verbose=False)])
            best = est.best_iteration_ or est.n_estimators
        self.best_n_estimators_ = int(best)
        self.model_ = clone(self.estimator).set_params(n_estimators=self.best_n_estimators_).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


@dataclass
class Spec:
    nombre: str
    familia: str
    variante: str                 # preprocesamiento: A, B, D o E
    score: str                    # "proba" o "decision"
    build: Callable[[dict], object]
    space: dict
    n_iter: int
    curve: tuple                  # (parametro de complejidad, valores)
    simplicity: Callable[[dict], tuple]   # menor = mas simple (mas regularizado)
    fold_jobs: int = 5            # folds en paralelo; 1 si el modelo ya usa todos los nucleos


def _cw(p):
    return p.get("class_weight")


def _logistic(p):
    return LogisticRegression(penalty="elasticnet", solver="saga", C=p.get("C", 1.0), l1_ratio=p.get("l1_ratio", 0.0),
                              class_weight=_cw(p), max_iter=500, random_state=SEED)


def _ridge(p):
    return RidgeClassifier(alpha=p.get("alpha", 1.0), class_weight=_cw(p), random_state=SEED)


def _knn(p):
    return KNeighborsClassifier(n_neighbors=p.get("n_neighbors", 5), weights=p.get("weights", "uniform"), p=p.get("p", 2), n_jobs=-1)


_SVC_CONFIG = {"l2_squared_hinge": ("l2", "squared_hinge"), "l1_squared_hinge": ("l1", "squared_hinge"), "l2_hinge": ("l2", "hinge")}


def _linsvc(p):
    penalty, loss = _SVC_CONFIG[p.get("config", "l2_squared_hinge")]
    return LinearSVC(C=p.get("C", 1.0), penalty=penalty, loss=loss, dual="auto", class_weight=_cw(p), max_iter=5000, random_state=SEED)


def _rf(p):
    return RandomForestClassifier(
        n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth"), min_samples_leaf=p.get("min_samples_leaf", 1),
        min_samples_split=p.get("min_samples_split", 2), max_features=p.get("max_features", "sqrt"),
        max_samples=p.get("max_samples"), class_weight=_cw(p), n_jobs=-1, random_state=SEED)


def _bagging(p):
    base = DecisionTreeClassifier(max_depth=p.get("max_depth"), min_samples_leaf=p.get("min_samples_leaf", 1), random_state=SEED)
    return BaggingClassifier(estimator=base, n_estimators=p.get("n_estimators", 100), max_samples=p.get("max_samples", 1.0),
                             max_features=p.get("max_features", 1.0), n_jobs=-1, random_state=SEED)


def _gb(p):
    return GradientBoostingClassifier(
        learning_rate=p.get("learning_rate", 0.1), n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", 3),
        subsample=p.get("subsample", 1.0), min_samples_leaf=p.get("min_samples_leaf", 1), max_features=p.get("max_features"),
        random_state=SEED)


def _hgb(p):
    # Sin parada temprana interna (su particion no respeta filas identicas): max_iter se busca.
    return HistGradientBoostingClassifier(
        learning_rate=p.get("learning_rate", 0.1), max_iter=p.get("max_iter", 100), max_leaf_nodes=p.get("max_leaf_nodes", 31),
        max_depth=p.get("max_depth"), min_samples_leaf=p.get("min_samples_leaf", 20),
        l2_regularization=p.get("l2_regularization", 0.0), max_features=p.get("max_features", 1.0),
        class_weight=_cw(p), categorical_features="from_dtype", early_stopping=False, random_state=SEED)


def _xgb(p):
    from xgboost import XGBClassifier
    est = XGBClassifier(
        n_estimators=p.get("n_estimators", 3000), learning_rate=p.get("learning_rate", 0.1), max_depth=p.get("max_depth", 6),
        min_child_weight=p.get("min_child_weight", 1), subsample=p.get("subsample", 1.0),
        colsample_bytree=p.get("colsample_bytree", 1.0), gamma=p.get("gamma", 0.0), reg_lambda=p.get("reg_lambda", 1.0),
        reg_alpha=p.get("reg_alpha", 0.0), scale_pos_weight=p.get("scale_pos_weight", 1.0),
        tree_method="hist", enable_categorical=True, eval_metric="logloss", n_jobs=-1, random_state=SEED)
    return EarlyStopping(est)


def _lgbm(p):
    from lightgbm import LGBMClassifier
    est = LGBMClassifier(
        n_estimators=p.get("n_estimators", 3000), learning_rate=p.get("learning_rate", 0.1), num_leaves=p.get("num_leaves", 31),
        max_depth=p.get("max_depth", -1), min_child_samples=p.get("min_child_samples", 20), subsample=p.get("subsample", 1.0),
        subsample_freq=1, colsample_bytree=p.get("colsample_bytree", 1.0), reg_lambda=p.get("reg_lambda", 0.0),
        reg_alpha=p.get("reg_alpha", 0.0), min_split_gain=p.get("min_split_gain", 0.0), cat_smooth=p.get("cat_smooth", 10.0),
        is_unbalance=p.get("is_unbalance", False), n_jobs=-1, random_state=SEED, verbose=-1)
    return EarlyStopping(est)


CW = ("cat", [None, "balanced"])
_depth = lambda p: p.get("max_depth") or 99  # noqa: E731

MODELS = {s.nombre: s for s in [
    Spec("logistic", "Lineales", "A", "proba", _logistic,
         {"C": ("log", 1e-4, 1e2), "l1_ratio": ("cat", [0.0, 0.1, 0.5, 0.9, 1.0]), "class_weight": CW}, 60,
         ("C", [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]), lambda p: (p["C"],)),
    Spec("ridge", "Lineales", "A", "decision", _ridge,
         {"alpha": ("cat", [float(a) for a in np.logspace(-3, 4, 25)]), "class_weight": CW}, 50,
         ("alpha", [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]), lambda p: (-p["alpha"],)),
    Spec("knn", "Distancias", "B", "proba", _knn,
         {"n_neighbors": ("cat", [3, 5, 7, 11, 15, 21, 31, 41, 61, 81, 101, 151, 201, 251, 301]),
          "weights": ("cat", ["uniform", "distance"]), "p": ("cat", [1, 2])}, 60,
         ("n_neighbors", [1, 3, 5, 11, 21, 41, 81, 151, 301]), lambda p: (-p["n_neighbors"],), fold_jobs=1),
    Spec("linear_svc", "SVM", "A", "decision", _linsvc,
         {"C": ("log", 1e-4, 1e1), "config": ("cat", list(_SVC_CONFIG)), "class_weight": CW}, 40,
         ("C", [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]), lambda p: (p["C"],)),
    Spec("random_forest", "Bagging", "D", "proba", _rf,
         {"n_estimators": ("int", 300, 1000), "max_depth": ("cat", [6, 8, 10, 14, 18, 24, 30, None]),
          "min_samples_leaf": ("int", 1, 50), "min_samples_split": ("int", 2, 50),
          "max_features": ("cat", ["sqrt", "log2", 0.2, 0.3, 0.4, 0.6]), "max_samples": ("float", 0.5, 1.0), "class_weight": CW}, 80,
         ("max_depth", [2, 4, 6, 8, 12, 16, 24, None]), lambda p: (_depth(p), -p["min_samples_leaf"]), fold_jobs=1),
    Spec("bagging", "Bagging", "D", "proba", _bagging,
         {"n_estimators": ("int", 100, 500), "max_samples": ("float", 0.4, 1.0), "max_features": ("float", 0.4, 1.0),
          "max_depth": ("cat", [6, 8, 10, 14, 18, 24, 30, None]), "min_samples_leaf": ("int", 1, 50)}, 60,
         ("max_depth", [2, 4, 6, 8, 12, 16, 24, None]), lambda p: (_depth(p), -p["min_samples_leaf"]), fold_jobs=1),
    Spec("gradient_boosting", "Boosting", "D", "proba", _gb,
         {"learning_rate": ("log", 0.01, 0.2), "n_estimators": ("int", 100, 800), "max_depth": ("int", 2, 6),
          "subsample": ("float", 0.5, 1.0), "min_samples_leaf": ("int", 5, 100), "max_features": ("float", 0.3, 1.0)}, 60,
         ("max_depth", [1, 2, 3, 4, 5, 6, 8]), lambda p: (p["max_depth"], p["n_estimators"])),
    Spec("hist_gradient_boosting", "Boosting", "E", "proba", _hgb,
         {"learning_rate": ("log", 0.01, 0.2), "max_iter": ("int", 100, 2000), "max_leaf_nodes": ("int", 7, 127),
          "max_depth": ("cat", [3, 4, 6, 8, 12, None]), "min_samples_leaf": ("int", 10, 300),
          "l2_regularization": ("float", 0.0, 10.0), "max_features": ("float", 0.4, 1.0), "class_weight": CW}, 120,
         ("max_leaf_nodes", [3, 7, 15, 31, 63, 127, 255]), lambda p: (p["max_leaf_nodes"], -p["min_samples_leaf"]), fold_jobs=1),
    Spec("xgboost", "Boosting", "E", "proba", _xgb,
         {"learning_rate": ("log", 0.01, 0.2), "max_depth": ("int", 3, 10), "min_child_weight": ("log", 1, 30),
          "subsample": ("float", 0.5, 1.0), "colsample_bytree": ("float", 0.4, 1.0), "gamma": ("float", 0.0, 5.0),
          "reg_lambda": ("log", 1e-2, 30), "reg_alpha": ("float", 0.0, 5.0), "scale_pos_weight": ("cat", [1.0, 1.4])}, 150,
         ("max_depth", [1, 2, 3, 4, 6, 8, 10, 12]), lambda p: (p["max_depth"], -p["min_child_weight"]), fold_jobs=1),
    Spec("lightgbm", "Boosting", "E", "proba", _lgbm,
         {"learning_rate": ("log", 0.01, 0.2), "num_leaves": ("int", 7, 255), "max_depth": ("cat", [3, 4, 6, 8, 12, -1]),
          "min_child_samples": ("int", 10, 300), "subsample": ("float", 0.5, 1.0), "colsample_bytree": ("float", 0.4, 1.0),
          "reg_lambda": ("log", 1e-2, 30), "reg_alpha": ("float", 0.0, 5.0), "min_split_gain": ("float", 0.0, 1.0),
          "cat_smooth": ("float", 1.0, 50.0), "is_unbalance": ("cat", [False, True])}, 150,
         ("num_leaves", [3, 7, 15, 31, 63, 127, 255]), lambda p: (p["num_leaves"], -p["min_child_samples"]), fold_jobs=1),
]}


# ---------- muestreo del espacio de busqueda ----------

def _draw(spec, rng):
    kind = spec[0]
    if kind == "cat":
        return spec[1][int(rng.integers(len(spec[1])))]
    lo, hi = spec[1], spec[2]
    if kind == "log":
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if kind == "float":
        return float(rng.uniform(lo, hi))
    return int(rng.integers(lo, hi + 1))


def sample_space(space: dict, n: int, seed: int = SEED) -> list[dict]:
    """n combinaciones distintas. Si el espacio es una grilla finita y chica, la recorre entera."""
    import itertools
    rng = np.random.default_rng(seed)
    if all(s[0] == "cat" for s in space.values()):
        grid = [dict(zip(space, v)) for v in itertools.product(*(s[1] for s in space.values()))]
        if len(grid) <= n:
            return grid
        return [grid[i] for i in rng.permutation(len(grid))[:n]]
    return [{k: _draw(s, rng) for k, s in space.items()} for _ in range(n)]


def refine_space(space: dict, top: list[dict]) -> dict:
    """Espacio estrecho alrededor de las mejores combinaciones (paso 3 del protocolo)."""
    out = {}
    for k, s in space.items():
        vals = [p[k] for p in top]
        if s[0] == "cat":
            seen = [v for v in s[1] if v in vals]
            out[k] = ("cat", seen or s[1])
        else:
            lo, hi = min(vals), max(vals)
            out[k] = (s[0], max(s[1], lo), min(s[2], hi)) if hi > lo else ("cat", [lo])
    return out
