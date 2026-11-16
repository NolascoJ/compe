"""Preprocesamiento como transformador de sklearn.

Se ajusta dentro de cada fold (aprende que categorias son frecuentes) y es la
misma pieza que despues va dentro del predictor final.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as cfg

# total_gpu_hours y total_processes se conservan (EDA paso 5.1): la cancelacion sube con la duracion total.
DROP_ALWAYS = [cfg.TARGET, *cfg.LEAK_COLS, "execution_week"]
OTROS = "otros"
# Etiqueta del nulo por columna: en los identificadores significa "no aplica".
NULL_LABELS = {
    "orchestration_agent_id": "sin_agente",
    "organization_id": "sin_organizacion",
    "client_region": "region_desconocida",
}
NULL_DEFAULT = "desconocido"


def _as_label(s: pd.Series) -> pd.Series:
    """Categoria o ID -> texto, con nulos explicitos. 9.0 y 9 quedan iguales."""
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype("Int64")
    return s.astype("string").fillna(NULL_LABELS.get(s.name, NULL_DEFAULT)).astype(object)


class Preprocessor(BaseEstimator, TransformerMixin):
    """Descarta fugas y redundancias, deriva variables y agrupa categorias raras.

    use_assigned: incluir `assigned_gpu_type` y el indicador de GPU cambiada.
    min_count: categorias con menos filas que esto (en el ajuste) pasan a "(otros)";
               las no vistas al predecir tambien.
    """

    def __init__(self, use_assigned: bool = True, min_count: int = 20, cost_cap_q: float = 0.99):
        self.use_assigned = use_assigned
        self.min_count = min_count
        self.cost_cap_q = cost_cap_q

    def _cat_cols(self):
        cols = cfg.CAT_COLS + cfg.ID_COLS
        return cols if self.use_assigned else [c for c in cols if c != "assigned_gpu_type"]

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_ = {}
        for c in self._cat_cols():
            vc = _as_label(X[c]).value_counts()
            keep = sorted(vc[vc >= self.min_count].index)
            # La etiqueta de nulo existe siempre, aunque en el ajuste no haya nulos.
            nulo = NULL_LABELS.get(c, NULL_DEFAULT)
            self.categories_[c] = sorted(set(keep) | {nulo}) + [OTROS]
        # Tope del costo: se aprende en el ajuste y se reutiliza al predecir.
        self.cost_cap_ = float(X["estimated_cost_per_hour"].quantile(self.cost_cap_q))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.drop(columns=[c for c in DROP_ALWAYS if c in X.columns]).copy()
        fecha = pd.to_datetime(
            dict(year=X["execution_year"], month=X["execution_month"], day=X["execution_day"]),
            errors="coerce",
        )
        X["execution_dow"] = fecha.dt.day_of_week
        # Reservas vacias: 0 horas de GPU y costo 0 (EDA paso 3, propuesta d).
        X["empty_job"] = (
            (X["peak_gpu_hours"] + X["offpeak_gpu_hours"] == 0) & (X["estimated_cost_per_hour"] == 0)
        ).astype(int)
        # Valor extremo de costo recortado al tope (EDA paso 3, propuesta b).
        X["estimated_cost_per_hour"] = X["estimated_cost_per_hour"].clip(upper=self.cost_cap_)
        # Unico nulo numerico (EDA paso 1, propuesta b).
        X["num_auxiliary_processes"] = X["num_auxiliary_processes"].fillna(0)
        X["has_organization"] = X["organization_id"].notna().astype(int)
        X["has_agent"] = X["orchestration_agent_id"].notna().astype(int)
        if self.use_assigned:
            X["gpu_changed"] = (X["reserved_gpu_type"] != X["assigned_gpu_type"]).astype(int)
        else:
            X = X.drop(columns=["assigned_gpu_type"])
        for c, cats in self.categories_.items():
            lab = _as_label(X[c])
            lab = lab.where(lab.isin(cats), OTROS)
            X[c] = pd.Categorical(lab, categories=cats)
        return X
