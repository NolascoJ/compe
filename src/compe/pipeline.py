"""Armado del pipeline: preprocesamiento -> codificacion/escalado -> modelo.

Todo queda dentro de un unico Pipeline de sklearn: cada paso se ajusta solo con
los datos de entrenamiento de cada fold y el predictor final es un solo objeto.

Variantes (TODO.md, seccion 2):
  A  lineal      one-hot, log en numericas con cola larga, estandarizacion, sin columnas redundantes
  B  distancias  igual que A
  D  arboles     codificacion ordinal, sin escalado
  E  nativas     las categoricas pasan como dtype category al modelo
"""
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import Preprocessor

# Numericas con cola larga a la derecha (EDA paso 3): se les aplica log(1 + x).
SKEWED = [
    "hours_before_execution", "peak_gpu_hours", "offpeak_gpu_hours", "total_gpu_hours",
    "previous_job_cancellations", "successful_previous_jobs", "resource_modifications",
    "queue_wait_time", "estimated_cost_per_hour",
]
# Redundantes para un modelo lineal (EDA paso 6).
DROP_LINEAR = ["total_processes", "high_value_client"]
CATEGORICAL = make_column_selector(dtype_include="category")


def _linear_encoder() -> ColumnTransformer:
    skew = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())])
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("skew", skew, SKEWED),
            ("drop", "drop", DROP_LINEAR),
        ],
        remainder=StandardScaler(),
        verbose_feature_names_out=False,
    )


def _ordinal_encoder() -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def make_preprocessing(variante: str = "A", **prep_kwargs) -> Pipeline:
    """Preprocesamiento completo, sin modelo."""
    steps = [("prep", Preprocessor(**prep_kwargs))]
    if variante in ("A", "B"):
        steps.append(("encode", _linear_encoder()))
    elif variante == "D":
        steps.append(("encode", _ordinal_encoder()))
    elif variante != "E":
        raise ValueError(f"variante desconocida: {variante}")
    return Pipeline(steps)


def make_pipeline(model, variante: str = "A", **prep_kwargs) -> Pipeline:
    """Preprocesamiento + modelo en un solo objeto."""
    return Pipeline([*make_preprocessing(variante, **prep_kwargs).steps, ("model", model)])
