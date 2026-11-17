"""Evidencia para la propuesta de adquisición de datos (consigna 3.4).

Usa solo predicciones fuera de fold de TRAIN del HistGB elegido (sin GPU asignada, corte interno de cada fold),
tomadas de `experiments/seleccion/calibracion/resultados/predicciones_externas.csv`. No entrena ni lee test.csv.
Describe dónde se concentran los errores según qué información falta en la reserva.

Uso: PYTHONPATH=src .venv/bin/python experiments/adquisicion_datos/evidencia.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from compe import config as cfg
from compe.data import feature_groups, load_train

AQUI = Path(__file__).resolve().parent
PRED = cfg.EXPERIMENTS / "seleccion" / "calibracion" / "resultados" / "predicciones_externas.csv"


def cargar() -> pd.DataFrame:
    p = pd.read_csv(PRED, usecols=["modelo", "assigned", "escenario", "fila_train", "y_true", "score", "y_pred", "umbral"])
    p = p[(p.modelo == "hist_gradient_boosting") & (~p.assigned) & (p.escenario == "cv")]
    df = load_train()
    assert len(p) == len(df) and p.fila_train.is_unique
    p = p.set_index("fila_train").sort_index()
    assert (p.y_true.to_numpy() == df[cfg.TARGET].to_numpy()).all()
    df = df.join(p[["score", "y_pred", "umbral"]])
    df["tamano_grupo"] = feature_groups(df).map(feature_groups(df).value_counts())
    return df


def segmentos(df: pd.DataFrame) -> dict[str, pd.Series]:
    sin_hist = (df.returning_client == 0) & (df.previous_job_cancellations == 0) & (df.successful_previous_jobs == 0)
    frec_region = df.client_region.map(df.client_region.value_counts())
    lead = pd.cut(df.hours_before_execution, [-1, 7, 30, 90, 180, 365, np.inf],
                  labels=["0-7", "8-30", "31-90", "91-180", "181-365", ">365"])
    margen = (df.score - df.umbral).abs()
    s = {
        "todo": pd.Series("todo", index=df.index),
        "compromiso": np.where(df.reservation_commitment == "locked_allocation", "locked_allocation", "otros compromisos"),
        "historial": np.where(sin_hist, "sin historial previo", "con historial previo"),
        "agente": np.where(df.orchestration_agent_id.isna(), "sin agente", "con agente"),
        "region": np.where(df.client_region == "PRT", "PRT",
                           np.where(frec_region >= 20, "otra region frecuente", "region rara (<20 filas)")),
        "anticipacion": lead.astype(str),
        "pedidos_especiales": np.where(df.special_resource_requests > 0, "con pedidos", "sin pedidos"),
        "copias": np.where(df.tamano_grupo > 1, "X repetido (reserva en bloque)", "X unico"),
        "estado_real": df.execution_status.astype(str),
        "margen_al_corte": pd.cut(margen, [-0.01, 0.1, 0.25, 1.0], labels=["<0,10", "0,10-0,25", ">0,25"]).astype(str),
    }
    return {k: pd.Series(v, index=df.index) for k, v in s.items()}


def tabla(df: pd.DataFrame) -> pd.DataFrame:
    y, p = df[cfg.TARGET].to_numpy() == 1, df.y_pred.to_numpy() == 1
    fn_tot, fp_tot = (y & ~p).sum(), (~y & p).sum()
    filas = []
    for dim, seg in segmentos(df).items():
        for valor in seg.unique():
            m = (seg == valor).to_numpy()
            yy, pp = y[m], p[m]
            rec = pp[yy].mean() if yy.any() else np.nan
            esp = (~pp[~yy]).mean() if (~yy).any() else np.nan
            fn, fp = int((yy & ~pp).sum()), int((~yy & pp).sum())
            filas.append({"dimension": dim, "segmento": valor, "n": int(m.sum()), "pct_filas": m.mean(),
                          "tasa_cancelacion": yy.mean(), "ba": 0.5 * (rec + esp) if yy.any() and (~yy).any() else np.nan,
                          "recall": rec, "especificidad": esp, "fn": fn, "fp": fp,
                          "tasa_error": (fn + fp) / m.sum(), "pct_fn_total": fn / fn_tot, "pct_fp_total": fp / fp_tot,
                          "pct_error_total": (fn + fp) / (fn_tot + fp_tot)})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    df = cargar()
    t = tabla(df)
    t.to_csv(AQUI / "evidencia_errores.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(4).to_string(index=False))
