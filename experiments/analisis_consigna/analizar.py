"""Completa puntos 2.3, 3.1 y 3.3 desde predicciones externas existentes.

No entrena ni abre test.csv. Ejecutar desde la raíz:
PYTHONPATH=src .venv/bin/python experiments/analisis_consigna/analizar.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compe import config as cfg
from compe.data import load_train

OUT = Path(__file__).resolve().parent
SOURCE = cfg.EXPERIMENTS / "seleccion/calibracion/resultados/predicciones_externas.csv"
MODELS = ("logistic", "gradient_boosting", "hist_gradient_boosting")
COHORTS = ("client_type", "service_tier", "reserved_gpu_type")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(y, pred, score):
    y, pred = np.asarray(y) == 1, np.asarray(pred) == 1
    tn, fp = int((~y & ~pred).sum()), int((~y & pred).sum())
    fn, tp = int((y & ~pred).sum()), int((y & pred).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return dict(n=len(y), n0=tn + fp, n1=fn + tp, TN=tn, FP=fp, FN=fn, TP=tp,
                tasa_cancelacion=y.mean(), recall=recall, especificidad=specificity,
                fpr=1 - specificity, fnr=1 - recall,
                ba=(recall + specificity) / 2,
                roc_auc=roc_auc_score(y, score) if y.any() and (~y).any() else np.nan)


def load_predictions(df):
    source = SOURCE if SOURCE.exists() else SOURCE.with_suffix(".csv.gz")
    cols = ["modelo", "assigned", "escenario", "fold_externo", "fila_train",
            "y_true", "score", "y_pred", "umbral"]
    p = pd.read_csv(source, usecols=cols)
    p = p[p.modelo.isin(MODELS) & ~p.assigned & p.escenario.eq("cv")].copy()
    by_model = {}
    for name in MODELS:
        q = p[p.modelo.eq(name)].sort_values("fila_train").reset_index(drop=True)
        if not np.array_equal(q.fila_train, np.arange(len(df))):
            raise ValueError(f"Cobertura o índices incorrectos: {name}")
        if not np.array_equal(q.y_true, df[cfg.TARGET]):
            raise ValueError(f"Etiquetas desalineadas: {name}")
        if not np.isfinite(q[["score", "umbral"]].to_numpy()).all():
            raise ValueError(f"Scores o cortes no finitos: {name}")
        if not np.array_equal(q.y_pred, (q.score >= q.umbral).astype(int)):
            raise ValueError(f"Predicciones inconsistentes: {name}")
        if q.fold_externo.nunique() != 5:
            raise ValueError(f"No hay cinco folds: {name}")
        if by_model and not np.array_equal(q.fold_externo, by_model[MODELS[0]].fold_externo):
            raise ValueError("Los modelos no comparten folds externos")
        by_model[name] = q
    return source, by_model


def tables(df, predictions):
    rows, global_rows, fold_rows = [], [], []
    for name, q in predictions.items():
        global_stats = stats(q.y_true, q.y_pred, q.score)
        global_rows.append({"modelo": name, **global_stats})
        for fold, f in q.groupby("fold_externo"):
            fold_rows.append({"modelo": name, "fold": fold, **stats(f.y_true, f.y_pred, f.score)})
        dimensions = {c: df[c].fillna("(faltante)").astype(str) for c in COHORTS}
        dimensions["anticipacion"] = pd.Series(np.where(df.hours_before_execution <= 7, "0-7", ">7"))
        for dim, values in dimensions.items():
            for value in sorted(values.unique()):
                mask = values.eq(value).to_numpy()
                f = q.loc[mask]
                s = stats(f.y_true, f.y_pred, f.score)
                rows.append({"modelo": name, "dimension": dim, "cohorte": value, **s,
                             "pct_filas": s["n"] / len(df),
                             "pct_FP_total": s["FP"] / global_stats["FP"],
                             "pct_FN_total": s["FN"] / global_stats["FN"],
                             "soporte_bajo": min(s["n0"], s["n1"]) < 30})
    cohorts, comparison, folds = pd.DataFrame(rows), pd.DataFrame(global_rows), pd.DataFrame(fold_rows)
    # Cada partición de cohortes debe reconstruir la matriz global de cada modelo.
    for (name, _), g in cohorts.groupby(["modelo", "dimension"]):
        expected = comparison.set_index("modelo").loc[name]
        for c in ["n", "TN", "FP", "FN", "TP"]:
            if g[c].sum() != expected[c]:
                raise ValueError(f"Las cohortes no reconstruyen {c} de {name}")
    for c in ["ba", "roc_auc"]:
        comparison[f"{c}_std_folds"] = comparison.modelo.map(folds.groupby("modelo")[c].std(ddof=0))
    baseline = comparison.set_index("modelo").loc["logistic"]
    comparison["delta_ba_pp"] = 100 * (comparison.ba - baseline.ba)
    comparison["delta_auc"] = comparison.roc_auc - baseline.roc_auc
    base = cohorts[cohorts.modelo.eq("logistic")].set_index(["dimension", "cohorte"])
    final = cohorts[cohorts.modelo.eq("hist_gradient_boosting")].set_index(["dimension", "cohorte"])
    delta = final[["n", "n0", "n1", "soporte_bajo"]].copy()
    for c in ["ba", "roc_auc", "FP", "FN"]:
        delta[f"{c}_baseline"] = base[c]
        delta[f"{c}_final"] = final[c]
        delta[f"delta_{c}"] = final[c] - base[c]
    return cohorts, comparison, folds, delta.reset_index()


def md_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in rows)


def pct(v):
    return "—" if pd.isna(v) else f"{100*v:.2f}%"


def report(cohorts, comparison, delta):
    names = {"logistic": "Logística (baseline)", "gradient_boosting": "GradientBoosting", "hist_gradient_boosting": "HistGB (final)"}
    text = """# Baseline, análisis de errores y estrategia de intervención

## Alcance y baseline (2.1–2.3)

Se declara como baseline del informe la **regresión logística calibrada sin GPU asignada**.
Es una referencia lineal interpretable, ya ajustada mediante búsqueda y confirmación sobre TRAIN:
`C=0.016779852749843744`, `class_weight='balanced'`, `l1_ratio=0`.
Usa one-hot, logaritmos y escalado dentro del pipeline. No es el predictor de la primera entrega
(RF + CatBoost), que corresponde a otro desarrollo y no tiene OOF comparables en este análisis.

Este diagnóstico se agrega después de la búsqueda histórica. Las nuevas cohortes no se presentan
como causa de decisiones anteriores. No se entrena ni se cambia el candidato o su umbral.

### Dos métricas y su papel

- **Balanced accuracy (BA)** = (recall de cancelados + recall de completados) / 2.
  Pondera ambas clases por igual, evita premiar la predicción constante de la clase mayoritaria
  y coincide con el objetivo de competencia. Es el criterio principal para hiperparámetros y
  para el corte de decisión, que se aprende solo en el entrenamiento de cada fold externo.
- **ROC-AUC** mide el ordenamiento de los scores a través de distintos cortes. Se usa como
  diagnóstico secundario al comparar modelos y configuraciones: permite distinguir una mejora
  de ordenamiento de una mejora debida al corte. No determina el umbral ni garantiza probabilidades
  calibradas. Si BA y AUC discrepan se conserva la prioridad de BA y se explica la discrepancia.

La búsqueda histórica registró ambas métricas, pero su regla automática seleccionó por BA,
gap y simplicidad; no se afirma que AUC haya sido un segundo objetivo de optimización.
La comparación de cierre considera explícitamente ambas. No se introduce ahora una búsqueda nueva.

### Protocolo comparable

Solo TRAIN (38.274 filas), mismos cinco folds externos agrupados y mismos índices por modelo.
Cada corte se elige con tres folds internos del entrenamiento externo; preprocesamiento ajustado
dentro de su entrenamiento. Una predicción externa por fila. Todos los modelos de esta tabla
excluyen `assigned_gpu_type` y `gpu_changed`. BA/AUC globales se calculan concatenando OOF;
también se guarda el desvío entre folds (ddof=0), que no es un intervalo de confianza.
La selección histórica de hiperparámetros no fue totalmente anidada; persiste optimismo de selección.

"""
    text += md_table(["Modelo", "BA", "Desvío BA (pp)", "ROC-AUC", "Δ BA vs baseline (pp)", "FP", "FN"], [
        [names[r.modelo], pct(r.ba), f"{r.ba_std_folds*100:.2f}", f"{r.roc_auc:.4f}", f"{r.delta_ba_pp:+.2f}", r.FP, r.FN]
        for r in comparison.itertuples()])
    text += """

HistGB mejora el baseline en ambas métricas. No se atribuye toda la mejora a una sola feature:
cambian familia, representación categórica y capacidad. El diagnóstico de curvas de aprendizaje
es consistente con limitación de capacidad del modelo lineal.

## Errores por cohortes exigidas (3.1)

**FP:** una reserva que se completa es marcada como cancelación; podría provocar contactos
innecesarios o una reasignación incorrecta de recursos. **FN:** una reserva cancelada se marca
como vigente; podría retener capacidad que finalmente no se utiliza. Son consecuencias posibles,
no costos medidos: no hay datos de utilización, penalidades ni política de acción para monetizarlas.
Las predicciones deben apoyar una revisión, no justificar cancelaciones automáticas.

Las cohortes se definen por los valores originales de `client_type`, `service_tier` y
`reserved_gpu_type`, antes del agrupamiento de categorías raras. Se incluyen todas las categorías.
`n0/n1` son completados/cancelados; BA y AUC quedan indefinidas si falta una clase.
El asterisco identifica menos de 30 observaciones de alguna clase: no priorizar por BA sin mirar soporte.
Las dimensiones se superponen; no sumar errores de dimensiones distintas.

"""
    for dim in COHORTS:
        text += f"### {dim}\n\n"
        d = delta[delta.dimension.eq(dim)]
        text += md_table(["Cohorte", "n (n0/n1)", "BA base", "BA final", "AUC base/final", "FP base→final", "FN base→final", "Δ BA (pp)"], [
            [r.cohorte + (" *" if r.soporte_bajo else ""), f"{r.n} ({r.n0}/{r.n1})", pct(r.ba_baseline), pct(r.ba_final),
             f"{r.roc_auc_baseline:.3f}/{r.roc_auc_final:.3f}" if pd.notna(r.roc_auc_final) else "—",
             f"{r.FP_baseline}→{r.FP_final}", f"{r.FN_baseline}→{r.FN_final}",
             "—" if pd.isna(r.delta_ba) else f"{100*r.delta_ba:+.2f}"] for r in d.itertuples()]) + "\n\n"
        b = cohorts[cohorts.modelo.eq("logistic") & cohorts.dimension.eq(dim)]
        worst = b[~b.soporte_bajo].sort_values("ba").head(1)
        mass = b.sort_values("FN", ascending=False).iloc[0]
        if len(worst):
            r = worst.iloc[0]
            text += f"Menor BA del baseline con soporte suficiente: **{r.cohorte}**, {pct(r.ba)} (n={r.n}). "
        text += f"Mayor cantidad de FN del baseline: **{mass.cohorte}**, {mass.FN} ({pct(mass.pct_FN_total)} de todos sus FN).\n\n"
    text += """Los CSV adjuntos incluyen recall, especificidad, FPR, FNR, tasa de cancelación,
fracción de filas y participación en el total de FP/FN, además de las matrices completas.
Una tasa baja de error no implica BA alta cuando una clase es muy poco frecuente.

## Cohorte prioritaria e intervención (3.3)

Se prioriza **anticipación registrada entre 0 y 7 inclusive** (`hours_before_execution <= 7`).
La unidad se mantiene en **horas**, según el diccionario del enunciado; no se interpreta como
una semana ni se transforma a días basándose en la hipótesis del origen hotelero del dataset.
Es una cohorte adicional a las tres obligatorias, definida por una regla reproducible y disponible al predecir.

"""
    lead = cohorts[cohorts.dimension.eq("anticipacion") & cohorts.cohorte.eq("0-7")]
    text += md_table(["Modelo", "n", "% filas", "BA", "Recall", "FP", "FN", "% FN totales"], [
        [names[r.modelo], r.n, pct(r.pct_filas), pct(r.ba), pct(r.recall), r.FP, r.FN, pct(r.pct_FN_total)] for r in lead.itertuples()])
    text += """

**Frecuencia e impacto:** la tabla cuantifica volumen y cancelaciones omitidas tanto por el
baseline como por el final. Se prioriza por recall bajo y presión operativa: con poca anticipación
hay menos margen para confirmar la reserva o volver a ofrecer capacidad liberada.
Esto es una hipótesis de operación; el dataset no mide capacidad desperdiciada.

**Causas posibles:** las variables estáticas pueden no representar el estado más reciente de
la reserva; faltan confirmaciones, preparación del workload y actividad del cliente.
También podrían cambiar las tasas de cancelación o existir menos positivos para aprender.
Los desgloses son asociaciones: no prueban ninguna de esas causas.

**Intervención propuesta, aún no implementada:** recopilar confirmación de reserva, carga de
contenedor/datos, pago acreditado y eventos de modificación con timestamp. Definir una fotografía
semanal de reservas activas con timestamp de predicción; usar solo eventos conocidos hasta ese
instante. Incorporar ID estable de reserva/cliente y fechas de creación/cancelación para definir
correctamente quién sigue activo y evitar eventos posteriores al corte.

1. Auditar significado y unidad de `hours_before_execution`: el campo describe creación→ejecución,
   no tiempo restante desde la fotografía semanal. Obtener fecha de creación y de ejecución para
   medir también ese tiempo restante sin confundirlos.
2. Medir cobertura y antigüedad de las nuevas señales en esta cohorte y en el resto; auditar faltantes.
3. Comparar HistGB fijo con/sin esas señales usando el mismo protocolo y nuevos datos reservados
   antes del análisis. Agrupar por cliente/reserva cuando haya IDs, además de controlar duplicados.
4. Reportar BA/AUC globales y por las tres cohortes obligatorias, recall, FPR y FN por 1.000 reservas.
   Exigir una mejora consistente entre folds y evaluar incertidumbre; no elegir por una categoría diminuta.
5. Simular una política de confirmación manual; medir contactos, cancelaciones detectadas y costo
   operativo antes de activar decisiones automáticas. La mejora esperada queda como hipótesis.

No se afirma que esta intervención haya mejorado el predictor: hoy no existen esas variables.
La consigna pide discutir la estrategia y los datos a adquirir, no inventar su implementación.

## Estrategias de optimización y decisión final (4–5)

**Selección de modelos:** comparar la referencia lineal con GradientBoosting e HistGB prueba si
la capacidad para interacciones no lineales mejora el sistema. La tabla inicial mantiene folds,
exclusión de assigned y procedimiento de corte comparables; ambos boosting superan la BA/AUC del baseline.

**Optimización de hiperparámetros:** se conservaron búsquedas amplias, refinamientos y
confirmaciones. Su comparación histórica puede reproducirse en `experiments/<modelo>/`.
No se mezclan métricas OOF con corte global y métricas externas con cortes internos para
atribuir una ganancia aislada al tuning. HistGB final conserva 100 iteraciones y 31 hojas por
rendimiento y simplicidad; más complejidad no produjo una ventaja concluyente.

**Ensembles:** promedio BA 84,73%, AUC 0,93525; stacking BA 84,66%, AUC 0,93582;
HistGB BA 84,83%, AUC 0,93554. Se compararon con el mismo protocolo externo. El stacking
ordena ligeramente mejor según AUC, pero no mejora BA y exige más componentes: se conserva HistGB.
Que una estrategia no mejore también es un resultado válido del experimento.

Las tablas por cohorte muestran si la ganancia es global o localizada; no se ocultan regresiones
de categorías particulares. Para comparar esas diferencias hay que considerar soporte y la
selección posterior de cohortes; no son pruebas confirmatorias de significancia.

**Modelo seleccionado:** HistGB sin GPU asignada; umbral de inferencia ya fijado
0,38294747099493937. BA externa 84,83%; BA histórica del holdout interno 85,09% y AUC 0,93774.
Este informe no vuelve a abrir el holdout. No es el test oculto de la competencia.
La expectativa depende de la población: en años excluidos HistGB obtuvo aproximadamente
77,47–84,32% BA; en extrapolación regional extrema llegó a 51,47%. El 85% local no garantiza el oculto.
El control posterior con agrupamiento de entradas efectivas obtuvo 84,68% BA: ver la auditoría existente.

## Reproducción y archivos

```sh
PYTHONPATH=src .venv/bin/python experiments/analisis_consigna/analizar.py
```

- `cohortes.csv`: todas las cohortes y tres modelos; matrices, tasas y soporte.
- `comparacion_modelos.csv` y `metricas_folds.csv`: métricas globales y por fold.
- `mejoras_cohortes.csv`: diferencias final−baseline, incluidos cambios en FP/FN.
- `manifest.json`: hashes, alcance y controles de alineación.

Fuentes relacionadas: [curva de aprendizaje](curva_aprendizaje.md),
[adquisición de datos](adquisicion_datos.md), [selección final](seleccion_final_stacking.md)
y [auditoría](../output/revision_modelos_20261116/informe.md).
"""
    return text


def main():
    df = load_train()
    source, predictions = load_predictions(df)
    cohorts, comparison, folds, delta = tables(df, predictions)
    # Contrastar el nuevo cálculo con el resumen histórico independiente.
    historical = pd.read_csv(SOURCE.parent / "metricas_externas_globales.csv")
    historical = historical[~historical.assigned & historical.escenario.eq("cv")].set_index("modelo")
    for row in comparison.itertuples():
        reference = historical.loc[row.modelo]
        for column in ["n", "TN", "FP", "FN", "TP"]:
            if getattr(row, column) != reference[column]:
                raise ValueError(f"Matriz distinta del resumen histórico: {row.modelo}/{column}")
        np.testing.assert_allclose([row.ba, row.roc_auc],
                                   [reference.balanced_accuracy, reference.roc_auc], rtol=0, atol=1e-12)
    for name, table in [("cohortes", cohorts), ("comparacion_modelos", comparison),
                        ("metricas_folds", folds), ("mejoras_cohortes", delta)]:
        table.to_csv(OUT / f"{name}.csv", index=False)
    (cfg.ROOT / "reports/analisis_consigna.md").write_text(report(cohorts, comparison, delta))
    manifest = dict(training_rows=len(df), models=list(MODELS), baseline="logistic",
                    assigned=False, scenario="cv", external_folds=5, retrained=False,
                    test_csv_accessed=False, aligned_labels=True, shared_external_folds=True,
                    confusion_counts_reconciled=True,
                    historical_metrics_reproduced=True,
                    sources={str(p.relative_to(cfg.ROOT)): digest(p) for p in [cfg.DATA_RAW / "train.csv", source]},
                    script_sha256=digest(Path(__file__)))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(comparison[["modelo", "ba", "roc_auc", "delta_ba_pp", "FP", "FN"]].to_string(index=False))
    print("Cohortes y etiquetas verificadas; sin entrenamiento ni acceso al holdout.")


if __name__ == "__main__":
    main()
