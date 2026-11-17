# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TP3 · Modelos, errores y decisión final · Grupo 9
#
# Esta notebook reúne la evidencia del TP y completa el análisis data-centered.
# Reutiliza las predicciones externas existentes: **no entrena ni abre test.csv**.
# El baseline es la regresión logística calibrada sin GPU asignada; el final es HistGB.
# Los resultados nuevos son descriptivos y posteriores a la búsqueda histórica.
#
# Esta notebook contiene el análisis para la entrega. Para regenerar sus tablas,
# ejecutar esta notebook desde el repo con el entorno del proyecto instalado.

# %%
from pathlib import Path
import importlib.util
import sys

ROOT = Path.cwd().resolve()
if not (ROOT / "src/compe").is_dir():
    ROOT = ROOT.parent
if not (ROOT / "src/compe").is_dir():
    raise RuntimeError("Abrir la notebook desde la raíz del repo o notebooks/.")
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("analisis_tp3", ROOT / "experiments/analisis_consigna/analizar.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
analysis.main()

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

RESULTS = ROOT / "experiments/analisis_consigna"
comparison = pd.read_csv(RESULTS / "comparacion_modelos.csv")
cohorts = pd.read_csv(RESULTS / "cohortes.csv")
delta = pd.read_csv(RESULTS / "mejoras_cohortes.csv")

# %% [markdown]
# ## Datos y preprocesamiento
#
# TRAIN contiene 38.274 filas. Se conserva el split previo al preprocesamiento.
# Se excluyen target, estado y fecha de estado; GPU asignada también se excluye del candidato.
# Nulos y categorías raras se tratan dentro del entrenamiento de cada fold.
# Los modelos lineales usan one-hot, logaritmos y escalado; HistGB usa categorías nativas.
# Las decisiones completas y el EDA están en `01_eda.ipynb` y `reports/eda_decisiones.md`.
#
# La CV principal utiliza cinco folds agrupados; el corte de cada fold se elige con tres
# folds internos. La búsqueda histórica de hiperparámetros no fue totalmente anidada.
# El agrupamiento original usa las entradas originales; el control posterior con las
# entradas efectivas obtuvo 84,68% BA para HistGB (`output/revision_modelos_20261116/`).

# %% [markdown]
# ## Baseline y dos métricas
#
# **Baseline:** regresión logística, C=0,01677985, class_weight=balanced y l1_ratio=0.
# Es la referencia lineal del informe; no es el ensemble enviado en la primera entrega.
#
# **Balanced accuracy** pondera ambos recalls por igual y es la métrica principal de
# competencia. **ROC-AUC** permite analizar el ordenamiento de scores sin fijar un corte.
# La selección histórica usó BA, gap y simplicidad; AUC se registró como diagnóstico.
# Esta comparación informa ambas sin redefinir retrospectivamente el objetivo de búsqueda.

# %%
display(comparison[["modelo", "ba", "ba_std_folds", "roc_auc", "delta_ba_pp", "FP", "FN"]])
fig, ax = plt.subplots(figsize=(8, 3.5))
labels = ["Logística\n(baseline)", "GradientBoosting", "HistGB\n(final)"]
ax.bar(labels, 100 * comparison.ba, color=["#8292A2", "#478E9B", "#125E71"])
ax.errorbar(labels, 100 * comparison.ba, yerr=100 * comparison.ba_std_folds,
            fmt="none", color="#243746", capsize=5)
ax.set_ylim(0, 100)
ax.set_ylabel("Balanced accuracy (%)")
ax.set_title("Validación externa: mismos folds y calibración interna")
ax.spines[["top", "right"]].set_visible(False)
for i, v in enumerate(comparison.ba):
    ax.text(i, 100 * v + 2, f"{v:.2%}", ha="center")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Errores del baseline y del final por cohortes
#
# Se incluyen todas las categorías de client_type, service_tier y reserved_gpu_type.
# FP: completado marcado como cancelación. FN: cancelado marcado como vigente.
# Cada dimensión reconstruye la matriz global; dimensiones diferentes se superponen.
# BA/AUC no se definen sin ambas clases. `soporte_bajo` indica menos de 30 casos de alguna
# clase; las métricas de categorías diminutas no justifican conclusiones de generalización.

# %%
for dimension in analysis.COHORTS:
    display(Markdown(f"### {dimension}"))
    display(cohorts[(cohorts.dimension == dimension) & cohorts.modelo.isin(
        ["logistic", "hist_gradient_boosting"])][[
        "modelo", "cohorte", "n", "n0", "n1", "ba", "roc_auc", "recall",
        "especificidad", "FP", "FN", "pct_FN_total", "soporte_bajo"]])
    display(delta[delta.dimension == dimension])

# %% [markdown]
# Entre las categorías con soporte suficiente, el baseline tiene menor BA en
# independent_client (61,56%), basic (74,08%) y GPU_T2 (72,83%).
# El volumen de errores también importa: cloud_marketplace concentra 57,79% de sus FN.
# HistGB mejora la BA en esas cohortes, pero las mejoras no son universales:
# basic pasa de 436 a 443 FN y GPU_T6 de 53 a 64, pese a mejorar su BA.

# %% [markdown]
# ## Cohorte prioritaria: reservas creadas con 0–7 horas de anticipación
#
# El campo hours_before_execution mide creación→ejecución, **en horas** según la consigna.
# No mide tiempo restante desde la predicción semanal ni cuándo ocurrió la cancelación.
# Esta cohorte tiene 5.220 reservas (13,64% de TRAIN). HistGB detecta solo 25,29% de sus
# cancelaciones, con 517 FN, el 20,55% de todos sus FN. El problema persiste frente al baseline.

# %%
display(cohorts[(cohorts.dimension == "anticipacion") & (cohorts.cohorte == "0-7")][[
    "modelo", "n", "pct_filas", "ba", "recall", "FP", "FN", "pct_FN_total"]])

# %% [markdown]
# **Impacto operativo posible:** las cancelaciones omitidas pueden retener capacidad
# que no se usa; con poca anticipación hay menos margen para ofrecerla nuevamente.
# Los FP podrían provocar contactos o reasignaciones incorrectas. No se monetiza el impacto
# porque no se conocen costos ni política de actuación.
#
# **Hipótesis:** faltan señales actuales de confirmación, preparación del workload y actividad.
# El análisis observacional no prueba causas ni que una nueva variable vaya a mejorar.
#
# **Intervención propuesta:** capturar eventos con fecha, confirmación, contenedor/datos
# preparados y pago acreditado. Obtener IDs estables y fotografías semanales de reservas
# activas; construir features usando exclusivamente información anterior a cada predicción.
# Comparar con/sin nuevas señales sobre datos futuros reservados y medir BA/AUC globales,
# recall/FPR de la cohorte y FN por 1.000 reservas. Validar también por cliente y tiempo.
# No se afirma que la intervención haya sido ejecutada.

# %% [markdown]
# ## Curva de aprendizaje y adquisición de datos
#
# Las curvas varían cantidad de entrenamiento, mantienen completa la validación y eligen
# cortes internamente. La logística converge cerca del 82% con brecha pequeña: limitación
# de capacidad. HistGB mejora con más datos, con rendimientos decrecientes; su brecha baja.
# La ganancia observada no garantiza una extrapolación a poblaciones distintas.

# %%
display(Image(filename=str(ROOT / "experiments/curva_aprendizaje/curva_aprendizaje.png")))
display(Markdown((ROOT / "reports/adquisicion_datos.md").read_text()))

# %% [markdown]
# ## Estrategias exploradas y decisión final
#
# - Comparar familias: HistGB mejora al baseline +2,82 pp BA y +0,03167 AUC.
# - Ajustar hiperparámetros: búsquedas amplias, refinamientos y confirmaciones registradas;
#   más complejidad no justificó reemplazar el HistGB final de 100 iteraciones y 31 hojas.
# - Ensembles: promedio y stacking no mejoran BA con el mismo protocolo externo.
#   Stacking mejora ligeramente AUC, pero no BA: se mantiene HistGB por el objetivo y simplicidad.

# %%
ensembles = pd.read_csv(ROOT / "experiments/seleccion/stacking_resultados/a92fbb4e4778849e/resultados/metricas_externas_globales.csv")
display(ensembles[ensembles.escenario == "cv"][["modelo", "balanced_accuracy", "roc_auc", "FP", "FN"]])

# %% [markdown]
# **Final:** HistGB sin assigned; umbral fijo 0,38294747099493937. BA externa 84,83%.
# El holdout interno ya evaluado dio BA 85,09%, AUC 0,93774; no se lo vuelve a consultar aquí.
# Las pruebas por años excluidos dieron 77,47–84,32% BA y la extrapolación regional extrema
# llegó a 51,47%. El rendimiento del test oculto depende de su distribución.
#
# ## Entrega y contrato
#
# `predictor_v1_Grupo_9(X)`, `predictor_v2_Grupo_9(X)` y `predictor(X)` devuelven el mismo
# ndarray 0/1, una etiqueta por fila y mismo orden. No ajustan el modelo ni su umbral.
# El nombre v1 satisface el nombre pedido por el PDF sin cambiar las predicciones v2.
#
# El artefacto se probó en un entorno local limpio de Python 3.12 con versiones exactas,
# sin warnings y con predicciones idénticas al ejemplo. Esa prueba no equivale a Colab.
# `entrega/validacion_colab.ipynb` permite verificarlo allí.
# Sigue pendiente confirmar que el evaluador entregue high_value_client y ambos totales.
# No se rellenan ni aproximan esas entradas porque son features activas del modelo.

# %%
import json
display(json.loads((ROOT / "entrega/compatibilidad_py312.json").read_text()))
