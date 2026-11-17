# TP3 · Predicción de cancelaciones · Grupo 9

Clasificación de reservas GPU: `canceled_job=1` indica cancelación o no ejecución. Se usa balanced accuracy (BA) como métrica principal y ROC-AUC como complemento.

El modelo seleccionado es HistGradientBoosting: 100 iteraciones, 31 hojas y umbral **0,38294747099493937**. Excluye `assigned_gpu_type` y `gpu_changed`. Obtuvo **84,83% de BA en validación externa** y **85,09% en el holdout interno**, distinto del conjunto oculto de la competencia.

## Predictor

[entrega/predictor_v2_Grupo_9.py](entrega/predictor_v2_Grupo_9.py) contiene el modelo y el preprocesamiento. No requiere archivos auxiliares para predecir, una vez instaladas sus dependencias.

```python
import pandas as pd
from predictor_v2_Grupo_9 import predictor_v2_Grupo_9

X = pd.read_csv("reservas_activas.csv")
y = predictor_v2_Grupo_9(X)
```

Devuelve un entero 0/1 por fila, conservando el orden de entrada. El artefacto actual fue entrenado únicamente con las 38.274 filas de train. El uso y las dependencias se detallan en [entrega/README.md](entrega/README.md).

El predictor también admite `predictor_v1_Grupo_9(X)`, con las mismas etiquetas 0/1, para el nombre de función del PDF. Se verificó en un entorno limpio local de Python 3.12.12 con las versiones exactas: 300 etiquetas y scores idénticos, sin advertencias. La prueba definitiva en Colab sigue pendiente; [esta notebook](entrega/validacion_colab.ipynb) instala las dependencias y registra la comprobación. También debe confirmarse que el evaluador entregue las columnas activas requeridas, incluidos los totales y `high_value_client`.

## Estructura

| Ruta | Contenido |
| --- | --- |
| `src/compe/config.py` | Columnas, rutas y semilla. |
| `src/compe/data.py` | Carga de datos y grupos de entradas idénticas. |
| `src/compe/features.py` | Nulos, categorías raras y transformaciones aprendidas en entrenamiento. |
| `src/compe/pipeline.py` | Preprocesamiento por familia de modelo. |
| `src/compe/models.py` | Estimadores, espacios de búsqueda y criterio de simplicidad. |
| `src/compe/validation.py` | Folds agrupados, predicciones OOF, umbrales y métricas. |
| `src/compe/run.py` | Búsqueda y evaluación por modelo. |
| `src/compe/resumen.py` | Tabla de resultados originales. |
| `data/raw/` | Datos originales y metadatos del split. |
| `notebooks/` | EDA y notebook de resultados, cohortes y decisión final. |
| `experiments/<modelo>/` | Búsquedas, curvas, confirmaciones y predicciones. |
| `experiments/seleccion/` | Calibración interna, ensembles y controles. |
| `experiments/evaluacion_final_histgb/` | Evaluación del holdout ya realizada y artefacto entrenado. |
| `experiments/curva_aprendizaje/` | Curva de aprendizaje de HistGB y logística sobre train. |
| `experiments/adquisicion_datos/` | Errores OOF de HistGB por segmento. |
| `entrega/` | Predictor autocontenido, construcción, verificación y ejemplo de entrada. |
| `reports/` | EDA, selección, controles y evaluación final. |
| `output/revision_modelos_20261116/` | Auditoría y controles adicionales sobre train. |
| `scripts/` y `artifacts/` | Restauración de archivos comprimidos y manifiesto de integridad. |

Las herramientas de `entrega/` se ejecutan explícitamente: `construir_predictor.py` empaqueta el modelo existente; `verificar_predictor.py` comprueba equivalencia; `probar_predictor.py` lee un CSV y guarda probabilidades y etiquetas. El CSV de 300 filas contiene datos de entrenamiento repetidos y sirve solo para probar la interfaz.

## Desarrollo y reproducción

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/restaurar_artefactos.py

PYTHONPATH=src python -m compe.run --listar
PYTHONPATH=src python experiments/seleccion/calibracion.py todo --dry-run
PYTHONPATH=src python experiments/seleccion/stacking.py todo --dry-run
PYTHONPATH=src python experiments/curva_aprendizaje/curva_aprendizaje.py calcular --jobs 8
PYTHONPATH=src python experiments/curva_aprendizaje/curva_aprendizaje.py resumir
PYTHONPATH=src python experiments/curva_aprendizaje/curva_aprendizaje.py graficar
PYTHONPATH=src python experiments/adquisicion_datos/evidencia.py
```

- [Calibración y robustez](experiments/seleccion/README.md).
- [Promedio y stacking](experiments/seleccion/README_STACKING.md).
- [Uso del predictor](entrega/README.md).

`requirements-lock.txt` registra el entorno de desarrollo histórico; `entrega/requirements.txt` fija las dependencias del predictor. Los rangos de `pyproject.toml` no garantizan compatibilidad con un modelo serializado. `scripts/restaurar_artefactos.py --check` verifica los archivos comprimidos sin restaurarlos.

## Contexto y estado de entrega

- [Contexto técnico y decisiones](docs/CONTEXTO.md).
- [Requisitos completados y pendientes](docs/ENTREGA.md).
- [Mapa del proyecto](docs/ESTRUCTURA.md).
- [Análisis de errores y estrategia de intervención](reports/analisis_consigna.md).
- [Notebook de resultados](notebooks/02_resultados.ipynb).

La publicación del código no equivale al envío a la cátedra. Colab y la disponibilidad de las entradas activas requieren las comprobaciones indicadas en el estado de entrega.

## Resultados y límites

- [Decisiones del EDA](reports/eda_decisiones.md).
- [Selección final y ensembles](reports/seleccion_final_stacking.md).
- [Comparación del gap](reports/control_gap_sin_assigned.md).
- [Evaluación del holdout](reports/evaluacion_final_histgb.md).
- [Curva de aprendizaje](reports/curva_aprendizaje.md) y [adquisición de datos](reports/adquisicion_datos.md).
- [Auditoría adicional](output/revision_modelos_20261116/informe.md).

El preprocesamiento y el umbral se ajustan dentro del entrenamiento de cada partición. Target, estado y fecha de estado se excluyen de las entradas. La elección histórica de hiperparámetros no fue completamente anidada. El holdout ya fue consultado y no es una evaluación independiente nueva para variantes posteriores. El modelo presenta caídas ante cambios de distribución; el resultado local no garantiza la puntuación de competencia.
