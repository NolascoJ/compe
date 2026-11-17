# Predictor HistGB · Grupo 9

Archivo para inferencia: **predictor_v2_Grupo_9.py**. Incluye el preprocesamiento y un HistGradientBoosting entrenado con las 38.274 filas de train. No reentrena al predecir y no necesita el repositorio ni archivos de modelo externos.

Umbral fijo: **0,38294747099493937**. La evaluación previa del holdout interno obtuvo BA 85,09%; no predice el rendimiento sobre el conjunto oculto de la competencia.

## Uso

Con el .py en la misma carpeta:

```python
import pandas as pd
from predictor_v2_Grupo_9 import predictor_v2_Grupo_9

X = pd.read_csv("reservas_activas.csv")
y = predictor_v2_Grupo_9(X)
```

`y` es un ndarray 0/1, con una etiqueta por fila en el orden original. `predictor(X)` es un alias; `probabilidades_v2_Grupo_9(X)` devuelve probabilidades de cancelación. Las columnas extras, incluido el target y las variables de estado, se ignoran.

## Dependencias

Las versiones del artefacto están en `requirements.txt`. Instalarlas antes de importar el modelo:

```sh
python -m pip install -r entrega/requirements.txt
```

También se pueden instalar desde el propio archivo:

```sh
python predictor_v2_Grupo_9.py --instalar-dependencias
```

En Colab, anteponer `!` al comando y reiniciar la sesión tras instalar. La instalación depende de la disponibilidad de esas versiones para el Python del runtime. Importar el predictor no instala paquetes.

El entorno local usa Python 3.14.2 y sklearn 1.9.1. La prueba con sklearn 1.6.1 en Colab produjo predicciones, pero emitió avisos de incompatibilidad. Falta validar el entorno definitivo; no ocultar esos avisos ni asumir portabilidad del pickle entre versiones.

## Esquema de entrada

`COLUMNAS_REQUERIDAS` enumera las entradas activas. Se usan los nombres originales, incluidos `resource_modifications`, `queue_wait_time`, `total_gpu_hours`, `total_processes` y `high_value_client`. Las entradas faltantes generan un error; no se sustituyen por ceros.

`assigned_gpu_type` y `execution_week` no son obligatorias. El wrapper agrega un valor auxiliar de assigned que el preprocesamiento descarta, para compatibilidad con la transformación histórica. No entra al estimador. Las categorías nuevas se asignan a `otros` usando el vocabulario aprendido.

## Herramientas

Desde la raíz del proyecto:

```sh
OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python entrega/verificar_predictor.py
.venv/bin/python entrega/probar_predictor.py entrega/dataset_prueba_original_300.csv /tmp/predicciones_histgb.csv
```

El ejemplo contiene 100 filas reales de train repetidas tres veces; sirve para probar carga y salida, no para medir generalización. `predicciones_esperadas_prueba_300.csv` permite comparar la salida.

- `construir_predictor.py`: empaqueta el artefacto existente, sin entrenar.
- `verificar_predictor.py`: compara scores con el pipeline original y comprueba carga aislada, orden e invariancia del lote.
- `probar_predictor.py`: guarda probabilidades y etiquetas desde un CSV, sin sobrescribir una salida existente.
- `manifiesto.json`: origen, hashes, columnas y versiones.
- `verificacion.json`: resultado de la verificación local.
