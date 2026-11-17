# Predictor HistGB Grupo 9

El archivo que se comparte es **predictor_v2_Grupo_9.py**. Contiene preprocesamiento y modelo entrenado; no necesita el repo ni archivos auxiliares, no entrena ni accede a internet. Dependencias en requirements.txt (no usa CatBoost).

Este artefacto conserva el modelo evaluado, **entrenado con las 38.274 filas de train** y umbral 0.38294747099493937. No es un nuevo ajuste con train+test. Su evaluación existente de holdout es 85,09% BA; esa cifra no se midió otra vez durante el empaquetado y no garantiza el test oculto.

## Prueba local

Desde la raíz del repo, usando el entorno existente:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python entrega/verificar_predictor.py
.venv/bin/python entrega/probar_predictor.py data/raw/train.csv /tmp/predicciones_histgb.csv
```

El segundo comando genera predicciones; predecir train no es una evaluación independiente del rendimiento. Si la salida ya existe, elegir otro nombre.

También se puede usar desde Python:

```python
import pandas as pd
from predictor_v2_Grupo_9 import predictor_v2_Grupo_9
X = pd.read_csv('datos.csv')
y = predictor_v2_Grupo_9(X)
```

Devuelve ndarray de enteros 0/1 en orden original. 1=cancelación. `predictor(X)` es un alias con la misma salida. `probabilidades_v2_Grupo_9(X)` devuelve probabilidades de clase 1.

## Colab

Subir predictor_v2_Grupo_9.py y ejecutar en un entorno compatible con sus dependencias:

```python
%pip install numpy==2.5.4 pandas==3.1.1 scipy==1.18.1 scikit-learn==1.9.1 joblib==1.6.0 threadpoolctl==3.8.0
```

Ver requirements.txt para las versiones exactas del empaquetado; si difieren del ejemplo, usar las de ese archivo. Reiniciar el runtime después de instalar si ya había importado las librerías. La disponibilidad de esas versiones depende del Python del runtime. **La ejecución real en Colab está pendiente**; no asumir compatibilidad de un pickle entre versiones de sklearn. El entorno verificado localmente usa Python 3.14.2.

## Entradas

Recibe las columnas originales como DataFrame. COLUMNAS_REQUERIDAS dentro del .py enumera las entradas activas. Ignora columnas extras y las etiquetas/columnas con fuga. Assigned y execution_week no son obligatorias. Las regiones nuevas se mapean a otros; conserva las reglas aprendidas de preprocesamiento.

No acepta silenciosamente la ausencia de queue_wait_time, resource_modifications, high_value_client, total_gpu_hours, total_processes u otra entrada activa. Si el evaluador solo proporciona las 25 entradas de la primera entrega, hay que resolver esa diferencia antes del envío. No se inventan features faltantes ni se derivan con reglas no verificadas.

El objeto original descartaba assigned exigiendo que existiera. El wrapper selecciona entradas en orden de entrenamiento y agrega una columna assigned constante que el preprocesamiento descarta: no entra al estimador. Esta corrección se verificó comparando exactamente los scores con los del objeto original. No se modificaron estadísticas aprendidas ni hiperparámetros.

## Archivos

- predictor_v2_Grupo_9.py: artefacto autocontenido para compartir.
- probar_predictor.py: CLI para leer CSV y guardar probabilidades/predicciones.
- construir_predictor.py: reconstrucción reproducible desde el artefacto original, sin reentrenar.
- verificar_predictor.py y verificacion.json: pruebas de equivalencia e independencia del repo, usando train solamente.
- manifiesto.json: hashes del archivo y modelo fuente, columnas, versiones y alcance del entrenamiento.

No cargar archivos pickle/.py de fuentes desconocidas. El checksum interno comprueba integridad del payload, no prueba seguridad de un archivo recibido de terceros.

## Instalar dependencias desde el mismo archivo

El .py incorpora las versiones exactas y una opción explícita de instalación, ejecutada antes de importar numpy/pandas/sklearn:

```python
# Celda de Colab, después de subir el archivo actualizado:
!python predictor_v2_Grupo_9.py --instalar-dependencias
```

Si pip termina sin errores, reiniciar la sesión de ejecución y después importar el predictor y probarlo. Si las versiones no están disponibles para el Python de Colab, la instalación fallará de forma visible; no usar otra versión de sklearn silenciosamente. Importar el predictor o llamar a predictor(X) no instala paquetes. No se verificó aún la instalación de estas versiones en el runtime de la cátedra.
