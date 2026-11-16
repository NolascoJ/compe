# Calibración interna del punto de corte y robustez

`calibracion.py` evalúa las configuraciones **fijas** de `parametros.json`. No elige modelos ni busca hiperparámetros, y nunca lee `data/raw/test.csv`. Reemplaza la parte de calibración y robustez del borrador `analisis.py`, que queda sin tocar (sus caches `logistic_*.npz` no se reutilizan).

## Qué calcula

**Parte 1 · Punto de corte sin usar el fold evaluado** (los 8 modelos con `assigned`; los finalistas también sin `assigned`). Para cada uno de los 5 folds externos de `make_folds(df, seed=42)`, que son los mismos del protocolo original:
1. dentro del entrenamiento externo, 3 folds internos agrupados (`make_folds(train_externo.reset_index(drop=True), n_splits=3, seed=42)`) y sus predicciones fuera de fold;
2. el punto de corte se elige con `best_threshold` sobre esos OOF internos;
3. se reentrena con todo el entrenamiento externo, se puntúa el fold externo y se le aplica ese punto de corte.

Además se calculan los OOF convencionales: son las mismas puntuaciones del paso 3, con un único punto de corte optimizado sobre todos los OOF. La caída que se informa es `BA con punto de corte global − BA con puntos de corte internos`, con su signo (puede salir negativa).

Esto calibra el **punto de corte**, no las probabilidades. Saca el optimismo de elegir el punto de corte mirando el fold evaluado, pero **no** el de haber elegido los hiperparámetros sobre este mismo train: no es una validación anidada de la selección de hiperparámetros.

**Parte 2 · Robustez** (finalistas provisionales `logistic`, `gradient_boosting` y `hist_gradient_boosting`, con y sin `assigned`). En cada escenario, el punto de corte sale de OOF internos de 3 folds hechos **solo** con el entrenamiento de ese escenario:

| Escenario | Entrena con | Evalúa en |
|---|---|---|
| `eval_anio_2015/2016/2017` | los otros dos años | el año excluido |
| `eval_PRT` | todo menos `PRT` | `PRT` |
| `eval_resto` | solo `PRT` | el resto |
| `sin_locked` | 5 folds agrupados armados sobre las filas sin `locked_allocation` | ídem |

Se usa un único punto de corte por entrenamiento, nunca uno por subgrupo. Los valores `PRT` y año definen evaluaciones, no reglas ni variables.

**Agrupamiento.** Los grupos son siempre los de `make_folds` sobre las variables originales, también en la variante sin `assigned`, para que todo sea comparable. No se recalculan equivalencias sobre las variables transformadas.

## Comandos

Desde la raíz del repo:

```sh
cd /Users/jnolascodecarles/ITBA/4_1Q/ml/compe

# 1. Pruebas cortas (~15 s, 9 pruebas)
PYTHONPATH=src .venv/bin/python experiments/seleccion/test_calibracion.py

# 2. Ver qué se va a correr, qué ya está y cuánto tarda (no entrena)
PYTHONPATH=src .venv/bin/python experiments/seleccion/calibracion.py todo --dry-run

# 3. Corrida completa en segundo plano (Parte 1 + Parte 2 + resumen)
PYTHONPATH=src nohup caffeinate -i .venv/bin/python experiments/seleccion/calibracion.py todo --jobs 4 \
  > experiments/seleccion/calibracion/nohup.out 2>&1 &

# Seguimiento: termina con "COMPLETO"
tail -f experiments/seleccion/calibracion/calibracion.log
```

Por etapas, si se quiere correr solo una parte:

```sh
PYTHONPATH=src .venv/bin/python experiments/seleccion/calibracion.py oof            # solo OOF convencionales
PYTHONPATH=src .venv/bin/python experiments/seleccion/calibracion.py calibracion    # Parte 1 (incluye los OOF)
PYTHONPATH=src .venv/bin/python experiments/seleccion/calibracion.py robustez       # Parte 2, finalistas, con y sin assigned
PYTHONPATH=src .venv/bin/python experiments/seleccion/calibracion.py resumen        # arma los CSV con lo que haya completo
```

Opciones:
- `--modelos logistic ridge ...`: elegir modelos.
- `--assigned auto|con|sin|ambos`: por defecto `auto`, que hace lo del plan.
- `--jobs N`: tareas en paralelo como máximo. Los modelos que ya usan todos los núcleos (knn, random_forest, bagging, hist_gradient_boosting) van de a uno.
- `--dry-run`: muestra el plan sin entrenar.

## Reanudación y caches

- Cada tarea (modelo × variante × escenario × fold × tipo) se guarda en `calibracion/cache/<clave>.npz` con su manifiesto `<clave>.json`. Ambos se escriben de forma atómica y el manifiesto va al final.
- Si la corrida se corta, relanzar el **mismo comando** sigue desde lo que falta. Si alguna tarea falla, el error queda en el log, el resto sigue y al relanzar se reintentan solo las que fallaron.
- Un cache se reutiliza solo si su manifiesto coincide en todo: parámetros, variante, tipo de puntuación, semilla, cantidad de folds, hash de los índices de entrenamiento y evaluación, hash de `train.csv` (verificado contra `split_metadata.json`), hash del código relevante (`src/compe/{config,data,features,pipeline,models,validation}.py` + `VERSION_CALCULO`) y versiones de las librerías. Además se comprueba que las puntuaciones sean finitas y del largo correcto.
- Un cache incompatible no se borra: se mueve a `calibracion/cache/descartados/` y se recalcula.

## Tiempo estimado

`--dry-run` estima ~18 minutos para todo (230 tareas, 460 ajustes, `--jobs 4`). La estimación sale de los `fit_seconds` de los experimentos, escalados por la cantidad de filas, y **no incluye el tiempo de predecir**.

Estimación realista: **20 a 45 minutos**.
- `gradient_boosting` es lo más pesado, con ~11 minutos estimados entre las dos variantes y la robustez.
- La mayor incertidumbre está en `knn`: predecir con 301 vecinos sobre ~7.600 filas puede tardar bastante más que ajustar.

Como referencia, la prueba real de los 10 OOF de `logistic` tardó 20 segundos con `--jobs 5`.

## Salidas (`calibracion/resultados/`)

| Archivo | Contenido |
|---|---|
| `calibracion_folds.csv` | por modelo × variante × fold: filas y clases, punto de corte interno, BA de los OOF internos (optimista dentro del entrenamiento), BA del fold con el punto de corte interno y con el global |
| `calibracion_resumen.csv` | por modelo × variante: BA con punto de corte global (convencional), BA con puntos de corte internos (global, media y desvío entre folds), caída con signo, ROC-AUC y rango de los puntos de corte. Cuando la configuración es idéntica a la del experimento, incluye `diferencia_vs_experimento` (control de reproducibilidad: debería dar 0) |
| `subgrupos_calibrados.csv` | BA por subgrupo (`todo`, `sin_locked_allocation`, `PRT`, resto, años) y `no_show` detectados, con puntos de corte internos y con el global |
| `robustez_calibrada.csv` | por escenario: filas, clases, BA (NA si el conjunto evaluado tiene una sola clase), caída respecto de la CV calibrada de la **misma** configuración y variante, y BA de esas mismas filas en la CV normal. En `sin_locked`, esa última columna es el modelo entrenado con todas las filas |
| `predicciones_externas.csv` | una fila por observación evaluada × modelo × variante × escenario (ver abajo) |
| `matriz_confusion_global.csv` | TN, FP, FN, TP y n por modelo × `assigned` × escenario |
| `metricas_externas_globales.csv` | sensibilidad, especificidad, BA, accuracy, precision, F1 y ROC-AUC globales, más media y desvío entre folds (`_std_folds_ddof0`) |
| `metricas_externas_por_fold.csv` | las mismas métricas y la matriz, por fold externo |
| `manifiesto_resultados.json` | hashes, versiones, parámetros y la lista de tareas usadas |

### `predicciones_externas.csv`

Columnas obligatorias:
- `modelo`, `assigned` (True/False) y `escenario`: `cv` para la CV normal; `sin_locked` para la CV entrenada sin `locked_allocation`; `eval_anio_*`, `eval_PRT` y `eval_resto` para la robustez.
- `fold_externo`: vacío en los escenarios de una sola evaluación (año o región).
- `fila_train`: posición original en `train.csv`, desde cero. Se conserva también al filtrar y resetear índices.
- `y_true`, `score`, `y_pred` y `umbral`: el punto de corte interno de ese fold, con `y_pred = score >= umbral`.

Columnas extra, solo de control y solo en `cv`: `variante`, `umbral_global_oof` y `y_pred_umbral_global` (el método convencional).

Cobertura, verificada al escribir el archivo:
- en `cv`, cada fila de train aparece exactamente una vez por modelo y variante;
- en `sin_locked`, cada fila del subconjunto aparece exactamente una vez;
- en los escenarios de año y región, cada fila del grupo evaluado aparece una vez.

Una misma observación puede aparecer en varios escenarios; el escenario los distingue.

### Métricas desde las predicciones

`resumen` calcula la matriz y las métricas **solo** desde `predicciones_externas.csv`, usando el `y_pred` guardado. Cada fold tiene su punto de corte interno y no se vuelve a optimizar ninguno.

Convenciones:
- La clase positiva es `canceled_job = 1`.
- Las métricas globales salen de concatenar las predicciones, no de promediar matrices ni métricas por fold.
- La media y el desvío entre folds usan `ddof=0`, igual que `validation.py`.
- Si un denominador es cero, la métrica queda NA. Por ejemplo, la precision si no se predijo ningún positivo, o la BA si falta una de las dos clases reales.
- El ROC-AUC global mezcla puntuaciones de modelos distintos (uno por fold), por eso también se informa por fold.
- Se verifica que TN + FP + FN + TP coincida con la cantidad de observaciones de cada grupo.

Para regenerar la matriz y las métricas sin cargar train ni ajustar modelos (solo lee el CSV):

```sh
.venv/bin/python experiments/seleccion/metricas_externas.py \
  experiments/seleccion/calibracion/resultados/predicciones_externas.csv \
  experiments/seleccion/calibracion/resultados
```

El resumen se verifica solo: la BA de la matriz coincide con la de `calibracion_resumen.csv` y con la de `robustez_calibrada.csv`.

## Verificaciones (`test_calibracion.py`)

Se corren sobre una submuestra de 3.000 filas de train:
- cada fila se evalúa exactamente una vez;
- entrenamiento y evaluación nunca se solapan;
- los grupos de `feature_groups` quedan disjuntos en las particiones externas, internas y de robustez;
- los índices internos son relativos al entrenamiento externo;
- un estimador espía confirma que el fold externo nunca aparece en un ajuste interno ni en el preprocesamiento;
- cambiar etiquetas y variables del fold externo no cambia el punto de corte ni las puntuaciones de las otras filas;
- sin `assigned`, las puntuaciones no cambian al barajar `assigned_gpu_type` (logistic, gradient_boosting y hist_gradient_boosting); con `assigned`, sí cambian;
- las salidas son finitas y del largo correcto, y la BA recalculada coincide con los CSV;
- la cobertura de `predicciones_externas.csv` es exacta (`cv` y `sin_locked`) y `y_pred = score >= umbral`;
- la matriz de confusión suma n, y las métricas a mano (incluidos los denominadores en cero → NA) dan lo esperado;
- `metricas_externas.py` corrido solo desde el CSV reproduce exactamente los archivos de `resumen`, y rechaza predicciones inconsistentes;
- los caches con otros parámetros, otro hash de train, otros índices o sin manifiesto se rechazan;
- en paralelo da lo mismo que en secuencia.

Prueba real adicional: los OOF convencionales de `logistic` (misma configuración que el experimento) reproducen exactamente `metricas.json`, con y sin `assigned` (0,82713 y 0,81972).

## Para tener en cuenta

- **Las configuraciones de `parametros.json` las propuso el borrador y no son las de `experiments/<modelo>/mejores_parametros.json` en 7 de los 8 modelos.** La única que coincide es `logistic`. El motivo de cada cambio está en `seleccion_configuraciones.csv`. Por eso el control de reproducibilidad solo aplica a `logistic`.
- Las advertencias de cada ajuste (por ejemplo, convergencia) quedan registradas en el log y en el manifiesto de cada tarea; no se ocultan. `logistic` emite un `FutureWarning` de scikit-learn por el parámetro `penalty`, que es inofensivo.
- El `.venv` usa Python 3.14 y scikit-learn 1.9, no las versiones de Colab. Para este análisis no importa; para el predictor sí.
