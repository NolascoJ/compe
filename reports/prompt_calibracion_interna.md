# Implementar y verificar la evaluación del umbral; el usuario ejecuta

Trabajás en `/Users/jnolascodecarles/ITBA/4_1Q/ml/compe`. Tu tarea es preparar código reproducible para evaluar configuraciones FIJAS mediante calibración interna del umbral y pruebas de robustez. **No elijas el ganador ni lances corridas largas**: el usuario ejecutará los comandos y otro modelo interpretará los resultados después.

## Contexto y estado de la entrega

Leé `CLAUDE.md`, `reports/prompt_seleccion_modelo.md` (ya corregido), `src/compe/{data,validation,pipeline,models}.py` y `experiments/seleccion/analisis.py`.

Hay un borrador de implementación en `experiments/seleccion/analisis.py`. NO lo consideres terminado ni validado. Se interrumpió voluntariamente a pedido del usuario. Se completaron los OOF convencionales de logística con y sin assigned (`logistic_True.*`, `logistic_False.*`); no se completó la calibración interna. El KeyboardInterrupt del log corresponde a esa interrupción.

`experiments/seleccion/parametros.json` contiene las ocho configuraciones provisionales que debés mantener fijas. `seleccion_configuraciones.csv` documenta por qué se propusieron. No hagas nuevas selecciones ni búsquedas. Los finalistas provisionales para robustez son `logistic`, `gradient_boosting` y `hist_gradient_boosting`. No son una decisión definitiva.

El borrador actual mezcla selección, referencias, controles y calibración en una sola ejecución. Separá la ejecución por etapas para que el usuario pueda correr solo lo necesario. No ejecutes ni importes ese script directamente para inspeccionarlo: actualmente tiene trabajo de entrenamiento en el nivel superior.

## Restricciones

- Nunca abrir, cargar ni usar `data/raw/test.csv`, tampoco indirectamente.
- No modificar `experiments/<modelo>/`, módulos existentes de `src/compe/`, ni los datos originales.
- Escribí código y resultados únicamente en `experiments/seleccion/`; podés crear un README allí.
- `execution_status`, `execution_status_date` y `canceled_job` no son features. La etiqueta solo se usa para entrenar/evaluar; execution_status únicamente para desgloses de no_show.
- Usar `.venv/bin/python`, con `PYTHONPATH=src`.
- Pipeline original: `make_pipeline(MODELS[nombre].build(params), MODELS[nombre].variante, use_assigned=...)`.
- Preprocesamiento aprendido exclusivamente dentro de cada entrenamiento, también en folds internos.
- No instalar paquetes, cambiar versiones, agregar features ni correr XGBoost/LightGBM.
- No suprimir indiscriminadamente warnings: registrar advertencias de convergencia y errores.

## Parte 1: evaluación independiente del umbral

Para los ocho modelos, con assigned; para los tres finalistas, también sin assigned:

1. Cargar únicamente train mediante `load_train()` y conservar su orden.
2. Crear los 5 folds externos con `make_folds(df, seed=42)`.
3. En cada entrenamiento externo, crear **3 folds internos agrupados** con `make_folds(train_externo.reset_index(drop=True), n_splits=3, seed=42)`.
4. Generar predicciones OOF internas entrenando el pipeline completo en cada entrenamiento interno. Usar `model_scores` y el tipo de puntuación de MODELS.
5. Elegir el umbral con `best_threshold(y_entrenamiento_externo, oof_internos)`.
6. Reentrenar un pipeline nuevo con todo el entrenamiento externo, predecir el fold externo y aplicar el umbral interno. El fold externo no interviene en ningún paso previo.
7. Repetir cinco veces. Guardar puntuaciones, predicciones, índices de filas externas y umbral de cada fold; BA por fold, desvío y BA global sobre las predicciones concatenadas en orden original. BA global no equivale necesariamente al promedio de BA de folds.

Esto calibra el **umbral**, no las probabilidades. No usar Platt ni isotónica. No rotar umbrales sobre los OOF globales existentes. La evaluación elimina el uso directo/indirecto del fold externo para ajustar su umbral, pero NO elimina el sesgo de haber elegido hiperparámetros previamente sobre train. No llamarla validación anidada completa de selección de hiperparámetros.

Para cada configuración/variante, obtener también OOF convencionales con los mismos 5 folds (reusar solo caches compatibles y verificados). Comparar BA con umbral optimizado globalmente contra BA con umbrales internos. La primera es levemente optimista. Guardar caída firmada: BA convencional menos BA con umbrales internos; no forzarla a ser positiva. No comparar una configuración nueva con la BA de otra guardada en experiments/<modelo>/metricas.json como si fueran la misma.

## Parte 2: robustez con hiperparámetros fijos

Solo para los tres finalistas provisionales, con y sin assigned:

- Dejar afuera cada año: entrenar con los otros dos y evaluar el excluido.
- Entrenar sin PRT y evaluar PRT; entrenar con PRT y evaluar el resto. Estos valores definen evaluaciones, nunca reglas predictivas ni features nuevas.
- En cada caso, elegir umbral con OOF internos de 3 folds agrupados SOLO en el entrenamiento de ese escenario. Luego ajustar con todo ese entrenamiento y evaluar el grupo excluido.
- Reportar BA en el subgrupo sin locked_allocation de las predicciones externas de Parte 1.
- Además, filtrar train para excluir locked_allocation, crear allí 5 folds externos agrupados y repetir calibración interna de 3 folds. Así se mide un modelo entrenado y evaluado sin esas filas, sin contaminar la evaluación.
- Usar umbral global por entrenamiento; no umbrales por subgrupo.
- Para cada escenario reportar filas, cantidades por clase, BA y caída respecto de CV normal con calibración interna de la MISMA configuración y variante. Para el entrenamiento sin locked comparar además contra BA en ese subgrupo del modelo entrenado con todas las filas.
- Si un conjunto de evaluación tiene una sola clase, BA queda indefinida: reportar NA y cantidades, no inventar un valor.

## Parte 3: ejecución segura, reanudación y verificaciones

Podés refactorizar el borrador o crear `experiments/seleccion/calibracion.py`. Preferí funciones pequeñas y CLI con:

- etapas `oof`, `calibracion`, `robustez`, `resumen`;
- selección de modelos y variante con/sin assigned;
- modo `--dry-run` que liste ajustes estimados y archivos sin entrenar;
- reanudación por modelo, variante, escenario y fold;
- límite de paralelismo para evitar saturar CPU/memoria;
- logs con progreso, duración y warnings; escrituras atómicas por resultado completo.

Un archivo existente NO basta para reutilizarlo: validar parámetros, variante assigned, semillas, número de folds, índices externos, hash de train y versiones del código/librerías. Guardar un manifiesto con esa información. Los caches del borrador no tienen ese manifiesto: no reutilizarlos silenciosamente; se pueden conservar aparte y recomputar. No borrar archivos del usuario.

Verificar con pruebas pequeñas/sintéticas:

- cobertura externa exactamente una vez por fila y sin solapamiento train/evaluación;
- grupos de `feature_groups` disjuntos en cada partición externa e interna;
- datos de validación externa jamás usados para aprender preprocesamiento ni umbral;
- índices relativos internos aplicados al entrenamiento externo, no al dataframe global;
- sin assigned realmente excluye esa variable y sus derivados a través del pipeline;
- todos los outputs finitos y del largo correcto; distinguir score de predicción binaria;
- BA recalculada desde predicciones coincide con CSV; reanudación rechaza caches incompatibles.

Mantener los grupos de `make_folds` del protocolo original también en variante sin assigned, para comparabilidad. Documentar que se agrupa por features originales, no recalculando equivalencias de features transformadas.

No modificar referencias/búsquedas ni implementar ensembles en esta tarea. No extenderla a selección de ganador.

## Entregables

1. Código listo, revisado y probado con datos sintéticos; ningún entrenamiento largo ejecutado.
2. README en `experiments/seleccion/` con comandos exactos para las etapas, cómo reanudar, estimación de ajustes y tiempo basada en los tiempos disponibles (aclarar incertidumbre).
3. CSV finales previstos: `calibracion_folds.csv`, `calibracion_resumen.csv`, `robustez_calibrada.csv`, `subgrupos_calibrados.csv`; scores/predicciones por fila y manifiestos para auditoría.
4. Al terminar, explicar qué comando debe correr el usuario y qué archivos debe devolver al analista. No anunciar resultados que todavía no se midieron ni recomendar ganador.

El usuario autoriza preparar el código y pruebas cortas. Él correrá la calibración y robustez. Si detectás un problema real de datos o fuga, reportalo con evidencia antes de continuar la parte afectada.

## Requisito adicional: conservar todas las predicciones externas

Además de media y desviación estándar, entregar `predicciones_externas.csv` con una fila por observación externa, modelo y variante. Columnas obligatorias:

- `modelo`, `assigned`, `escenario` (CV normal o CV entrenada sin locked), `fold_externo`;
- `fila_train`: posición original, base cero, en train.csv; conservarla también al filtrar y resetear índices;
- `y_true`, `score`, `y_pred`, `umbral` aplicado en ese fold.

Para cada modelo/variante de CV normal debe haber exactamente tantas filas como train, sin duplicados ni omisiones de `fila_train`. En CV sin locked, cubrir exactamente una vez cada fila del subconjunto. Conservar también las predicciones por fila de las evaluaciones por año/región, identificando escenario para no mezclar evaluaciones repetidas de una observación.

Desde estas predicciones ya guardadas, la etapa `resumen` debe calcular SIN entrenar:

- matriz de confusión global: TN, FP, FN y TP, con clase positiva canceled_job=1;
- recall/sensibilidad = TP/(TP+FN);
- especificidad = TN/(TN+FP);
- balanced accuracy = (sensibilidad + especificidad)/2;
- accuracy, precision y F1;
- métricas por fold, media y desviación estándar (declarar ddof; usar ddof=0 para comparar con validation.py).

La matriz y las métricas globales se calculan concatenando las predicciones externas, no promediando matrices normalizadas ni métricas por fold. Usar `y_pred` guardado: cada fold tiene su propio umbral interno. NO volver a optimizar un umbral sobre las predicciones externas para estas métricas. Si un denominador es cero, registrar NA y explicar la convención. Se puede incluir ROC-AUC desde `score`, aclarando que el AUC global combina puntuaciones de distintos modelos ajustados por fold; acompañarlo del AUC por fold.

Guardar la matriz en `matriz_confusion_global.csv` y las métricas en `metricas_externas_globales.csv`. El README debe incluir un comando para regenerar ambos únicamente a partir de los archivos de predicciones, sin cargar ni ajustar modelos. Verificar que TN+FP+FN+TP coincide con la cantidad de observaciones evaluadas por modelo/variante/escenario.
