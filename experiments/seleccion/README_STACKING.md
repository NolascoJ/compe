# Comparación de HistGB, promedio y stacking estricto

El script `stacking.py` conserva las tres alternativas y sus predicciones. **No selecciona un ganador, no abre test.csv, no produce el predictor definitivo y no busca hiperparámetros.** El equipo decidirá cuando estén todas las opciones; recién después se congela una opción y se evalúa una vez en test.

Las tres estrategias usan únicamente datos de train y excluyen assigned_gpu_type/gpu_changed:

1. HistGradientBoosting individual, default del repo.
2. Promedio aritmético, pesos 1/3, de probabilidades GB, HistGB y RF.
3. Regresión logística sobre esas tres probabilidades, L2 por defecto, C=1, solver=lbfgs, max_iter=1000, class_weight=None, intercepto incluido. No usa features originales, transformaciones de scores ni búsqueda de pesos.

Las configuraciones base se leen de `parametros.json` y se congelan en el manifiesto. Las columnas siempre son GB, HistGB, RF, con score de clase positiva cancelado=1.

## Ejecutar

Desde la raíz del repositorio:

```sh
cd /Users/jnolascodecarles/ITBA/4_1Q/ml/compe

# Verificar el plan sin entrenar ni escribir resultados:
PYTHONPATH=src .venv/bin/python experiments/seleccion/stacking.py todo --dry-run

# Ejecutar CV y todas las pruebas de robustez en segundo plano (macOS):
PYTHONPATH=src nohup caffeinate -i .venv/bin/python -u experiments/seleccion/stacking.py todo --jobs 2 > experiments/seleccion/stacking_ejecucion.log 2>&1 &

# Ver progreso; Ctrl+C sale del seguimiento, no detiene la corrida:
tail -f experiments/seleccion/stacking_ejecucion.log
```

El log termina con `COMPLETO` cuando finalizan todas las tareas solicitadas y los resúmenes. Si aparece un traceback o se interrumpe, revisar el problema y repetir el mismo comando: retoma checkpoints compatibles. No lanzar dos instancias simultáneas contra la misma salida. El log se sobrescribe al relanzar; resultados y checkpoints se conservan.

Se puede ejecutar en primer plano omitiendo `nohup`, la redirección y `&`. Se puede usar `--jobs 1` para menor consumo. Cada worker limita las librerías numéricas y los estimadores base a un hilo; `--jobs` controla el paralelismo entre folds/escenarios.

Etapas disponibles:

```sh
PYTHONPATH=src .venv/bin/python experiments/seleccion/stacking.py cv --jobs 2
PYTHONPATH=src .venv/bin/python experiments/seleccion/stacking.py robustez --jobs 2
PYTHONPATH=src .venv/bin/python experiments/seleccion/stacking.py resumen
```

`resumen` no entrena. Puede ejecutarse después de una interrupción para inspeccionar lo completado. Las métricas agregadas de un escenario incompleto están marcadas `escenario_completo=False`; no usarlas como resultados finales. El filtro de comparación CV solo se genera con los cinco folds completos.

Tiempo orientativo: CV 15–30 minutos; CV más robustez 45–90 minutos. No es un límite ni una garantía; depende de CPU, caché y paralelismo. Cada tarea totalmente nueva requiere 39 ajustes base y cuatro logísticas; reutilizar resultados reduce ese total. En la primera CV, con los caches actuales válidos de GB/HistGB, faltan aproximadamente 155 ajustes base (45 GB, 45 HistGB y 65 RF). Todos los ajustes profundos usan subconjuntos más chicos que un entrenamiento externo normal. El dry-run indica tareas externas pendientes, no un inventario exacto de todos los checkpoints profundos.

## Aislamiento estricto del stacking y del umbral

En cada entrenamiento externo T y evaluación externa V:

1. Crear Z(T), tres columnas de OOF de las bases con 3 folds internos agrupados. Cada bloque se predice con bases entrenadas en los otros dos folds de T.
2. Para cada partición interna A/B de T, generar **nuevos OOF de las bases únicamente dentro de A**, con 3 subfolds agrupados. Ajustar una logística con estos OOF y las etiquetas de A. Predecir B aplicando esa logística a las predicciones de bases entrenadas en A (el bloque correspondiente de Z(T)).
3. Concatenar las predicciones del stacking de B. Elegir el umbral del stacking con esas predicciones y las etiquetas de T. Los umbrales de HistGB y promedio se eligen usando Z(T).
4. Ajustar una nueva logística sobre todo Z(T) y etiquetas de T. Aplicarla a las probabilidades externas de bases ajustadas en todo T. Aplicar los umbrales elegidos en T a las tres estrategias sobre V.

En particular, **no se hace CV de la logística reutilizando Z(T)[A] como su entrenamiento**: esas filas podrían haber sido producidas por bases que vieron B. Se regenera Z(A), que es la diferencia esencial respecto de la aproximación descartada.

Se usa el pipeline original de cada modelo y todo su preprocesamiento se aprende dentro de cada ajuste. Se verifican grupos disjuntos en todos los niveles. Los grupos se construyen con features originales según el protocolo existente, incluyendo assigned para la definición de equivalencias, aunque esa variable no entre al predictor. No se agrupa por features transformadas.

No elimina el sesgo histórico de selección de hiperparámetros ni la selección posterior entre estrategias usando los folds ya observados. Se corrige el aislamiento del entrenamiento/calibración del ensemble, manteniendo configuraciones fijas. No se garantiza ausencia de sobreajuste ni generalización bajo cualquier cambio de distribución.

## Robustez

Se ejecutan las tres estrategias, sin elegir previamente una ganadora:

- Entrenar con otros años, evaluar cada año excluido.
- Entrenar sin PRT y evaluar PRT; entrenar con PRT y evaluar el resto.
- Excluir locked_allocation antes de crear cinco folds externos y repetir todo el protocolo.
- Desgloses de todas las predicciones por año, región y sin locked, con umbrales fijos por entrenamiento.

Cada escenario tiene calibración estricta interna propia. Las OOF de la CV normal no se usan como sustituto de estas pruebas. Las filas de un escenario nunca son entrenamiento de ese mismo escenario. La comparación contra CV en las mismas filas se conserva separadamente.

## Reutilización y reanudación

- Los caches anteriores de calibracion se leen, nunca se modifican. Se valida su manifiesto mediante `calibracion.estado`: parámetros, assigned, versiones, hash de train/código e índices. Para OOF se comprueba además la identidad de los folds internos antes de reutilizar bloques.
- Todo resultado nuevo queda en `stacking_resultados/<id_corrida>/`. El identificador depende del hash de train, parámetros, versiones, semillas y código. Un cambio genera otra carpeta, conservando los resultados previos.
- Nuevos checkpoints por ajuste base, OOF y resultado externo. Incluyen índices y manifiestos, checksum SHA-256 del NPZ, origen y warnings. Las escrituras son atómicas; un ajuste interrumpido se repite, los completos se reutilizan.
- Un checkpoint nuevo corrupto provoca error explícito; no se acepta ni borra silenciosamente.
- Se guardan warnings de las bases. Un warning de la logística del metamodelo detiene esa tarea para revisar la convergencia, conservando los ajustes base.
- Cada resultado externo guarda también los scores internos usados para elegir el umbral, las probabilidades externas de las bases y los coeficientes de la logística final.

## Resultados que hay que conservar y devolver al analista

La carpeta `stacking_resultados/<id_corrida>/resultados/` contiene:

- `estado.csv`: completitud de los 15 folds/escenarios externos (5 CV + 5 año/región + 5 sin locked).
- `predicciones_externas.csv`: índice original en train, etiqueta, estrategia, escenario, fold, score, umbral y clase predicha. Permite recalcular métricas sin entrenar.
- `matriz_confusion_global.csv`, `metricas_externas_globales.csv`, `metricas_externas_por_fold.csv`: BA, recall, especificidad, precision, F1, accuracy y AUC; media y desvío ddof=0, matrices con clase positiva 1. BA global se calcula concatenando filas, no promediando BA de folds.
- `comparacion_cv.csv`: mejora media/global respecto de HistGB, folds ganados e intervalo bootstrap pareado por grupos (2.000 réplicas). El intervalo es condicional a los ajustes existentes, no incorpora volver a entrenar/seleccionar.
- `diferencias_folds.csv`: diferencias pareadas por fold.
- `comparacion_escenarios.csv`: diferencia contra HistGB y caída contra CV en las mismas filas.
- `subgrupos.csv`: métricas por subgrupo sin reajustar umbrales.
- `coeficientes_meta.csv`: pesos e intercepto de la logística por entrenamiento, como diagnóstico de estabilidad.

El flag `pasa_filtro_cv` aplica exactamente la heurística propuesta: al menos 4/5 folds mejores y mejora media >0,003. **No declara un ganador.** La elección final queda pendiente del análisis conjunto de incertidumbre, robustez y simplicidad. Tampoco se decide automáticamente entre promedio y stacking por diferencias pequeñas. No seguir ajustando C, pesos o estrategias tras mirar los resultados.

Podés regenerar matrices/métricas directamente desde un CSV sin ajustar modelos usando `metricas_externas.py`; preferí `stacking.py resumen` para conservar también las columnas de completitud y las comparaciones.

## Verificaciones cortas

```sh
PYTHONPATH=src .venv/bin/python experiments/seleccion/test_stacking.py
```

Incluyen: agrupamiento y exclusión del fold externo en todos los ajustes; reconstrucción de entrenamientos profundos; invariancia ante cambios de etiquetas externas; umbrales obtenidos solo de scores internos; reanudación sin nuevos ajustes; invalidación de cache por parámetros y detección de corrupción; cobertura, matrices y regeneración; reutilización exacta del control HistGB existente; smoke test con pipelines reales sobre datos sintéticos; exclusión efectiva de assigned; igualdad secuencial/paralela.
