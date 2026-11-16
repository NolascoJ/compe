# Elección final antes del holdout: modelos individuales y ensembles

## Decisión

**Recomiendo pasar únicamente HistGradientBoosting sin `assigned_gpu_type` a la evaluación final en `test.csv`.** Mantener la configuración por defecto del repositorio, el preprocesamiento E y excluir también el derivado `gpu_changed`.

El experimento nuevo no cambió la recomendación: HistGB obtiene BA externa **84,829%**, el promedio **84,730%** y el stacking **84,658%**. Ningún ensemble supera el filtro previamente acordado de ganar al menos 4 de 5 folds y mejorar más de 0,003 de BA media (0,3 puntos porcentuales). El promedio gana 2 folds; el stacking, 1. Ambos tienen mejora media negativa.

**La conclusión no es que HistGB sea estadísticamente superior de manera concluyente.** Los intervalos pareados incluyen cero y las diferencias son pequeñas. La conclusión es que no apareció una mejora consistente que justifique reemplazar al candidato simple por tres pipelines y, para stacking, una logística adicional. La elección usa la regla fijada antes de la corrida y no cambia el criterio después de ver el resultado.

De las nuevas alternativas, conservaría el promedio simple como segunda opción documental; tampoco merece pasar al holdout para competir allí. GradientBoosting individual sigue siendo una alternativa previa con menor complejidad que un ensemble de tres modelos. No hace falta decidir un orden total entre alternativas empatadas para elegir qué modelo validar finalmente.

## 1. Integridad del experimento

Corrida: `a92fbb4e4778849e`. Se completaron las 15 tareas externas: 5 folds de CV, 3 años excluidos, 2 direcciones regionales y 5 folds sin locked. Cada tarea produjo las tres estrategias. Hay **21 conjuntos agregados de métricas** (3 estrategias × 7 escenarios) y **440.844 filas de predicciones**; las observaciones se repiten entre escenarios, no son 440.844 ejemplos independientes.

La auditoría verificó:

- Hash de train, identidad del código, parámetros, versiones y manifiesto de la corrida.
- Checksums de todos los checkpoints nuevos, índices de entrenamiento/evaluación y grupos disjuntos externos.
- Etiquetas contra train y puntuaciones exportadas contra los checkpoints.
- Umbrales recalculados exclusivamente desde scores internos y etiquetas del entrenamiento externo.
- Promedio de las tres probabilidades y predicción logística reconstruida desde coeficientes, intercepto y probabilidades externas.
- Matrices y métricas recalculadas desde las predicciones guardadas.
- HistGB reproduce exactamente las predicciones, umbrales y scores de la calibración anterior en todos los escenarios.

Los manifiestos de esta corrida no registran advertencias en los ajustes nuevos. Se hicieron 465 ajustes base nuevos; se reutilizaron 30 resultados internos y 30 externos previos, con sus bloques internos correspondientes. Las verificaciones previas de implementación incluyeron aislamiento en los niveles profundos, invariancia ante cambios de etiquetas externas y equivalencia secuencial/paralela.

Se auditaron y analizaron resultados; no se reentrenaron modelos en esta revisión y **no se abrió ni evaluó test.csv**. La consulta anterior autorizada sobre coincidencias de features entre train y test permanece declarada: 281 filas del holdout, 2,94%, comparten las 31 entradas originales con train. No se habían leído etiquetas del holdout.

## 2. Qué evalúa esta comparación

Todas las estrategias excluyen assigned y usan las configuraciones base fijas de `parametros.json`. El promedio tiene pesos iguales. La logística del stacking usa L2, C=1, solver lbfgs, max_iter=1000, sin class_weight y solo tres entradas: probabilidades de GB, HistGB y RF.

Cada fold externo se reserva completamente. HistGB y promedio eligen umbral con OOF internas de tres folds agrupados. Para stacking se genera una evaluación interna del **procedimiento completo**: dentro de cada entrenamiento interno se construyen nuevas OOF base con tres subfolds, se entrena la logística y luego se predice la validación interna. Esas predicciones permiten elegir el umbral. Finalmente se entrena una logística sobre todas las OOF base del entrenamiento externo y se aplica al fold externo.

No se hizo CV de una logística sobre una matriz OOF global cuyos modelos base hubieran visto su validación. Preprocesamiento, entrenamiento y elección de umbral respetan las particiones. Esto evita esa fuga específica del stacking. No elimina el optimismo histórico de seleccionar hiperparámetros y estrategias sobre datos ya usados: no se repitió la búsqueda completa de hiperparámetros dentro de cada fold externo.

En las tablas siguientes, BA usa umbrales internos independientes del fold evaluado. Cuando se menciona una BA con umbral global OOF se identifica expresamente como levemente optimista. El desvío entre folds usa ddof=0 y no es un intervalo de confianza.

## 3. Rendimiento principal y regla de selección

| Estrategia | BA global | BA media folds | Desvío folds (puntos) | ROC-AUC global |
| --- | --- | --- | --- | --- |
| histgb | 84.83% | 84.83% | 1.030 | 0.93554 |
| promedio | 84.73% | 84.73% | 1.186 | 0.93525 |
| stacking | 84.66% | 84.66% | 1.270 | 0.93582 |

La BA global se calcula concatenando predicciones externas; la media se calcula dando igual peso a los cinco folds. Ambas coinciden casi exactamente porque los folds tienen tamaños y distribuciones similares, pero se guardan como métricas distintas.

BA de cada fold, numerados de 0 a 4:

| fold_externo | histgb | promedio | stacking |
| --- | --- | --- | --- |
| 0.0 | 84.75% | 84.98% | 84.72% |
| 1.0 | 82.98% | 82.57% | 82.39% |
| 2.0 | 85.76% | 85.69% | 85.75% |
| 3.0 | 85.82% | 85.89% | 85.95% |
| 4.0 | 84.83% | 84.53% | 84.48% |

Diferencias e intervalos de ensemble menos HistGB:

| Estrategia | Folds ganados / 5 | Diferencia media (puntos) | IC 95% inferior (puntos) | IC 95% superior (puntos) | Pasa regla |
| --- | --- | --- | --- | --- | --- |
| promedio | 2 | -0.100 | -0.473 | +0.249 | False |
| stacking | 1 | -0.172 | -0.515 | +0.135 | False |

El intervalo se calculó sobre la diferencia de BA global con 2.000 réplicas pareadas, remuestreando grupos completos de entradas idénticas y conservando la multiplicidad de filas. Es condicional a las predicciones ya generadas: no incorpora repetir ajustes, buscar parámetros ni seleccionar estrategias. No se interpreta como cobertura garantizada del rendimiento futuro.

La regla previamente definida ya determina conservar HistGB. No hace falta aumentar C, cambiar pesos, quitar una base o mover el umbral buscando que algún ensemble la cumpla: eso sería abrir otra búsqueda después de observar la validación. Tampoco se puede afirmar equivalencia exacta entre los tres modelos solo porque un intervalo incluya cero.

## 4. Qué cambia en los errores

| modelo | TN | FP | FN | TP | sensibilidad | especificidad | precision | f1 | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| histgb | 19057 | 3253 | 2516 | 13448 | 84.24% | 85.42% | 80.52% | 82.34% | 84.93% |
| promedio | 18965 | 3345 | 2482 | 13482 | 84.45% | 85.01% | 80.12% | 82.23% | 84.78% |
| stacking | 19183 | 3127 | 2661 | 13303 | 83.33% | 85.98% | 80.97% | 82.13% | 84.88% |

El promedio detecta 34 cancelaciones adicionales, pero genera 92 falsos positivos adicionales frente a HistGB. La mejora de recall no compensa la reducción de especificidad bajo balanced accuracy. El stacking reduce 126 falsos positivos, pero deja de detectar 145 cancelaciones que HistGB detectaba en el balance de cantidades. Esos conteos se refieren a diferencias netas de las matrices, no a que todas las filas cambiadas tengan el mismo sentido.

En las transiciones individuales, el promedio corrige 691 errores de HistGB e introduce 749 errores nuevos; el stacking corrige 563 e introduce 582. El total de errores por sí solo no es la métrica de selección, porque BA pondera ambas clases por igual.

Stacking tiene ROC-AUC global apenas mayor (0,93582 frente a 0,93554), pero no mayor BA. No cambiaría la métrica principal a AUC para justificarlo. AUC evalúa ordenamiento a través de umbrales; BA corresponde al umbral elegido y al objetivo acordado. Además, el AUC global mezcla puntuaciones de modelos de folds distintos; se guardó también por fold.

## 5. Estabilidad, diversidad y complejidad

HistGB tiene el menor desvío de BA entre folds: 1,03 puntos, frente a 1,19 del promedio y 1,27 del stacking. La muestra de cinco folds no permite concluir que siempre tendrá menor variabilidad, pero en esta corrida combinar modelos no la redujo.

Las correlaciones de probabilidades externas son altas: GB–HistGB = **0,980**, GB–RF = **0,983** e HistGB–RF = **0,964**. Esto es evidencia de similitud de scores, no una prueba de que todas las bases cometan los mismos errores ni de que un ensemble jamás pueda mejorar. En esta corrida, la complementariedad no produjo una mejora neta consistente de BA.

La logística asigna el mayor coeficiente a HistGB en los cinco folds de CV. Sus coeficientes van aproximadamente de 3,10 a 3,81 para HistGB, 0,99 a 2,25 para GB y 1,53 a 2,09 para RF. **No son porcentajes, importancias causales ni pesos que deban sumar uno.** Las entradas correlacionadas permiten variación de coeficientes sin grandes cambios de predicción. En PRT excluido cambia incluso el signo del coeficiente de GB: muestra que la combinación aprendida depende de la distribución de entrenamiento.

Sensibilidad descriptiva del umbral, desplazando ±5 puntos de cuantil respecto del corte interno y calculando los nuevos cortes solo con scores internos: la peor caída de BA media es aproximadamente 0,76 puntos para HistGB, 0,50 para el promedio y 0,45 para stacking. Los ensembles son algo más planos en este diagnóstico. Esa ventaja no compensa la ausencia de mejora en BA/robustez, pero debe reconocerse. Estos cortes alternativos no se seleccionan ni reemplazan el umbral elegido; es un diagnóstico, no una nueva optimización.

Para desplegar HistGB se necesita un solo pipeline, con 100 iteraciones. Ambos ensembles requieren además GB con 586 estimadores y RF con 959 árboles; stacking agrega una logística. No se midió latencia final en Colab, por lo que no se cuantifica una aceleración. Sí hay más componentes que entrenar, serializar, cargar y mantener. Si no aportan una mejora, esa complejidad no está justificada.

## 6. Robustez fuera de la distribución de entrenamiento

BA con el mismo aislamiento estricto, sin usar el año o región excluidos para calibrar:

| escenario | histgb | promedio | stacking |
| --- | --- | --- | --- |
| eval_PRT | 75.56% | 74.13% | 76.26% |
| eval_anio_2015 | 84.32% | 84.25% | 84.18% |
| eval_anio_2016 | 77.47% | 76.36% | 75.88% |
| eval_anio_2017 | 79.21% | 78.51% | 78.66% |
| eval_resto | 51.47% | 53.46% | 51.64% |
| sin_locked | 80.60% | 80.71% | 80.81% |

**Temporal:** HistGB obtiene la mayor BA en los tres años excluidos. Frente al promedio, sus ventajas son 0,06, 1,11 y 0,70 puntos en 2015, 2016 y 2017. Frente a stacking son 0,14, 1,59 y 0,55 puntos. La ventaja de 2015 es mínima; no se presenta como una diferencia concluyente. El patrón conjunto no sugiere mejora temporal del ensemble.

**PRT no visto:** stacking alcanza 76,26%, frente a 75,56% de HistGB: gana 0,70 puntos. Es una ventaja real medida en este escenario y debe conservarse en el informe. No cumple, sin embargo, el objetivo de una mejora consistente: empeora los años y no gana en CV. El promedio baja a 74,13%.

**Entrenar solo con PRT y evaluar el resto:** promedio gana, 53,46% contra 51,47% de HistGB y 51,64% de stacking. Aun así los tres quedan cerca de BA 50%. No es un problema resuelto por el ensemble. Si el criterio único fuera maximizar el peor escenario, el promedio sería preferible; no es el criterio acordado, que exige también mejora consistente de CV. Ninguno domina en todos los escenarios.

**Sin locked en entrenamiento ni evaluación:** stacking alcanza 80,81%, promedio 80,71% e HistGB 80,60%. Ventajas pequeñas de 0,21 y 0,11 puntos; no bastan para sustituir el candidato ni para decidir eliminar locked del entrenamiento final. Es una población y un esquema de folds diferentes del entrenamiento normal.

Es importante separar la dificultad intrínseca de cada grupo del efecto de excluirlo del entrenamiento. Para HistGB, las caídas contra las **mismas filas en CV normal** son 7,03 puntos en 2015, 7,44 en 2016, 2,33 en 2017, 9,97 en PRT y 29,40 en el resto tras entrenar solo con PRT. No atribuir toda la diferencia respecto de la BA global de 84,83% al cambio de distribución.

No confundir un futuro test sin PRT con entrenar un modelo solo en PRT: el candidato final se entrenará con todas las regiones de train. El desglose de regiones conocidas distintas de PRT es otro diagnóstico, mostrado a continuación.

## 7. Subgrupos dentro de la CV normal

| subgrupo | histgb | promedio | stacking |
| --- | --- | --- | --- |
| PRT | 85.53% | 85.00% | 85.31% |
| anio_2015 | 91.35% | 90.03% | 90.20% |
| anio_2016 | 84.91% | 85.11% | 84.94% |
| anio_2017 | 81.54% | 81.57% | 81.53% |
| resto | 80.87% | 80.81% | 80.46% |
| sin_locked | 79.92% | 79.90% | 79.48% |

El peor subgrupo de HistGB es sin locked, con 79,92%. El promedio tiene 79,90% y stacking 79,48%. HistGB mantiene el mejor mínimo, aunque su rango entre mejor y peor subgrupo es mayor: 11,42 puntos, frente a 10,13 del promedio y 10,72 del stacking. El menor rango de los ensembles viene acompañado de menor BA en 2015, no de una mejora clara del mínimo.

En la CV normal, stacking empeora sin locked y resto de regiones frente a HistGB. Su pequeña ganancia al **reentrenar** sin locked no equivale a una mejora del predictor normal cuando el test contenga pocas filas locked. Son experimentos diferentes y no deben mezclarse.

## 8. Relación con los modelos individuales anteriores

Estas son las configuraciones fijas evaluadas previamente con umbrales internos, no los máximos brutos de las búsquedas:

| modelo | variante | ba_umbral_interno |
| --- | --- | --- |
| logistic | con_assigned | 82.53% |
| logistic | sin_assigned | 82.01% |
| ridge | con_assigned | 81.95% |
| linear_svc | con_assigned | 82.18% |
| knn | con_assigned | 80.86% |
| random_forest | con_assigned | 84.32% |
| bagging | con_assigned | 84.26% |
| gradient_boosting | con_assigned | 84.92% |
| gradient_boosting | sin_assigned | 84.27% |
| hist_gradient_boosting | con_assigned | 85.13% |
| hist_gradient_boosting | sin_assigned | 84.83% |

HistGB con assigned tiene BA algo mayor, pero la ganancia frente a excluirla es solo 0,30 puntos y no resulta concluyente en el bootstrap pareado anterior. La variable se registra tarde; se mantiene excluida junto con gpu_changed. No se reabre esa decisión por diferencias pequeñas.

HistGB sin assigned supera a logística sin assigned por 2,82 puntos y ganó en los cinco folds de la comparación previa. También supera holgadamente las referencias de clase constante (50%) y regla locked (~69,19%). El anterior árbol de una sola variable seleccionado exclusivamente dentro del entrenamiento obtuvo ~69,18%. No hay señal de subajuste frente a esas referencias.

La evidencia de robustez completa cubre logística, GB, HistGB y los dos ensembles. RF, Bagging, Ridge, SVC y KNN no recibieron todas las pruebas de cambio de distribución en ambas variantes. No se afirma superioridad universal sobre opciones no evaluadas; se elige el candidato con mejor respaldo conjunto disponible. XGBoost y LightGBM siguen pendientes y no se corrieron.

## 9. Sobreajuste: qué permite y qué no permite concluir

La evaluación estricta elimina la fuga del umbral y de la logística hacia su fold externo. Eso no significa «cero overfitting».

En el control previo de HistGB sin assigned, el gap entrenamiento−validación fue **4,17 puntos**. El K-Fold común dio 86,67% de BA frente a 84,84% agrupado: **1,82 puntos de inflado por copias**. Ambas BA del control usan umbral global OOF y son levemente optimistas; no confundirlas con 84,83% de evaluación con umbrales internos.

El candidato aún aprende patrones sensibles a duplicados y a cambios de población. No se calculó un gap de entrenamiento comparable para los ensembles en este experimento; no se puede afirmar que stacking tenga mayor sobreajuste solo porque su BA sea ligeramente menor. La conclusión observada es que no mejora la generalización medida lo suficiente para justificarlo.

Siguen pendientes la validación anidada de toda la búsqueda de hiperparámetros y repeticiones externas con otras semillas. Los cinco folds tampoco son cinco ensayos independientes, porque sus entrenamientos se solapan. Los intervalos bootstrap publicados no cubren todas esas fuentes de incertidumbre. Seguir probando variantes después de mirar estos resultados aumentaría el sesgo de selección.

## 10. Candidato que propongo congelar para test

- Estimador: `HistGradientBoostingClassifier`, constructor `MODELS['hist_gradient_boosting'].build({})`.
- Pipeline E original y `use_assigned=False`; incluye exclusión de gpu_changed.
- learning_rate=0.1; max_iter=100; max_leaf_nodes=31; max_depth=None; min_samples_leaf=20; l2_regularization=0; max_features=1; class_weight=None; categorical_features='from_dtype'; early_stopping=False; random_state=42.
- Mantener las filas locked y las demás features del pipeline, sin reglas nuevas por región o año.
- Entrenar el pipeline completo con todo train para la evaluación final.
- Propuesta de umbral congelado a partir de los OOF de train de esta misma configuración: **0,38294747099493937**.

Este último es el umbral global OOF disponible, seleccionado solo con train. La BA sobre esos mismos OOF sería levemente optimista. **No es el único umbral que produjo el 84,83% externo**: esa métrica usó cinco umbrales aprendidos en entrenamientos distintos. Para el predictor ajustado en todo train corresponde fijar un umbral a partir de train; no elegirlo ni corregirlo mirando test.

La configuración y recomendación quedan registradas en `experiments/seleccion/revision_stacking/candidato_para_test.json`. Es una especificación para la siguiente etapa, no un modelo ya entrenado ni una evaluación del holdout.

## 11. Qué esperar y cómo usar el holdout

Como referencia de CV, el candidato rindió 84,83% BA con desvío entre folds 1,03 puntos. Para datos comparables a train, 82–86% puede usarse como expectativa orientativa; **no es un intervalo formal ni una garantía**. Bajo años excluidos observamos 77,47–84,32%, y bajo extrapolación regional extrema hasta 51,47%. No hay una banda estrecha defendible para cualquier distribución desconocida.

Los siguientes pasos se proponen, pero no se ejecutaron:

1. Aceptar y congelar este candidato, su pipeline y el umbral/procedimiento antes de abrir etiquetas del holdout.
2. Comprobar ejecución/compatibilidad en Colab. La corrida actual usa Python 3.14.2, pandas 3.1.1 y sklearn 1.9.1; no asumir compatibilidad de artefactos serializados con otras versiones.
3. Ajustar en todo train y evaluar HistGB una sola vez en test.csv: BA, matriz, recall, especificidad y desgloses previamente definidos. Declarar el 2,94% de coincidencias de features y reportar adicionalmente el subconjunto sin coincidencias, en la misma evaluación.
4. No probar los demás candidatos en test para elegir el mejor, ni ajustar el umbral si el resultado decepciona. Si se decide volver al desarrollo por un problema material, declarar que ese holdout dejó de ser una comprobación final independiente.
5. Para la entrega final, reentrenar con train+test según las reglas, manteniendo fija la configuración y el procedimiento de umbral, y validar predictor.py en Colab. Eso ocurre después de registrar la evaluación final independiente.

**Cierre:** el nuevo experimento era razonable, se ejecutó con aislamiento estricto y no encontró un candidato que justifique reemplazar HistGB. Recomiendo cerrar la selección actual y usar únicamente HistGradientBoosting sin assigned para la próxima comprobación en test, aceptando y comunicando sus límites de robustez.

## Archivos y reproducción

- Resultados conservados: `experiments/seleccion/stacking_resultados/a92fbb4e4778849e/resultados/`.
- Auditoría: `experiments/seleccion/auditar_stacking_final.py`, ejecutable con `PYTHONPATH=src .venv/bin/python`; no entrena ni abre test.
- Tablas adicionales: `experiments/seleccion/revision_stacking/` (auditoría, correlaciones, sensibilidad y cambios de errores).
- Generación de este informe: `experiments/seleccion/revision_stacking/generar_informe.py`.
- Informe anterior, complementario: `reports/seleccion_modelo.md`.
