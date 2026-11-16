# Selección de modelo — revisión de resultados de calibración

## Recomendación y alcance

**Recomiendo HistGradientBoosting con los valores por defecto del repositorio y `use_assigned=False`. Segundo: GradientBoosting regularizado, también sin assigned.** Es una elección defendible entre los candidatos evaluados, no una demostración de que el problema esté resuelto para cualquier distribución.

La BA externa del recomendado es **84,83%**, con desvío entre cinco folds de **1,03 puntos porcentuales**. Supera a logística sin assigned por 2,82 puntos y a GradientBoosting sin assigned por 0,56 puntos. La segunda diferencia es menor que la dispersión: no la uso como argumento suficiente para declarar superioridad. Desempato por comportamiento regional, mejor mínimo de BA en los subgrupos observados y menor cantidad de iteraciones (100 frente a 586). GradientBoosting tiene menor gap y una curva de sensibilidad al umbral más plana: son ventajas reales del segundo.

**No diría que «ya está bien todo».** HistGradientBoosting todavía tiene un gap de 4,17 puntos y gana 1,82 puntos al permitir copias entre folds. Además, al entrenar solo con PRT y evaluar otras regiones, cae a 51,47%. Ningún finalista supera satisfactoriamente ese escenario. La recomendación sirve para continuar hacia la comprobación final, con estos límites explícitos.

En datos de composición comparable a train, usaría **82–86% de BA como expectativa orientativa**, no como intervalo de confianza ni garantía. En años excluidos se observó 77,47–84,32%; para extrapolación regional extrema, 51,47–75,56%. No hay un único rango estrecho honesto para el test desconocido. La selección previa entre hiperparámetros sigue aportando optimismo.

## Integridad y método

Se auditaron los 47 conjuntos modelo/variante/escenario: 11 CV completas y 36 evaluaciones de robustez. Se comprobó cobertura exacta de índices, etiquetas contra train, `y_pred = score >= umbral`, matrices y BA recalculadas desde predicciones. Se comprobó el hash de train, el agrupamiento disjunto de cada fold interno guardado y se recalcularon sus umbrales. Las nueve pruebas de `test_calibracion.py` pasaron. La inspección del código confirma que el entrenamiento externo contiene toda la calibración interna, con pipeline ajustado dentro de cada fold.

Cada resultado principal usa 5 folds externos y 3 internos agrupados para elegir umbral. No se eligió un nuevo umbral mirando los resultados externos. Esto evalúa independientemente la elección del umbral; **no constituye validación anidada de la búsqueda de hiperparámetros**. El desvío entre folds usa ddof=0 y no es un intervalo de confianza.

Hay 42 ConvergenceWarning de logística: 17 por variante en sin_locked y 4 por variante en eval_PRT. Estos escenarios de logística deben interpretarse como resultados de ajustes que alcanzaron max_iter, no como óptimos convergidos. No hubo esas advertencias en los boosting ni en la CV normal de logística. No se usan esos ajustes problemáticos como prueba decisiva de superioridad del ganador. Las advertencias de deprecación de penalty no son fallos de convergencia.

No se volvió a abrir test.csv en esta revisión. En una consulta anterior autorizada se compararon exclusivamente sus features: 281/9.568 filas (2,94%) coinciden con features de train; las etiquetas no se leyeron. Esto debe declararse al presentar la independencia del futuro holdout.

## Comparación de las configuraciones fijas

Los parámetros completos son los de `experiments/seleccion/parametros.json`; no son necesariamente los máximos originales de búsqueda. La siguiente tabla usa umbrales internos. La columna de caída compara contra BA con umbral global OOF, que es levemente optimista; una caída negativa es posible porque cinco umbrales pueden rendir mejor que uno global.

| Modelo | Variante | BA externa | Desvío entre folds (puntos) | Caída por calibrar (puntos) |
| --- | --- | --- | --- | --- |
| logistic | con_assigned | 82.53% | 1.365 | 0.182 |
| logistic | sin_assigned | 82.01% | 1.386 | -0.039 |
| ridge | con_assigned | 81.95% | 1.188 | 0.481 |
| linear_svc | con_assigned | 82.18% | 1.224 | 0.135 |
| knn | con_assigned | 80.86% | 1.360 | 0.221 |
| random_forest | con_assigned | 84.32% | 1.123 | 0.067 |
| bagging | con_assigned | 84.26% | 1.116 | 0.071 |
| gradient_boosting | con_assigned | 84.92% | 1.442 | 0.170 |
| gradient_boosting | sin_assigned | 84.27% | 1.503 | 0.120 |
| hist_gradient_boosting | con_assigned | 85.13% | 1.215 | 0.177 |
| hist_gradient_boosting | sin_assigned | 84.83% | 1.030 | 0.014 |

La calibración cambia poco a los boosting. En HGB sin assigned, la BA global OOF (levemente optimista) es 84,843%, frente a 84,829% con calibración interna: solo 0,014 puntos de diferencia. Esto NO demuestra ausencia de sobreajuste: corrige una fuente específica de optimismo, no las otras.

Referencias: clase constante = 50%; regla locked_allocation → 1 = 69,19%. Una referencia de una sola variable, definida como árbol de profundidad 1 y elegida dentro del entrenamiento de cada fold entre todas las entradas excepto assigned (one-hot para categorías/IDs, imputación numérica aprendida en entrenamiento), obtiene 69,18%. Elige reservation_commitment en los cinco folds. No es una búsqueda de todas las funciones univariables posibles. Logística sin assigned obtiene 82,01%. El ganador mejora a la referencia lineal en los cinco folds: +2,68, +3,59, +2,41, +3,47 y +1,94 puntos. No presenta subajuste frente a esas referencias.

## Comparación pareada y sensibilidad

Diferencia de BA: HGB sin assigned menos el comparador. Bootstrap pareado de 2.000 réplicas, semilla 42, remuestreando uniformemente grupos completos de features idénticas, conservando su multiplicidad de filas. Intervalos percentiles al 95%, en puntos porcentuales:

| comparador | assigned | diferencia | ic025 | ic975 |
| --- | --- | --- | --- | --- |
| bagging | True | 0.570 | 0.113 | 1.041 |
| gradient_boosting | False | 0.557 | 0.070 | 1.062 |
| gradient_boosting | True | -0.092 | -0.603 | 0.425 |
| hist_gradient_boosting | False | 0.000 | 0.000 | 0.000 |
| hist_gradient_boosting | True | -0.305 | -0.694 | 0.091 |
| knn | True | 3.974 | 3.277 | 4.678 |
| linear_svc | True | 2.649 | 1.923 | 3.491 |
| logistic | False | 2.818 | 2.245 | 3.450 |
| logistic | True | 2.298 | 1.720 | 2.879 |
| random_forest | True | 0.509 | 0.040 | 0.982 |
| ridge | True | 2.883 | 2.112 | 3.785 |

Estos intervalos son condicionales a los modelos ya entrenados y a esta selección. No incluyen la incertidumbre de repetir el entrenamiento ni de buscar hiperparámetros. Aunque frente a GradientBoosting sin assigned el intervalo queda ligeramente por encima de cero, la diferencia sigue siendo pequeña respecto de la dispersión entre folds y no fundamenta por sí sola la elección.

HGB sin assigned vence a GradientBoosting sin assigned en 4/5 folds; sus diferencias son −0,43, +1,64, +0,74, +0,41 y +0,42 puntos. Frente a GradientBoosting con assigned hay empate práctico: diferencia global −0,09 puntos, intervalo −0,60 a +0,43.

Sensibilidad descriptiva: desplazar el corte ±5 puntos de cuantil dentro de cada distribución de scores externos hace perder a HGB sin assigned hasta 0,80 puntos de BA; a GradientBoosting sin assigned, hasta 0,33 puntos. A ±2,5 puntos de cuantil, las pérdidas máximas son 0,24 y 0,075 puntos. **GradientBoosting es más plano en este diagnóstico.** No se utilizó para modificar el umbral ni se lo interpreta como evaluación independiente de nuevos cortes. Una variación de prevalencia por sí sola no obliga a cambiar el umbral óptimo para BA si se mantienen las distribuciones condicionales de scores; las pruebas regionales cambian mucho más que la prevalencia.

## Sobreajuste y simplicidad

Para el recomendado se repitieron únicamente controles con parámetros fijos. La CV agrupada sin assigned tiene gap 4,17 puntos; la CV común alcanza BA 86,67%, frente a 84,84% agrupada (ambas con umbral global OOF, levemente optimistas). **Inflado por copias: 1,82 puntos.** No es despreciable: sustenta conservar agrupamiento y descarta prometer «sin sobreajuste».

La curva original con assigned coloca 31 hojas antes del aumento fuerte de gap de 63/127 hojas, pero ya con gap cercano a cuatro puntos. No prueba por sí sola que la variante sin assigned sea sana. La versión ajustada original de HGB usa 868 iteraciones y 32 hojas; su ventaja original sobre default es solo 0,41 puntos de BA OOF (levemente optimista), inferior a la dispersión. Se mantiene default por simplicidad, no por demostrar igualdad exacta.

GradientBoosting regularizado tiene gap previo de 2,96 puntos con assigned frente a 4,32 de su elección original. Esta es su ventaja de regularización; no confundirla con una medición de gap de su variante sin assigned. Los defaults de RF y Bagging tenían gaps alrededor de 14–15 puntos, motivo para preferir sus configuraciones regularizadas. KNN queda por debajo de los lineales y en el borde de 301 vecinos. Ridge/SVC son simples pero no compensan su menor BA frente a HGB en la CV evaluada.

No impongo retroactivamente un máximo de gap de 0,03 para conservar solo los resultados favorables. El 0,05 es un filtro preliminar, no un certificado de generalización. No se hicieron nuevas búsquedas. La confirmación de tres semillas de los experimentos originales no debe atribuirse a las nuevas configuraciones ni a sus variantes sin assigned.

## Robustez de los tres finalistas sin assigned

BA con calibración exclusivamente dentro de cada entrenamiento:

| escenario | gradient_boosting | hist_gradient_boosting | logistic |
| --- | --- | --- | --- |
| eval_PRT | 55.26% | 75.56% | 72.85% |
| eval_anio_2015 | 84.32% | 84.32% | 86.17% |
| eval_anio_2016 | 77.37% | 77.47% | 77.50% |
| eval_anio_2017 | 78.67% | 79.21% | 77.18% |
| eval_resto | 53.86% | 51.47% | 52.19% |
| sin_locked | 80.08% | 80.60% | 77.25% |

HGB y GradientBoosting empatan aproximadamente en las pruebas por año sin assigned. HGB tiene una ventaja grande cuando PRT es totalmente nuevo: 75,56% frente a 55,26%. En la dirección contraria todos fracasan: 51,47%, 53,86% y 52,19% para HGB, GradientBoosting y logística. No elegiría el segundo por 2,40 puntos en un escenario donde ambos quedan cerca de azar y el primero tiene una ventaja de 20,30 puntos en la otra dirección. Si el criterio fuera maximizar estrictamente el peor escenario, esta preferencia cambiaría; no existe dominancia en todas las pruebas.

Los cambios por año tampoco son inocuos: para HGB, en 2016 la BA cae 7,44 puntos frente a las mismas filas evaluadas en CV normal (84,91% → 77,47%). Comparar las mismas filas es más informativo que atribuir toda la diferencia con 84,83% global al cambio de distribución. En 2015 la caída comparable es 7,03 puntos y en 2017, 2,33 puntos.

No equiparar «test sin PRT» con entrenar solo con PRT: el predictor final verá todas las regiones de train. Un test compuesto por regiones conocidas distintas de PRT se aproxima más al desglose region_resto (80,87% para HGB). La prueba PRT→resto es una extrapolación extrema que revela fragilidad, no una predicción de que todo test sin PRT dará 51%.

Desgloses de la CV normal, conservando el umbral externo de cada fold:

| subgrupo | gradient_boosting | hist_gradient_boosting | logistic |
| --- | --- | --- | --- |
| anio_2015 | 89.60% | 91.35% | 87.72% |
| anio_2016 | 84.61% | 84.91% | 81.97% |
| anio_2017 | 81.16% | 81.54% | 79.32% |
| region_PRT | 84.66% | 85.53% | 78.25% |
| region_resto | 79.88% | 80.87% | 78.93% |
| sin_locked_allocation | 78.90% | 79.92% | 76.43% |
| todo | 84.27% | 84.83% | 82.01% |

HGB tiene el mejor mínimo entre estos subgrupos: 79,92%, contra 78,90% de GradientBoosting y 76,43% de logística. Eso no significa menor rango mejor−peor: sus rangos son 11,42, 10,71 y 11,29 puntos respectivamente. HGB es mejor en nivel mínimo, pero no gana todas las definiciones de uniformidad.

Entrenado y evaluado sin locked, HGB obtiene 80,60%; entrenado con todas las filas y evaluado solo sin locked obtiene 79,92%. La mejora de 0,68 puntos no justifica eliminar esas filas del entrenamiento final: cambian entrenamiento, folds y población, y no se validó esa nueva decisión de selección. HGB detecta 44,63% de no_show (191 de 428); persiste una limitación importante.

## Decisión sobre assigned_gpu_type

Recomiendo excluirla junto con gpu_changed. HGB con assigned tiene BA 85,13%, frente a 84,83% sin assigned: aporta 0,30 puntos. El intervalo pareado para sin−con es −0,69 a +0,09 puntos; no es una ganancia concluyente. La variable se registra tarde y puede representar información no disponible al predecir. Esa dependencia temporal pesa más que la pequeña ventaja.

Declaración sugerida: «Se evaluó la variable de GPU asignada y se excluyó del predictor por registrarse tardíamente. La pérdida observada fue de aproximadamente 0,3 puntos de balanced accuracy con calibración interna; se priorizó disponibilidad de información al momento de predecir».

No se sostiene que quitarla mejore todos los escenarios: en PRT nuevo baja de 76,60% a 75,56%, por ejemplo. La decisión se basa en validez temporal, no en seleccionar solo las pruebas favorables.

## Matriz y métricas del recomendado

Matriz global externa, clase positiva cancelado=1:

| Real / predicho | 0 | 1 |
| --- | ---: | ---: |
| 0 | 19.057 | 3.253 |
| 1 | 2.516 | 13.448 |

Recall: 84,24%; especificidad: 85,42%; precision: 80,52%; F1: 82,34%; accuracy: 84,93%; BA: 84,83%. ROC-AUC global: 0,9355 (combina scores de cinco modelos; el archivo por fold permite inspeccionarlos por separado).

## Configuración propuesta y cierre

Usar `MODELS['hist_gradient_boosting'].build({})`, variante E y `use_assigned=False`. Actualmente eso implica learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0, max_features=1, class_weight=None, early_stopping=False y random_state=42, con categóricas nativas. Se debe congelar esta configuración explícitamente en el predictor.

El umbral global OOF de esa configuración es **0,38294747099493937**. Es una propuesta para ajustar un modelo en todo train y hacer la evaluación final, no el umbral único que produjo la BA externa de 84,83% (esa BA usó los cinco umbrales internos). No promediar umbrales a posteriori para atribuirles esa métrica. La BA medida sobre los mismos OOF que eligen el umbral global es levemente optimista.

**Hay suficiente evidencia para recomendar este candidato y cerrar la búsqueda actual; no para afirmar robustez universal ni declarar el predictor listo para entregar.** Quedan:

1. Acordar la elección con el equipo y congelar configuración, features y procedimiento de umbral antes de evaluar el holdout.
2. Verificar compatibilidad en Colab: esta corrida usó Python 3.14.2, pandas 3.1.1 y sklearn 1.9.1. No se comprobó ejecución en Colab ni están fijadas sus versiones. No asumir portabilidad de un pickle entre versiones.
3. Evaluar una sola vez en test.csv; declarar el solapamiento de features conocido y, en esa misma evaluación, informar también el subconjunto sin coincidencias con train. No usarlo para buscar umbrales ni elegir otro modelo oportunistamente.
4. Reentrenar para la entrega con train+test según las reglas, manteniendo hiperparámetros y el procedimiento de umbral fijados; armar predictor.py y probarlo en Colab. Ninguno de estos pasos se ejecutó aquí.

Pendientes metodológicos: validación anidada completa de hiperparámetros, confirmación de las nuevas configuraciones entre semillas, robustez de RF/Bagging y sus variantes sin assigned. Por ello, «ganador» significa el recomendado entre los finalistas evaluados en robustez; RF/Bagging no están demostrados inferiores bajo todos los cambios de distribución. XGBoost y LightGBM siguen sin correr. No se probaron ensembles. Si se exigiera proteger específicamente contra PRT→resto, habría que revisar el planteo: ninguno de estos tres sería una solución satisfactoria.

## Archivos reproducibles

- Resultados originales: `experiments/seleccion/calibracion/resultados/` (conservados).
- Auditoría y tablas añadidas: `experiments/seleccion/revision/`.
- `experiments/seleccion/auditar_resultados.py`: verifica datos, recalcula métricas, comparaciones pareadas y sensibilidad; no entrena.
- `experiments/seleccion/revision/controles_ganador.py`: controles fijos de HGB y referencia univariable, sin búsqueda de hiperparámetros.
- `experiments/seleccion/revision/generar_informe.py`: reconstruye este informe.
