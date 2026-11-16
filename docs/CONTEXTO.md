# Contexto técnico y decisiones actuales

Actualizado el 16 de noviembre de 2026. Este documento resume el contexto de trabajo necesario para continuar el proyecto; los informes y resultados originales se conservan como evidencia. No es una transcripción de conversaciones ni contiene credenciales.

## Objetivo y datos

TP3 de Machine Learning del ITBA, Grupo 9. Clasificación binaria de cancelación de reservas de GPU, target `canceled_job`; `no_show` cuenta como cancelación. Objetivo de competencia: maximizar BA, promedio de sensibilidad y especificidad.

- TRAIN: 38.274 filas.
- Holdout interno etiquetado: 9.568 filas, archivo `test.csv`.
- El test oculto de los profesores es otro conjunto.
- Split compartido aproximadamente 80/20, semilla 42, por grupos de las 34 columnas completas.
- Las repeticiones se conservaron. El agrupamiento es una aproximación a identidad de entradas, no un identificador probado de cliente o reserva.

## Pipeline y disponibilidad de variables

Se excluyen `execution_status` y `execution_status_date` por describir el resultado. `canceled_job` se separa como objetivo. El predictor elegido excluye `assigned_gpu_type` y su derivada `gpu_changed` por posible registro tardío.

Las categorías raras se agrupan con umbral de 20 ejemplos, los nulos categóricos tienen etiqueta explícita, el único nulo numérico de procesos auxiliares se rellena con cero y el costo se recorta al percentil 99 aprendido en entrenamiento. Se generan `empty_job`, `has_agent`, `has_organization` y `execution_dow`; esta última reemplaza a `execution_week`.

- Lineales y KNN: one-hot, log1p en variables de cola larga y estandarización.
- RF, Bagging y GradientBoosting: codificación ordinal, sin escalado.
- HistGradientBoosting: categóricas nativas, sin one-hot ni escalado.

Todo lo aprendido de los datos se ajusta dentro del entrenamiento correspondiente.

## Recorrido experimental original

Ocho familias: logística, Ridge, Linear SVC, KNN, Random Forest, Bagging, GradientBoosting e HistGradientBoosting. Se guardaron 530 configuraciones de búsqueda amplia y 139 de refinamiento: 669 configuraciones, además de referencias, curvas y réplicas de confirmación.

La selección original utilizó BA OOF, un filtro preliminar de gap y preferencia por simplicidad dentro de un error estándar de la mejor. Se confirmaron tres candidatos por familia con semillas 42, 43 y 44. Esa confirmación corresponde a las configuraciones originales, no automáticamente a sus revisiones posteriores.

Se revisaron configuraciones para reducir complejidad. HistGB pasó de la elección inicial de 868 iteraciones y 32 hojas a 100 iteraciones y 31 hojas. GradientBoosting regularizado usa 586 árboles y hoja mínima 82. Logística se mantuvo como referencia lineal. Los finalistas estudiados no son simplemente los tres primeros por BA.

## OOF y validación

OOF: cada fila de TRAIN recibe un score de un ajuste que no entrenó con esa fila. Se entrenan cinco pipelines, cada uno con cuatro folds y puntuando el restante. Se reúnen los scores en el orden original: una predicción OOF por fila, no un promedio de cinco predicciones por fila.

En la evaluación del procedimiento de umbral:

1. Reservar un fold externo para evaluar.
2. Usar los otros cuatro como entrenamiento externo.
3. Dentro de ese entrenamiento, producir OOF con tres folds internos y elegir el corte que maximiza BA.
4. Ajustar con todo el entrenamiento externo y evaluar el fold reservado con ese corte.
5. Repetir para los cinco folds externos.

Esto separa la elección del corte de su fold evaluado, pero **no es una validación completamente anidada de toda la búsqueda histórica de hiperparámetros**.

## Candidato y evaluación histórica

HistGradientBoosting, pipeline E, `use_assigned=False`, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, min_samples_leaf=20, max_depth=None, l2_regularization=0, class_weight=None, max_features=1, early_stopping=False, random_state=42.

- BA externa con cinco cortes internos: 84,8293%; desvío entre folds 1,0295 puntos.
- ROC-AUC OOF global: 0,935542.
- Corte global OOF congelado antes de evaluar holdout: 0,38294747099493937.
- BA sobre los mismos OOF que eligen ese corte global: 84,8433%, algo optimista.
- Gap diagnóstico original: BA train 89,0178% menos BA OOF global 84,8433% = 4,1745 puntos. No se calcula mezclando esta BA train con la BA de cortes internos.
- Holdout interno: BA 85,0904%; ROC-AUC 0,937743; matriz [[4765,812],[609,3382]].
- En holdout, 281 filas (2,94%) comparten las 31 entradas originales con train. Sin coincidencias: n=9.287, BA=85,90%. Esto no garantiza independencia de clientes.

El holdout ya fue consultado. No sirve para una nueva confirmación independiente de variantes elegidas después. La compatibilidad en Colab y el esquema de entradas del evaluador deben comprobarse antes de entregar; las versiones locales no prueban esa compatibilidad.

## Ensembles y auditoría adicional

Se probaron promedio y stacking de GB, HistGB y RF, excluyendo GPU asignada y respetando la selección interna de umbrales. Obtuvieron BA externa 84,73% y 84,66%, frente a 84,83% de HistGB; no hubo mejora consistente que justificara su complejidad.

La revisión del 16/11 verificó las ocho familias, 669 configuraciones y 47 conjuntos de predicciones previos. Recuperó RF sin GPU asignada del caché sin entrenar: BA externa 83,97%, ROC-AUC 0,9278. Los resultados y métodos están en `output/revision_modelos_20261116/`.

### Alcance del agrupamiento

Los grupos originales de CV usan las 31 entradas originales, incluida GPU asignada aunque después se descarte. Al considerar las entradas crudas efectivas del predictor, 1.259 filas (3,29% de TRAIN) tienen coincidencias entre validación y entrenamiento de sus folds originales. Eso no demuestra identidad de reservas ni fuga de etiquetas, pero limita la afirmación de que ninguna entrada efectiva idéntica cruza folds.

Se realizó un control nuevo con grupos de las 29 entradas crudas efectivas, sin GPU asignada ni semana descartada: 25.444 grupos. Se compararon tres candidatos fijos con cinco folds externos y tres internos; 60 ajustes sobre TRAIN, sin usar TEST y sin advertencias.

| Candidato | BA externa | Desvío folds (pp) | Gap diagnóstico (pp) |
| --- | ---: | ---: | ---: |
| HistGB, 31 hojas | 84,68% | 0,93 | 4,00 |
| HistGB, 15 hojas | 84,28% | 0,71 | 2,58 |
| GradientBoosting regularizado | 84,06% | 0,93 | 3,37 |

El gap de esta tabla usa el corte global OOF; la BA externa usa cortes internos. Cambiar grupos también cambia particiones: la diferencia contra la CV anterior no es un efecto causal aislado del agrupamiento.

Conclusión: no hay base para diagnosticar sobreajuste grave solamente por un gap de 4,17 puntos, ni para afirmar ausencia de sobreajuste. Reducir hojas disminuye gap y dispersión, pero también BA. Se mantiene el candidato original de 31 hojas cuando se prioriza BA. La variante de 15 hojas es una alternativa conservadora dentro de un error estándar de la mejor en ese control; no se seleccionó como predictor final ni hereda el TEST de 85,09%. No se cambió el modelo o umbral congelado al hacer esta revisión.

## Comparación con el TP2 anterior

Se revisó el repo `agkorman/ml_tp2_grupo9` y después los resultados locales del TP2. El RF final tenía 100 árboles, profundidad 12, split mínimo 200 y hoja mínima 10. Sus archivos locales guardan ROC-AUC train=0,8265374368 y validación=0,7922620964: gap de 0,0342753404, o **3,43 puntos de AUC**. Se verificó contra la fila exacta de la grilla, no contra la curva exploratoria de 9,8 puntos. El gap de AUC no se compara directamente con el de BA del TP3.

El repositorio remoto del TP2 excluye resultados generados; por eso el dato numérico no estaba disponible allí. La copia local sí lo conservaba. El informe de revisión tiene la actualización y los fuentes remotos inspeccionados para trazabilidad.

## Cómo continuar

1. Leer este contexto y `output/revision_modelos_20261116/informe.md`.
2. Restaurar el CSV comprimido con `python scripts/restaurar_artefactos.py`.
3. Preservar los resultados originales y distinguir siempre configuración, variables, agrupamiento y protocolo de umbral.
4. Cualquier experimento adicional se decide con TRAIN. No reutilizar el holdout como competición entre candidatos.
5. Si se modifica el predictor, documentarlo como variante nueva y verificar interfaz y entorno de ejecución antes de enviar.

Las presentaciones y los artefactos de Canva se excluyen de esta publicación por pedido del usuario. La evidencia técnica completa se conserva en los informes y experimentos.
