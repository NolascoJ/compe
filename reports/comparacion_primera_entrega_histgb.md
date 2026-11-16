# Primera entrega vs HistGB: comparación y viabilidad de reentrega

## Conclusión

HistGB es un candidato razonable para una segunda entrega, pero no está demostrado que supere al predictor enviado en el test oculto. Antes de considerarlo un reemplazo directo hay que confirmar el esquema de entrada: el predictor enviado requiere 25 columnas; HistGB fue entrenado usando otras cinco features además de esas (y hoy su preprocesamiento exige también la presencia de assigned para descartarla). Una prueba de compatibilidad con dos filas de train limitadas a las 25 columnas confirmó un KeyError. No se ejecutó ni abrió test.csv durante esta revisión.

La captura del ranking muestra Grupo 09: BA 0,790, ROC-AUC 0,894, puesto 9 entre los nueve grupos mostrados; primero BA 0,828. La diferencia es aproximadamente 3,8 puntos porcentuales. La columna brecha del ranking no es el gap entrenamiento-validación.

## Qué se inspeccionó

Se leyó el código del .py entregado y su estado serializado. Se verificó el checksum del payload. El entorno local no tiene catboost; la inspección sustituyó únicamente la clase de CatBoost por un contenedor de estado para leer metadatos, sin ejecutar predicciones del ensemble ni instalar dependencias. No es una ejecución funcional de la entrega.

Se inspeccionaron el manifiesto y el informe ya existente del holdout HistGB, no sus datos crudos. Se cargó su pipeline para una prueba de esquema con dos filas de train. No se entrenó ningún modelo. El archivo recibido pesa 67,83 MB.

## Diferencias verificadas

| Aspecto | Primera entrega | HistGB seleccionado |
|---|---|---|
| Estimador | Promedio 50% RF + 50% CatBoost | Un HistGradientBoosting |
| RF | 500 árboles, max_depth=None, min_samples_leaf=1, max_features=0.3 | No usa RF |
| Boosting | CatBoost profundidad 8, 1.000 iteraciones, balanced=True en wrapper | 100 iteraciones, 31 hojas, min_samples_leaf=20 |
| Umbral | 0,49 | 0,38294747099493937 |
| Features | 25 requeridas, más ingeniería distinta por rama | Pipeline E y features originales adicionales |
| Calendario | RF agrega día semanal, fin de semana, senos/cosenos y fecha ordinal | Día semanal; descarta execution_week |
| Ingeniería CatBoost activa | Historial, recursos/proporciones, logaritmos e interacciones de costo por GPU | empty_job, presencia de agente/organización y clipping de costo |
| Categóricas | Calendario parcial como categorías; representación distinta por rama | Categorías nativas; categorías raras (<20) y nuevas a otros |
| Assigned/estado/fecha de estado | No forman parte de las 25 entradas | Excluidos del modelo |
| Entrenamiento declarado | 38.274 filas | 38.274 filas para holdout |
| Fuente train | Hash idéntico al train de este repo | Mismo hash |
| CV reportada | 85,30%, según metadata; protocolo no auditable con el predictor solo | 84,83%, folds agrupados y calibración interna auditada |
| Holdout interno | No hay resultado comparable en el archivo entregado; declara test_accessed=False | 85,09% según evaluación final guardada |
| Test oculto | 79,0% BA, 0,894 AUC según captura | No evaluado |

La primera entrega NO es el promedio ni el stacking de nuestro experimento: usa otro RandomForest, CatBoost y otra ingeniería. Nuestro ensayo negativo de ensembles no demuestra que esta entrega sea peor.

## Qué puede explicar la caída

**Mayor capacidad y sensibilidad a distribución, como hipótesis.** Un RF con hojas de una fila y sin límite de profundidad puede aprender patrones muy específicos; la rama CatBoost también tiene bastante capacidad. Las features temporales e interacciones pueden ayudar en una partición aleatoria y trasladarse peor a otra población. Son motivos para investigar, no prueba de la causa: sin sus curvas, OOF y gaps no puede atribuirse el 79% a sobreajuste de forma concluyente.

**Protocolo de validación desconocido.** La metadata dice selected development CV y advierte que la replicación también es desarrollo; reporta 85,30% y una ganancia frente a D52 de solo 0,000484 (0,0484 puntos). No contiene las particiones ni permite auditar cómo se seleccionaron features, pesos o umbral. La caída descriptiva CV→competencia es aproximadamente 6,30 puntos; no es un gap train-validación ni identifica su causa.

**Umbral.** 0,49 y 0,383 no se comparan directamente entre modelos distintos. CatBoost balanceado y la mezcla pueden tener una escala de probabilidades distinta. No hay evidencia para cambiar el umbral a partir del ranking. El AUC 0,894 indica que el score conserva capacidad de ordenar, pero sin curva ROC o matriz del test oculto no sabemos cuánto podría recuperar un umbral distinto. Tampoco se explica toda la caída por calibración solo con ese número.

**Compatibilidad.** Las métricas publicadas sugieren que se ejecutó una evaluación, no un fallo puro de importación. Se desconoce si hubo adaptación del evaluador. La falta de catboost en nuestro entorno no prueba que faltara en el de los profesores. El empaquetado por pickle y versiones exactas agrega requisitos de ejecución; tampoco explica por sí solo la BA publicada.

No se detectó uso directo de target/estado/assigned en las entradas activas de la primera entrega. El archivo define clases con otras variantes, incluyendo target encoding, pero la configuración activa es calendar/base y engineered: no corresponde acusar fuga por métodos presentes pero no utilizados. El predictor final solo permite inferencia; eso no prueba que su entrenamiento histórico haya sido independiente.

## Diferencia de esquema que hay que resolver

Las entradas originales que existen en nuestro conjunto y no están entre las 25 requeridas por la entrega son:

- assigned_gpu_type: no se usa como feature en HistGB, pero transform() intenta descartarla sin errors='ignore', por lo que exige su presencia accidentalmente.
- resource_modifications y queue_wait_time: features activas de HistGB; no pueden reconstruirse de forma fiable con las otras entradas. Debe confirmarse también su disponibilidad al momento de predecir.
- total_gpu_hours y total_processes: features activas; podrían derivarse de sumas si el esquema oficial garantiza esa definición.
- high_value_client: feature activa; no asumir un corte aproximado para reconstruirla.

El predictor enviado acepta columnas extras y selecciona sus 25 requeridas. Por tanto, que requiera 25 NO prueba que el test tenga solamente 25. Hace falta la consigna/esquema oficial. Si solo hay 25, el HistGB actual no es desplegable tal cual. Arreglar el descarte de assigned es un cambio de robustez de interfaz; eliminar features activas o sustituirlas por cero modifica el predictor y exige entrenamiento/validación coherentes.

## ¿Tiene chances de mejorar?

Sí, por su menor complejidad, tratamiento de categorías raras, exclusión de assigned y validación auditada. La evidencia propia es consistente: CV 84,83% y holdout interno 85,09%. Pero la primera entrega también reportaba CV cercana al 85% y obtuvo 79% oculto. No debe inferirse una mejora de seis puntos comparando nuestro holdout con la competencia: son datos diferentes.

HistGB también mostró problemas de distribución: 77,47% en 2016 excluido, 75,56% en PRT no visto y 51,47% al entrenar solo PRT y evaluar el resto. Su gap sin assigned fue 4,17 puntos y el inflado por copias, 1,82. Es un candidato con evidencia local, no una solución garantizada al cambio de distribución.

## Próximo paso concreto

1. Confirmar esquema y momento de disponibilidad de variables del evaluador; revisar cualquier mensaje técnico de la entrega.
2. Obtener, si existen, OOF, folds, curvas y pruebas temporales/regionales del desarrollo D53 de la primera entrega.
3. Comparar ambos procedimientos en train con los mismos folds agrupados y calibración interna, usando features permitidas. No evaluar el predictor ya entrenado en train como si fuera validación: vio esas filas. Para una comparación nueva haría falta reconstruir su entrenamiento o disponer de sus OOF verificables.
4. Si el esquema admite HistGB y la evidencia comparable favorece la alternativa, empaquetarlo y probar la interfaz en el entorno esperado antes de enviar. Si hay que restringirlo a 25 columnas, tratarlo como una variante nueva y evaluarla únicamente con CV de train; el resultado de holdout existente no se hereda.
5. No volver a usar el holdout interno ya consultado como una confirmación independiente de nuevas variantes. El ranking de la primera entrega también es feedback de selección: se puede usar según las reglas de la competencia, declarando esa adaptación y evitando prometer una estimación insesgada de la segunda entrega.

La conclusión actual es condicional: vale la pena considerar HistGB, pero primero hay una diferencia concreta de entradas que resolver. No alcanza con reemplazar el .py ni bajar el umbral de la primera entrega.

Artefactos: `experiments/comparacion_entrega_v1/estructura.json` e `inspeccionar.py` permiten revisar la estructura extraída del archivo recibido. El informe de holdout existente está en `reports/evaluacion_final_histgb.md`.
