# Decisiones del EDA

Registro de lo acordado en la discusión del EDA, paso por paso. Cada decisión indica dónde está implementada.

## Paso 1: estructura y calidad
- **Nulos de `orchestration_agent_id` y `organization_id`:** son "no aplica", no datos faltantes. Se tratan como categoría propia `(nulo)`. → `features.py`
- **Nulos de `client_region` (19) y `num_auxiliary_processes` (1):** se rellenan con `(nulo)` y 0. No se descartan filas porque el predictor tiene que responder por cada fila que reciba. → `features.py`
- **Duplicados:** se conservan. 12.567 filas (32,8%) son copia exacta de otra; no hay identificador de reserva y los grupos grandes son reservas en bloque canceladas juntas.
- **Fuga:** `execution_status` y `execution_status_date` quedan fuera. La fecha sola acierta el objetivo en el 99,96% de las filas. → `config.LEAK_COLS`

## Paso 2: variable objetivo
- **Sin rebalanceo:** 58,3% / 41,7% es un desbalance leve; se ajusta el punto de corte más adelante.
- **Evaluación:** reportar siempre cancelados detectados, completados detectados y su promedio (balanced accuracy).
- **Pendiente (paso 5):** comparar los 428 `no_show` con el resto de los cancelados.

## Paso 3: variables numéricas
- **Valores extremos:** no se eliminan filas.
- **`estimated_cost_per_hour`:** se recorta al percentil 99 de train (hay un único 270 contra un segundo máximo de 18,6). → `features.py`
- **Reservas vacías:** nueva columna `empty_job` = 0 horas de GPU y costo 0 (157 filas, 7,6% de cancelación). → `features.py`
- **Pendiente (paso 5):** en las variables casi siempre cero, ver si alcanza con un indicador "mayor que cero".

## Paso 4: variables categóricas
- **Categorías raras:** toda categoría con menos de 20 filas (en el entrenamiento de cada fold) pasa a `otros`; las que aparecen por primera vez al predecir, también. → `features.py`
- **Etiquetas de nulo:** `sin_agente`, `sin_organizacion`, `region_desconocida`. → `features.NULL_LABELS`
- **Codificación dentro del pipeline:** `pipeline.make_encoder` codifica las categóricas (one-hot por defecto, ordinal como alternativa) como paso del `Pipeline`, ajustado en cada fold. La salida es toda numérica y sin nulos. → `pipeline.py`
- **`CN` no se unifica con `CHN`:** existen los dos códigos (234 y 448 filas) y se comportan distinto (35% vs 52,5% de cancelación; anticipación mediana 81 vs 31). No hay `CAN` en train, así que `CN` probablemente sea Canadá. Quedan separados.
- **`client_type` y `request_channel`:** se conservan las dos.
- **Pendiente (paso 5):** ver si para `organization_id` alcanza con el indicador `has_organization`.

## Paso 5.1: numéricas contra el objetivo
- **Doble lectura:** toda relación se mira sobre todo train y sin `locked_allocation`. `queue_wait_time` y el pico de costo 3–3,5 parecían señales fuertes y desaparecen sin `locked_allocation`.
- **Numéricas con su valor original:** sin indicadores ni tramos. En `previous_job_cancellations`, 1 y 2 son opuestos (90% vs 16% de cancelación), así que un indicador "mayor que cero" perdería información.
- **Relaciones no lineales:** anticipación (cae a 8% por encima de 365, sin `locked_allocation`) y cancelaciones previas. Un modelo lineal necesitaría tramos; los árboles no.
- **`queue_wait_time`:** se conserva aunque su señal propia sea casi nula.
- **`total_gpu_hours` y `total_processes`:** se conservan (se revierte la idea inicial de descartarlas). → `features.py`

## Paso 5.2: categóricas contra el objetivo
- **`organization_id`:** se usan el indicador `has_organization` y el identificador agrupado. El indicador captura casi todo (11% vs 32% sin `locked_allocation`), pero 3 organizaciones cancelan 30–40%.
- **`orchestration_agent_id` y `client_region`:** son las categóricas con más señal (tasas de 0% a 84% entre agentes) y con más riesgo de sobreajuste; la protección es el umbral de 20 filas.
- **`reserved_storage_units = 1`:** 0% de cancelación en 916 filas. Lo aprende el modelo; no se fija una regla a mano.
- **`high_value_client`:** se conserva aunque casi no tenga señal (32% vs 30%).
- **`assigned_gpu_type` y GPU cambiada:** se prueban los dos casos, con y sin, al modelar (`Preprocessor(use_assigned=...)`). Con GPU cambiada la cancelación es 6% contra 34%; sospecha de que se registra tarde.

## Paso 5.3: `no_show`
- **Perfil:** los 428 `no_show` se parecen a los completados (anticipación mediana 28, 54% sin pedidos especiales, 0,2% `locked_allocation`), no al resto de los cancelados. Solo `PRT` (52%) y la ausencia de almacenamiento los acercan a los cancelados.
- **Sin tratamiento especial:** al predecir no se sabe cuáles son (`execution_status` no está disponible). Es un límite conocido: como máximo cuestan ~1,3 puntos de balanced accuracy.
- **Evaluación:** `validation.report` informa cancelados detectados, completados detectados y `no_show` detectados. `execution_status` se usa solo para ese desglose. → `validation.py`
- **Evidencia sobre `assigned_gpu_type`:** la GPU cambiada aparece en 14% de los `no_show` y 15% de los completados, contra 1% de los `canceled`. La reasignación ocurre cerca del día de ejecución, sobre las reservas que siguen vigentes; la variable indica "seguía viva", no intención de cancelar.

## Paso 6: relaciones entre variables
- **Redundancia:** no se elimina ninguna variable. Los pares que se pisan (totales, `high_value_client` con costo, `returning_client` con trabajos previos, `client_type` con canal) no dañan a los árboles. Para un modelo lineal, sacar los totales y `high_value_client`.
- **Agente:** cada agente principal trabaja con un solo tipo de cliente; es una versión más fina de `client_type`.
- **Costo:** se explica en buena parte por tipo de GPU, mes, cantidad de procesos y tipo de cliente; por eso tiene poca señal propia.
- **Interacción región × tipo de cliente (sin `locked_allocation`):** `research_consortium` cancela 58% en `PRT` y 1% en el resto; `managed_account`, 48% y 2%. En `cloud_marketplace` la región casi no importa (42% vs 37%).
- **Interacción anticipación × pedidos especiales:** sin pedidos la cancelación sube de 16% a 58% con la anticipación; con pedidos se estanca cerca de 25%.
- **Columna `client_type` × PRT:** NO se agrega ahora. Queda como candidata para probar al modelar.
- **Criterio para toda columna nueva:** entra solo si mejora la validación cruzada más que la variación entre folds. Elegir columnas mirando el objetivo arrastra casualidad; `test.csv` queda sin tocar como control final.
- **Sin más combinaciones a mano** por ahora.
- **Referencia de subajuste:** un modelo lineal simple; los modelos de árboles tienen que superarlo con margen.

## Test desconocido
- **Supuesto de trabajo:** no se sabe qué distribución tiene el test de los profesores ni se puede consultar; es parte del problema.
- **Evaluación por subgrupos:** `PRT` y resto, por año, con y sin `locked_allocation`.
- **Validación dura adicional:** dejar afuera un año completo y dejar afuera una región completa.
- **Sin valores de train fijados a mano:** refuerza la decisión de no agregar la columna `client_type` × PRT.
- **Punto de corte:** fuera de `PRT` la cancelación es ~27% (41,7% en todo train); revisar al modelar qué tan sensible es el punto de corte a ese cambio.
