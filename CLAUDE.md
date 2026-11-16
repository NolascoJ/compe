# TP3 – Competencia ML (ITBA, Grupo 9)

## Objetivo
Clasificación binaria: predecir `canceled_job` (1 = el trabajo de GPU reservado se cancela).
**Métrica de la competencia: Balanced Accuracy** (promedio del recall de cada clase).

## Regla principal: sin fuga de datos, sin sobreajuste, sin subajuste, robusto a un test desconocido
Es el requisito más importante del repo. Todo cambio tiene que cumplirlo.

**Fuga de datos**
- `execution_status` y `execution_status_date` nunca entran a un modelo (`config.LEAK_COLS`).
- Todo lo que se aprende de los datos (categorías frecuentes, topes, medias, codificaciones por tasa de cancelación, escalados) se ajusta **dentro de cada fold**, solo con la parte de entrenamiento. Por eso el preprocesamiento es un transformador de sklearn dentro del pipeline (`features.Preprocessor`); nunca se transforma el dataset completo antes de partirlo.
- Los folds se arman con `validation.make_folds`, que agrupa por variables de entrada: dos filas con el mismo X nunca quedan una en entrenamiento y otra en validación.
- `test.csv` no se usa para EDA, selección de variables, ajuste de hiperparámetros ni elección del punto de corte. Se evalúa una sola vez, al final.
- Ninguna variable puede depender de otras filas del conjunto que se predice (por ejemplo, contar copias dentro del lote), salvo que se decida explícitamente y se justifique.

**Sobreajuste**
- Toda métrica que se reporta es fuera de fold. Nunca se reporta el rendimiento sobre los datos de entrenamiento como resultado.
- Junto a cada resultado se mira la diferencia entre entrenamiento y validación y la dispersión entre folds. Una diferencia grande es sobreajuste y hay que regularizar.
- El punto de corte elegido sobre predicciones fuera de fold es levemente optimista: decirlo cuando se reporta.
- No elegir entre muchas variantes por diferencias menores que la dispersión entre folds.
- Toda columna nueva (combinaciones, indicadores, codificaciones) entra solo si mejora la validación cruzada más que la dispersión entre folds. Elegir columnas mirando el objetivo arrastra casualidad.

**Subajuste**
- Todo modelo se compara contra referencias simples (predecir siempre una clase: 0,50; reglas de una variable; un modelo lineal simple). Si no las supera con margen, está subajustado.
- Si entrenamiento y validación dan parecido y ambos bajos, falta capacidad o faltan variables.

**Test desconocido (cambio de distribución)**
- No se sabe cómo está armado el test de los profesores y no se puede averiguar: es parte del problema. Puede no parecerse a train (por ejemplo, sin reservas de `PRT` o casi sin `locked_allocation`).
- El modelo no puede depender de que el test se parezca a train. Toda evaluación reporta la balanced accuracy también por subgrupos: `PRT` y resto de regiones, cada año, con y sin `locked_allocation`.
- Además de la validación cruzada normal, se hace una validación más dura: entrenar con dos años y validar en el tercero, y entrenar sin una región y validar en ella. Una caída grande indica que el modelo memoriza.
- No fijar a mano valores específicos de train (una región, un agente) en columnas o reglas.
- Entre dos modelos con resultado parecido, preferir el más parejo entre subgrupos.

## Estructura del repo
- `data/raw/`: CSV originales, no se modifican.
- `src/compe/`: `config.py` (columnas y rutas), `data.py` (carga y grupos), `features.py` (preprocesamiento), `pipeline.py` (variantes de preprocesamiento por tipo de modelo), `models.py` (registro de los 10 modelos y sus espacios de búsqueda), `validation.py` (folds, punto de corte y métricas), `run.py` (ejecutor por modelo: `python -m compe.run <modelo>`), `resumen.py` (tabla comparativa), `plots.py` (estilo de gráficos).
- `notebooks/`: `01_eda` (pareada `.py`/`.ipynb` con jupytext).
- `reports/eda_decisiones.md`: decisiones acordadas en el EDA, paso por paso. `reports/figures/`: gráficos.
- `experiments/<modelo>/`: resultados de cada modelo (curva de validación, búsquedas, mejores hiperparámetros, métricas, subgrupos, predicciones fuera de fold). `experiments/_prueba/` es para corridas cortas y no se versiona.
- `TODO.md`: plan de modelado, decisiones tomadas, modelos fuera de alcance y cómo se corre cada cosa.
- `models/`: vacía por ahora.

## Forma de trabajo
El EDA se hace como discusión, un paso por vez: hallazgos, interpretación y propuestas; el usuario decide antes de avanzar. No adelantarse a modelar sin que lo pida.

## Entregables y fechas
- **El predictor tiene que poder correr en Google Colab.** Desarrollar con las mismas versiones que Colab (Python 3.12, pandas 2.x, scikit-learn de Colab), fijadas en `pyproject.toml`. No usar funciones que solo existan en versiones más nuevas. Probar el predictor en una notebook de Colab antes de cada envío.
- **Función predictor** independiente: recibe X (columnas del dataset) y devuelve y (0/1 por fila).
- Notebook + presentación (formato habitual de TP). Se entregan el día anterior a la defensa del TP3.
- Envíos del predictor: **04/11, 11/11 y 18/11 de 2026, antes de las 16:00**. Cuenta el último (18/11).
- Competition Score = 90% Model Score (envío del 18/11) + 10% nota de presentación.
- Premios: Top 1 en Model Score y Top 4 en Competition Score quedan eximidos del final. Los ganadores presentan el 25/11.

## Datos (`data/raw/`, sacados de `dataset_dividido_grupo9.zip`)
| Archivo | Contenido |
|---|---|
| `train.csv` | 38.274 filas × 34 columnas. 58,3% clase 0 / 41,7% clase 1 |
| `test.csv` | 9.568 filas. **Holdout interno con etiqueta**, NO es el test de la competencia |
| `indices_split.csv` | `fila_original`, `grupo_duplicados`, `particion` (las filas `train` siguen el orden de `train.csv`) |
| `split_metadata.json`, `README_SPLIT.md` | Semilla 42, split 80/20 estratificado por grupos de filas idénticas, hashes |

El **test de la competencia lo tienen los profesores**: corren nuestro predictor sobre un X oculto.
Reglas del README del split:
- Elegir el modelo y el umbral solo con CV interna sobre `train.csv`.
- Usar `grupo_duplicados` para armar los folds (StratifiedGroupKFold), nunca como feature.
- Usar `test.csv` una sola vez, para la comparación final.
- Para la entrega final, reentrenar con train+test.

## Leakage: excluir siempre
- `execution_status` (`completed`/`canceled`/`no_show`): mapea 1:1 al target (no_show cuenta como 1).
- `execution_status_date`: en los cancelados es anterior a la fecha de ejecución (mediana −57 días); en los completados es 0 a 4 días después. Revela el target.
- El predictor tiene que **descartar adentro** `canceled_job`, `execution_status` y `execution_status_date` si vienen en el X.
- Variable dudosa: `assigned_gpu_type` y el indicador de GPU cambiada. Se registra tarde: la GPU cambiada aparece en 15% de los completados y 14% de los `no_show`, contra 1% de los `canceled`, o sea que indica que la reserva seguía vigente el día de ejecución. Decisión: evaluar todo modelo con y sin (`Preprocessor(use_assigned=...)`) y declararlo en la presentación.

## Diccionario de columnas
El dataset parece ser *Hotel Booking Demand* (City Hotel) disfrazado. La equivalencia ayuda a interpretar y a armar features.

| Columna | Tipo | Notas | Equivalente probable |
|---|---|---|---|
| hours_before_execution | int 0–629 | La más correlacionada (+0,29) | lead_time (días) |
| execution_year/month/week/day | int | 2015-07 → 2017-08 | arrival_date_* |
| peak_gpu_hours / offpeak_gpu_hours | int, múltiplos de 8 | | weekend / week nights ×8 |
| total_gpu_hours | int | = peak + offpeak (redundante) | |
| num_primary/auxiliary/background_processes | int | auxiliary tiene 1 nulo | adults / children / babies |
| total_processes | float | = suma de los tres (redundante) | |
| service_tier | cat 4 | standard / basic / priority / enterprise | meal |
| client_region | cat 143 | PRT 39% de las filas y 65% de cancelación. Hay categorías raras y no vistas en test | country |
| client_type | cat 8 | research_consortium 68% de cancelación | market_segment |
| request_channel | cat 5 | broker_platform 45% de cancelación | distribution_channel |
| returning_client | bin | | is_repeated_guest |
| previous_job_cancellations | int 0–21 | +0,18 | previous_cancellations |
| successful_previous_jobs | int 0–70 | | previous_bookings_not_canceled |
| reserved_gpu_type / assigned_gpu_type | cat | ver leakage | reserved / assigned room |
| resource_modifications | int 0–18 | −0,15 | booking_changes |
| reservation_commitment | cat 3 | **locked_allocation → 99,8% de cancelación** (6.148 filas) | deposit_type (Non Refund) |
| orchestration_agent_id | float, 10% nulo | Es un ID: tratarlo como categoría | agent |
| organization_id | float, 95% nulo | Es un ID. Que sea nulo ya es señal (20% vs 43%) | company |
| queue_wait_time | int 0–391 | | days_in_waiting_list |
| workload_profile | cat 4 | | customer_type |
| estimated_cost_per_hour | float 0–270 | 574 ceros, 1 outlier (270) | adr |
| reserved_storage_units | bin | | required_car_parking_spaces |
| special_resource_requests | int 0–5 | −0,29, cuanto más pedidos menos cancelación | total_of_special_requests |
| high_value_client | bin | Derivada: ≈ cost_per_hour > ~6,6 | |

## Datos rápidos
- 32,8% de las filas de train son duplicados exactos (hasta 180 copias en un grupo). Por eso los folds van por grupos.
- La tasa de cancelación es bastante estable por mes (36–46%) y por año.
- Split aleatorio por grupos, no temporal.

## Entorno
- `.venv/` creado con `uv` (pandas 3.1, scikit-learn 1.9). Correr con `.venv/bin/python`.
- Instalar lo que haga falta con `uv pip install --python .venv/bin/python <paquete>`.
