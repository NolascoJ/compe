> Plan histórico: las casillas y la frase «Nada de esto está ejecutado» reflejan el inicio del trabajo. Para el estado de entrega actual, consultar `docs/ENTREGA.md` y `docs/CONTEXTO.md`.

# TODO · Comparación de clasificadores

Plan para probar 10 clasificadores (de una lista original de 17; 7 quedan fuera por ahora, ver sección 7). **Nada de esto está ejecutado.** Primero se cierran las decisiones de la sección 0.

**Protocolo pedido**
- Métrica principal: balanced accuracy.
- Métricas secundarias: ROC-AUC, F1, precision y recall.
- Validación: Stratified K-Fold, 5 folds.
- Punto de corte: optimizado sobre predicciones fuera de fold (OOF).

---

## 0. Decisiones a cerrar antes de empezar

- [x] **D1 · Folds.** DECIDIDO: `StratifiedGroupKFold` de 5 folds agrupando por variables de entrada (`validation.make_folds`) como validación principal. Se reporta además el resultado con Stratified K-Fold común como número de control: si un modelo sube mucho con el común, depende de memorizar copias. El archivo original (47.842 filas) tiene 33,0% de copias, igual que train: los duplicados vienen de origen.
- [x] **D2 · Entorno.** DECIDIDO: recrear el entorno con Python 3.12 y versiones iguales a las de Google Colab, fijadas en `pyproject.toml`. **Requisito: el predictor tiene que correr en Colab.** Pendiente al ejecutar: obtener las versiones reales de Colab (`pip freeze` en una notebook de Colab), instalar `libomp` con Homebrew (pedir confirmación) y compilar Annoy. Nota: Colab trae XGBoost y LightGBM preinstalados; CatBoost y Annoy no, habría que instalarlos con `pip` desde la notebook (ver D7).
- [x] **D3 · SVM con kernel.** DECIDIDO: quedan fuera por ahora (ver sección 7). `LinearSVC` sí se prueba.
- [x] **D4 · Presupuesto de búsqueda.** DECIDIDO: ajuste completo de los 10 modelos con el protocolo de la sección 4 (sin filtro previo).
- [x] **D5 · Desbalance.** DECIDIDO: `class_weight` (o su equivalente) es un hiperparámetro más de la búsqueda; el punto de corte se optimiza siempre sobre las predicciones fuera de fold.
- [ ] **D6 · `assigned_gpu_type`.** Acordado: probar con y sin. Propuesta: correr los 10 en las dos variantes en la etapa 1, y seguir a la etapa 2 con la variante que se elija.
- [x] **D7 · Librerías en el predictor final.** DECIDIDO: por ahora solo modelos que Colab trae instalados (scikit-learn, XGBoost, LightGBM). CatBoost y Annoy quedan fuera (ver sección 7).

---

## 1. Infraestructura

- [x] `src/compe/models.py`: registro de los 10 modelos (estimador, variante de preprocesamiento, espacio de búsqueda, curva de validación, criterio de simplicidad).
- [x] `src/compe/pipeline.py`: variantes A, B, D y E.
- [x] `src/compe/validation.py`: puntuación por probabilidad o función de decisión; punto de corte OOF; balanced accuracy, ROC-AUC, F1, precision, recall; entrenamiento vs validación; dispersión entre folds; desglose por subgrupos y `no_show`.
- [x] `src/compe/run.py`: ejecutor por modelo (ver "Cómo se corre").
- [x] `src/compe/resumen.py`: tabla comparativa de los modelos corridos.
- [ ] Prueba corta (`--rapido`) de cada modelo. Hecha y correcta solo en Ridge; interrumpida a pedido en el resto.
- [ ] Recrear el entorno con las versiones de Colab (D2). Falta: salida de `pip freeze` de Colab.
- [ ] Instalar `libomp` (`brew install libomp`) y probar XGBoost y LightGBM, que están escritos pero **sin probar**.
- [ ] Prueba de tiempos antes de lanzar las búsquedas largas (Random Forest, Bagging, Gradient Boosting, KNN).
- [ ] Validación anidada, y validación dejando afuera un año o una región (pasos 6 y 7 del protocolo): se escriben cuando haya finalistas.

### Cómo se corre

```sh
.venv/bin/python -m compe.run --listar                       # modelos disponibles
.venv/bin/python -m compe.run ridge                          # protocolo completo de un modelo
.venv/bin/python -m compe.run ridge --etapa curva            # una sola etapa
.venv/bin/python -m compe.run ridge --rapido                 # prueba corta, guarda en experiments/_prueba/
.venv/bin/python -m compe.resumen                            # tabla comparativa -> experiments/resumen.csv
```

Etapas, en orden: `base`, `curva`, `busqueda`, `refinar`, `confirmar`, `final`. La búsqueda se guarda combinación por combinación; si se corta, al relanzar sigue donde quedó.

### Qué queda en `experiments/<modelo>/`

| Archivo | Contenido |
|---|---|
| `base.json` | métricas con los valores por defecto |
| `curva_validacion.csv` y `.png` | entrenamiento vs validación según el hiperparámetro de complejidad |
| `busqueda.csv` | búsqueda amplia: una fila por combinación, con todas las métricas |
| `refinamiento.csv` | búsqueda estrecha alrededor del 10% mejor |
| `confirmacion.csv` | las 3 mejores combinaciones, promediadas sobre 3 particiones distintas |
| `mejores_parametros.json` | hiperparámetros elegidos por la regla de selección |
| `metricas.json` | métricas finales, control con K-Fold común y resultado sin `assigned_gpu_type` |
| `subgrupos.csv` | balanced accuracy por subgrupo (`PRT` y resto, por año, sin `locked_allocation`) y `no_show` detectados |
| `oof.npy`, `oof_sin_assigned.npy` | predicciones fuera de fold del modelo elegido, en el orden de `train.csv` |

Diferencias respecto del plan original:
- HistGradientBoosting no usa parada temprana (su partición interna separa filas idénticas); `max_iter` se busca como un hiperparámetro más.
- XGBoost y LightGBM usan parada temprana con una partición interna del entrenamiento que mantiene juntas las filas idénticas, y después reentrenan con todo el entrenamiento del fold.
- El límite de diferencia entrenamiento-validación es un parámetro (`--max-gap`, 0,05 por defecto) hasta fijarlo con las curvas.

---

## 2. Variantes de preprocesamiento

Todas parten de `features.Preprocessor` (fugas afuera, categorías raras a `otros`, tope de costo, `empty_job`) y se ajustan dentro de cada fold.

| Variante | Pasos adicionales | Modelos |
|---|---|---|
| **A · lineal** | one-hot; logaritmo en las numéricas con cola larga; estandarización; sin totales ni `high_value_client` | 1, 2, 5 |
| **B · distancias** | igual que A (la escala es crítica) | 4 |
| **D · árboles** | codificación ordinal; sin escalado | 10, 11, 12 |
| **E · categóricas nativas** | categóricas pasadas como tales al modelo; sin escalado | 13, 15 (14 si la versión lo soporta; si no, D) |

- [ ] Implementar A, B, D y E como opciones de `make_pipeline`.
- [ ] Verificar que cada variante devuelve una matriz numérica sin nulos (salvo E).

---

## 3. Etapa 1 · Los 10 con valores por defecto

Objetivo: tener una tabla comparable y descartar lo que no compite. Cada modelo se corre con y sin `assigned_gpu_type` (D6).

| # | Familia | Clasificador | Variante | Puntuación | Costo |
|---|---|---|---|---|---|
| 1 | Lineales | LogisticRegression | A | probabilidad | bajo |
| 2 | Lineales | RidgeClassifier | A | decisión | bajo |
| 4 | Distancias | KNeighborsClassifier | B | probabilidad | medio |
| 5 | SVM | LinearSVC | A | decisión | bajo |
| 10 | Bagging | RandomForestClassifier | D | probabilidad | medio |
| 11 | Bagging | BaggingClassifier | D | probabilidad | medio |
| 12 | Boosting | GradientBoostingClassifier | D | probabilidad | medio |
| 13 | Boosting | HistGradientBoostingClassifier | E | probabilidad | bajo |
| 14 | Boosting | XGBClassifier | E o D | probabilidad | bajo |
| 15 | Boosting | LGBMClassifier | E | probabilidad | bajo |

- [ ] 1 LogisticRegression
- [ ] 2 RidgeClassifier
- [ ] 4 KNeighborsClassifier
- [ ] 5 LinearSVC
- [ ] 10 RandomForestClassifier
- [ ] 11 BaggingClassifier
- [ ] 12 GradientBoostingClassifier
- [ ] 13 HistGradientBoostingClassifier
- [ ] 14 XGBClassifier
- [ ] 15 LGBMClassifier
- [ ] Tabla de resultados de la etapa 1 y filtro: pasan a la etapa 2 los que queden a menos de ~2 puntos del mejor, más el mejor de cada familia para la presentación.

---

## 4. Ajuste de hiperparámetros (protocolo y rangos APROBADOS el 9/11)

Objetivo: encontrar el mejor punto de cada modelo sin sobreajuste ni subajuste. Se ajustan los 10, sin filtro previo.

### 4.1 Protocolo (igual para todos los modelos)

- [ ] **Paso 1 · Curvas de validación.** Para cada modelo, barrer su hiperparámetro de complejidad y graficar balanced accuracy en entrenamiento y en validación. Zona de subajuste: las dos bajas. Zona de sobreajuste: entrenamiento sube y validación se estanca o baja. Sirve para fijar los rangos de la búsqueda.
- [ ] **Paso 2 · Búsqueda amplia.** Grilla completa en los modelos de pocos hiperparámetros; búsqueda aleatoria en los demás. Cada combinación se evalúa en los 5 folds agrupados, con su punto de corte OOF.
- [ ] **Paso 3 · Refinamiento.** Segunda búsqueda con rangos estrechos alrededor del 10% mejor del paso 2.
- [ ] **Paso 4 · Regla de selección.** No se elige el máximo a secas:
  - se descartan las combinaciones con diferencia entrenamiento-validación por encima del límite que se fije al ver las curvas;
  - entre las que quedan a menos de un error estándar de la mejor, se elige la más simple (más regularizada).
- [ ] **Paso 5 · Confirmación con repeticiones.** Las 3 mejores combinaciones de cada modelo se reevalúan con 3 repeticiones de los 5 folds (15 ajustes, semillas distintas), para que la elección no dependa de una partición.
- [ ] **Paso 6 · Validación anidada para los finalistas.** Para los 2 o 3 mejores modelos: 5 folds externos para medir y 3 internos para elegir hiperparámetros y punto de corte. Da la estimación sin el optimismo de haber buscado sobre los mismos folds.
- [ ] **Paso 7 · Robustez.** Resultados por subgrupo (`PRT` y resto, por año, con y sin `locked_allocation`), validación dejando afuera un año y dejando afuera una región, y número de control con K-Fold común.
- [ ] **Paso 8 · `test.csv`, una sola vez**, para el modelo elegido.

Reglas fijas: parada temprana de los boosting con una partición interna del entrenamiento de cada fold, nunca con el fold de validación; se registra cada combinación probada, no solo la mejor; semilla 42.

### 4.2 Hiperparámetros por modelo

La columna "complejidad" indica hacia dónde sobreajusta.

| # | Modelo | Hiperparámetros y rangos | Complejidad | Búsqueda |
|---|---|---|---|---|
| 1 | LogisticRegression | `C` 1e-4 a 1e2 (log); `penalty` l1, l2, elasticnet; `l1_ratio` 0,1 a 0,9; `class_weight` ninguno o balanced | `C` alto = menos regularización | aleatoria, 60 |
| 2 | RidgeClassifier | `alpha` 1e-3 a 1e4 (log, 25 valores); `class_weight` | `alpha` bajo = menos regularización | grilla, 50 |
| 4 | KNeighborsClassifier | `n_neighbors` 3 a 301 (15 valores); `weights` uniform, distance; `p` 1, 2 | pocos vecinos = sobreajuste | grilla, 60 |
| 5 | LinearSVC | `C` 1e-4 a 1e1 (log); `penalty` l1, l2; `loss`; `class_weight` | `C` alto = menos regularización | aleatoria, 40 |
| 10 | RandomForestClassifier | `n_estimators` 300 a 1000; `max_depth` 6 a 30 o sin límite; `min_samples_leaf` 1 a 50; `min_samples_split` 2 a 50; `max_features` sqrt, log2, 0,2 a 0,6; `max_samples` 0,5 a 1; `class_weight` | profundidad alta y hojas chicas | aleatoria, 80 |
| 11 | BaggingClassifier | `n_estimators` 100 a 500; `max_samples` 0,4 a 1; `max_features` 0,4 a 1; árbol base: `max_depth` 6 a 30, `min_samples_leaf` 1 a 50 | profundidad del árbol base | aleatoria, 60 |
| 12 | GradientBoostingClassifier | `learning_rate` 0,01 a 0,2 (log); `n_estimators` 100 a 800; `max_depth` 2 a 6; `subsample` 0,5 a 1; `min_samples_leaf` 5 a 100; `max_features` 0,3 a 1 | más árboles, más profundos, tasa alta | aleatoria, 60 |
| 13 | HistGradientBoostingClassifier | `learning_rate` 0,01 a 0,2 (log); `max_iter` hasta 2000 con parada temprana; `max_leaf_nodes` 7 a 127; `max_depth` 3 a 12 o sin límite; `min_samples_leaf` 10 a 300; `l2_regularization` 0 a 10; `max_features` 0,4 a 1; `class_weight` | más hojas, hojas chicas, poca regularización | aleatoria, 120 |
| 14 | XGBClassifier | `learning_rate` 0,01 a 0,2 (log); `n_estimators` hasta 3000 con parada temprana; `max_depth` 3 a 10; `min_child_weight` 1 a 30; `subsample` 0,5 a 1; `colsample_bytree` 0,4 a 1; `gamma` 0 a 5; `reg_lambda` 1e-2 a 30 (log); `reg_alpha` 0 a 5; `scale_pos_weight` 1 o proporción de clases | profundidad alta, `min_child_weight` bajo | aleatoria, 150 |
| 15 | LGBMClassifier | `learning_rate` 0,01 a 0,2 (log); `n_estimators` hasta 3000 con parada temprana; `num_leaves` 7 a 255; `max_depth` 3 a 12 o sin límite; `min_child_samples` 10 a 300; `subsample` 0,5 a 1; `colsample_bytree` 0,4 a 1; `reg_lambda` 1e-2 a 30 (log); `reg_alpha` 0 a 5; `min_split_gain` 0 a 1; `cat_smooth` 1 a 50; `is_unbalance` | más hojas, `min_child_samples` bajo | aleatoria, 150 |

### 4.3 Hiperparámetros de preprocesamiento (solo en los 3 mejores modelos)

- [ ] `min_count` de categorías raras: 10, 20, 50, 100.
- [ ] Codificación de las categóricas de muchos valores: one-hot, ordinal, o por tasa de cancelación calculada dentro de cada fold.
- [ ] Con y sin `assigned_gpu_type`.
- [ ] Columna candidata `client_type` × `PRT`.
- [ ] Tope del costo: percentil 99 o sin tope.

### 4.4 Tareas

- [ ] Curvas de validación de los 10 modelos.
- [ ] Búsqueda amplia y refinamiento: 1 LogisticRegression
- [ ] Búsqueda amplia y refinamiento: 2 RidgeClassifier
- [ ] Búsqueda amplia y refinamiento: 4 KNeighborsClassifier
- [ ] Búsqueda amplia y refinamiento: 5 LinearSVC
- [ ] Búsqueda amplia y refinamiento: 10 RandomForestClassifier
- [ ] Búsqueda amplia y refinamiento: 11 BaggingClassifier
- [ ] Búsqueda amplia y refinamiento: 12 GradientBoostingClassifier
- [ ] Búsqueda amplia y refinamiento: 13 HistGradientBoostingClassifier
- [ ] Búsqueda amplia y refinamiento: 14 XGBClassifier
- [ ] Búsqueda amplia y refinamiento: 15 LGBMClassifier
- [ ] Confirmación con repeticiones (top 3 de cada modelo).
- [ ] Validación anidada de los finalistas.
- [ ] Pruebas de preprocesamiento (4.3).

---

## 5. Controles (se aplican a cada resultado)

- [ ] **Fuga:** todo el preprocesamiento dentro del pipeline; folds agrupados; `test.csv` sin tocar.
- [ ] **Sobreajuste:** diferencia entre entrenamiento y validación; una mejora solo cuenta si supera la dispersión entre folds.
- [ ] **Sobreajuste a la validación:** al probar muchas combinaciones, la mejor siempre sale algo inflada. Control: confirmar el modelo final una sola vez en `test.csv`.
- [ ] **Punto de corte:** elegirlo y medirlo sobre las mismas predicciones OOF es levemente optimista. Reportarlo así y confirmar en `test.csv`.
- [ ] **Subajuste:** comparar contra las referencias y contra la regresión logística.
- [ ] **Test desconocido:** resultados por subgrupo; validación dejando afuera un año y dejando afuera una región, para los finalistas.

---

## 6. Cierre

- [ ] Tabla final de los 10: balanced accuracy, ROC-AUC, F1, precision, recall, punto de corte, dispersión entre folds, diferencia entrenamiento-validación, tiempo.
- [ ] Elegir el modelo final (o una combinación de 2 o 3) con el criterio: mejor balanced accuracy, parejo entre subgrupos, y que corra en el entorno de los profesores (D7).
- [ ] Evaluación única en `test.csv`.
- [ ] Reentrenar con train + test.
- [ ] `predictor.py` independiente y prueba en un entorno limpio.
- [ ] Notebook `02_modelos` con la comparación, para la entrega.

**Fechas:** envío del 11/11 a las 16:00 (opcional, para ver la tabla de posiciones); envío del 18/11 a las 16:00 (el que cuenta).

---

## 7. Fuera de alcance por ahora

Decisión del 9/11: estos 7 modelos de la lista original no se prueban en esta etapa. Quedan documentados para retomarlos si sobra tiempo.

| # | Clasificador | Motivo | Qué haría falta para retomarlo |
|---|---|---|---|
| 3 | GaussianNB | Descartado por decisión del equipo: supone variables independientes y con forma de campana, y el EDA mostró lo contrario (variables muy asociadas, casi todas concentradas en cero) | Variante propia de preprocesamiento: one-hot y numéricas llevadas a forma simétrica; `var_smoothing` (1e-12 a 1e-2) |
| 9 | DecisionTreeClassifier (árbol solo) | Descartado por decisión del equipo: Random Forest, Bagging y los boosting ya son conjuntos de árboles y lo superan | Variante D; `max_depth` (3 a 20), `min_samples_leaf` (5 a 200), `ccp_alpha` |
| 6 | SVC rbf | Costo: el tiempo crece con el cuadrado de las filas; con ~30.000 filas por fold son minutos por ajuste y horas por búsqueda | Entrenar con submuestra de 8.000 a 10.000 filas por fold (respetando grupos), valores por defecto primero y búsqueda solo si queda a menos de ~2 puntos del mejor |
| 7 | SVC poly | Igual que 6 | Igual que 6 |
| 8 | SVC sigmoid | Igual que 6 | Igual que 6 |
| 16 | CatBoostClassifier | Colab no lo trae instalado: el predictor tendría que instalarlo al arrancar | `pip install catboost` en el entorno y en la notebook de Colab; variante E de preprocesamiento |
| 17 | Annoy + KNN | Colab no lo trae instalado y no tiene instalador precompilado (se compila); además necesita un wrapper propio porque no trae clasificador | `pip install annoy`, escribir el wrapper con `fit` y `predict_proba`, variante B |

Notas:
- `LinearSVC` (5) sí se prueba: es rápido y representa a la familia SVM.
- `KNeighborsClassifier` (4) cubre la familia de vecinos; Annoy es su versión aproximada y no debería dar un resultado mejor, solo más rápido.
- XGBoost (14) y LightGBM (15) siguen en la lista porque Colab los trae instalados. En la máquina local necesitan `libomp` (`brew install libomp`).
