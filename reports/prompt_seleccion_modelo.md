# Prompt: análisis de experimentos y elección del modelo ganador

Sos un analista de machine learning. Trabajás en el repo `/Users/jnolascodecarles/ITBA/4_1Q/ml/compe`, que es la competencia del TP3 (ITBA, grupo 9). Tenés que revisar los experimentos ya corridos, quedarte con la mejor configuración **defendible** de cada modelo y recomendar un ganador. El criterio que más pesa es que el modelo **no esté sobreajustado** y que **aguante un test desconocido**. Sacar la mejor balanced accuracy en validación cruzada viene después.

## 1. Qué leer antes de empezar

1. `CLAUDE.md`: reglas del repo (fuga de datos, sobreajuste, subajuste, test desconocido). Son obligatorias.
2. `TODO.md`, secciones 4 a 6: protocolo de ajuste, regla de selección y criterio de cierre. Ojo: el TODO dice "nada de esto está ejecutado", pero eso quedó viejo. El estado real está en la sección 2 de este prompt.
3. `reports/eda_decisiones.md`: decisiones del EDA.
4. `src/compe/validation.py` (cómo se calculan las métricas), `src/compe/run.py` (etapas y regla `select`) y `src/compe/models.py` (espacios de búsqueda y criterio de simplicidad de cada modelo).

## 2. Contexto y estado actual

- Clasificación binaria de `canceled_job`. La métrica es **balanced accuracy (BA)**.
- El test de la competencia lo tienen los profesores y no sabemos cómo está armado. Puede no parecerse a train: por ejemplo, puede no tener `PRT`, tener casi nada de `locked_allocation` o venir de otro año.
- `data/raw/test.csv` es un holdout interno. **No lo abras ni lo uses para nada en esta tarea.** Se evalúa una sola vez, después de que el equipo elija el modelo.
- Los 8 modelos con el protocolo completo terminado están en `experiments/<modelo>/`: `logistic`, `ridge`, `linear_svc`, `knn`, `random_forest`, `bagging`, `gradient_boosting`, `hist_gradient_boosting`.
- **Faltan** `xgboost` y `lightgbm`: no se corrieron. No los inventes ni los corras. Anotalos como pendientes.
- `experiments/resumen.csv` está desactualizado (tiene solo 3 modelos). Regeneralo con `.venv/bin/python -m compe.resumen`, o armá la tabla directo desde los `metricas.json`.
- Todavía no se hicieron la validación anidada ni la validación dejando afuera un año o una región (pasos 6 y 7 del protocolo).
- Para correr Python usá `.venv/bin/python`, con `PYTHONPATH=src` si hace falta.

### Qué hay en cada `experiments/<modelo>/`

| Archivo | Contenido |
|---|---|
| `base.json` | métricas con los valores por defecto |
| `curva_validacion.csv` / `.png` | BA de entrenamiento y de validación según el hiperparámetro de complejidad |
| `busqueda.csv`, `refinamiento.csv` | una fila por combinación probada (`params` está en JSON) |
| `confirmacion.csv` | las 3 mejores combinaciones (con `gap <= 0,05`), promediadas sobre 3 particiones (semillas 42, 43 y 44). `ba_entre_particiones_std` es la dispersión entre esas particiones |
| `mejores_parametros.json` | la combinación que eligió la regla automática |
| `metricas.json` | métricas finales de esa combinación, más dos controles: `ba_val_kfold_comun` / `inflado_por_copias` y `ba_val_sin_assigned` |
| `subgrupos.csv` | BA por subgrupo (`todo`, `sin_locked_allocation`, `region_PRT`, `region_resto`, `anio_2015/16/17`) y cuántos `no_show` se detectan |
| `oof.npy`, `oof_sin_assigned.npy` | puntuaciones fuera de fold en el orden de `train.csv`, con los folds de `make_folds(df, seed=42)` |

### Cómo leer las métricas (está en `validation.py`)

- `ba_val`: BA fuera de fold con el punto de corte que la **maximiza sobre esas mismas predicciones**. Por eso es levemente optimista.
- `ba_train`: BA sobre el entrenamiento de cada fold, con ese mismo punto de corte, promediada entre folds. `gap = ba_train - ba_val`.
- `ba_std`: desvío de la BA entre los 5 folds. Es la vara para decidir si una diferencia es real.
- `inflado_por_copias`: cuánto sube la BA si se usa K-Fold común, que deja copias de una misma fila en entrenamiento y en validación. Si es alto, el modelo memoriza filas.
- `ba_val_sin_assigned`: la misma configuración sin `assigned_gpu_type`. Esa variable se registra tarde y es dudosa (ver `CLAUDE.md`). La decisión D6 del `TODO.md` sigue abierta.
- Los subgrupos usan el punto de corte global, no uno por subgrupo.
- Regla automática (`run.select`): descarta las combinaciones con `gap > 0,05` y, entre las que quedan a menos de un error estándar (`ba_std/√5`) de la mejor, elige la más simple. El límite de 0,05 se puso **provisoriamente**, a la espera de fijarlo con las curvas. Parte de tu trabajo es revisar si es razonable.

## 3. Reglas que no se negocian

- No uses `test.csv`. No uses `execution_status` ni `execution_status_date` como entrada.
- No modifiques ni borres nada de `experiments/<modelo>/`. Si calculás algo nuevo, guardalo en `experiments/seleccion/`.
- No agregues variables ni relances búsquedas de hiperparámetros. Esta tarea es de **análisis y selección** sobre lo que ya está.
- No elijas entre variantes por diferencias menores que la dispersión entre folds. Un empate se desempata por robustez y simplicidad, nunca por el tercer decimal.
- Reportá todo fuera de fold. Cada vez que des una BA con punto de corte OOF, aclará que es levemente optimista.
- No fijes a mano valores puntuales de train (una región, un agente) en reglas o en el punto de corte.

## 4. Tareas

### Tarea A · La mejor configuración de cada modelo

Para cada uno de los 8 modelos:

1. **Curva de validación.** ¿La configuración elegida está en la zona sana, o sea antes de que entrenamiento siga subiendo mientras validación se estanca? ¿O cae en la zona de sobreajuste o en la de subajuste?
2. **Gap.** Mirá todas las combinaciones de `busqueda.csv` + `refinamiento.csv`. ¿Hay alguna con BA dentro de un error estándar de la elegida pero con un gap claramente menor? Si la hay, es mejor candidata. Para los árboles el gap ronda 0,04–0,046, cerca del límite. Decí si conviene un límite más estricto y qué se perdería con él.
3. **Bordes del espacio.** Si el mejor valor quedó en el borde del rango buscado, anotalo: la búsqueda podría estar incompleta.
4. **Estabilidad.** En `confirmacion.csv`, ¿la elegida se sostiene con las 3 semillas? ¿Las 3 finalistas se distinguen entre sí, o están empatadas?
5. **Contra los valores por defecto** (`base.json`): si el ajuste no supera al modelo por defecto por más que la dispersión, decilo y preferí la versión más simple.
6. **Controles:** `inflado_por_copias`, caída sin `assigned_gpu_type` y BA por subgrupo. Marcá el subgrupo más débil y la diferencia entre el mejor y el peor subgrupo.
7. **Veredicto:** la configuración que recomendás (puede ser distinta de la de `mejores_parametros.json` si lo justificás con la regla), con su diagnóstico: sano, sobreajustado o subajustado.

### Tarea B · Comparación entre modelos

1. **Referencias para descartar subajuste:** predecir siempre una clase (0,50); la regla de una variable `locked_allocation -> 1`; la mejor regla de una sola variable, elegida dentro de cada fold; y la regresión logística. Calculá las reglas con los mismos folds (`make_folds(df, seed=42)`) y usando solo el entrenamiento de cada fold.
2. **Comparación pareada con los OOF.** Por cada fold, calculá la BA de cada modelo y las diferencias contra el mejor, fold a fold. Comparar fold a fold es más justo que comparar promedios sueltos. Sumá un bootstrap por grupos de filas idénticas (`data.feature_groups`) para tener un intervalo de la diferencia.
3. **Punto de corte sin optimismo.** Con los hiperparámetros fijos, en cada fold externo: generá OOF internos con folds agrupados solo sobre el entrenamiento externo, elegí ahí el punto de corte, reentrená con todo el entrenamiento externo y aplicalo al fold externo. Informá cuánto baja respecto de `ba_val`. (Rotar el punto de corte sobre los OOF ya guardados no alcanza: los modelos que generaron los OOF de los otros 4 folds vieron el fold evaluado.) Esto no corrige el optimismo de haber elegido entre muchas combinaciones; para eso está la validación anidada, que queda pendiente.
4. **Sensibilidad al punto de corte.** Graficá o tabulá la BA en función del punto de corte alrededor del óptimo (por ejemplo, de cuantil −10 a +10 puntos). Si el test trae otra proporción de clases, el punto de corte óptimo se corre. Un modelo con la curva plana es más seguro.
5. **Combinación (opcional).** Probá el promedio de rangos de los OOF de los 2 o 3 mejores modelos que **no sean de la misma familia**. Solo cuenta si mejora más que la dispersión entre folds y no empeora el subgrupo más débil. Recordá que el predictor final tiene que correr en Colab.

### Tarea C · Robustez de los finalistas (2 o 3 modelos)

Esto es el paso 7 del protocolo. Con los hiperparámetros **fijos** de la Tarea A, el punto de corte elegido solo con la parte de entrenamiento y el pipeline de siempre (`pipeline.make_pipeline` + `models.MODELS[...]`, que ajusta el preprocesamiento dentro de cada partición):

- **Dejando afuera un año:** entrenar con dos años y validar en el tercero (3 corridas).
- **Dejando afuera una región:** entrenar sin `PRT` y validar en `PRT`; y entrenar solo con `PRT` y validar en el resto.
- **Sin `locked_allocation`:** BA en ese subgrupo y BA de un modelo entrenado sin esas filas.
- Todo con y sin `assigned_gpu_type`.

Escribí el código en `experiments/seleccion/` o en un módulo nuevo `src/compe/robustez.py`, sin tocar los existentes. Reportá la caída respecto de la validación cruzada normal. Una caída grande, o mucho mayor que la de un modelo lineal, indica memorización. Antes de lanzar algo que tarde más de ~30 minutos, avisá cuánto estimás que va a tardar.

### Tarea D · Elegir el ganador

Usá este criterio, en este orden:

1. Supera a las referencias con margen (no está subajustado).
2. No muestra señales de sobreajuste: gap razonable, poco `inflado_por_copias`, estable entre semillas y con la curva de validación en zona sana.
3. Es el más parejo entre subgrupos y el que menos cae en la Tarea C.
4. Tiene la BA más alta, pero **solo si la diferencia supera la dispersión**. Si no, es un empate.
5. Si hay empate: el más simple, el menos sensible al punto de corte y el que corre sin problemas en Colab.

Decidí también sobre `assigned_gpu_type` (D6). Poné en la balanza cuánto aporta, el riesgo de que en el test no esté disponible o se comporte distinto, y que es una variable registrada tarde. Recomendá una opción y explicá qué se declara en la presentación.

## 5. Qué entregar

Un informe en `reports/seleccion_modelo.md`, en español, con:

1. **Resumen de una página:** el ganador, el segundo, la decisión sobre `assigned_gpu_type` y la BA esperada con un rango realista. No des un número inflado: tené en cuenta el optimismo del punto de corte y el de haber elegido entre muchas combinaciones.
2. **Tabla por modelo:** configuración recomendada, `ba_val`, BA con punto de corte sin optimismo, `ba_std`, gap, `inflado_por_copias`, BA sin `assigned`, subgrupo más débil y su BA, diagnóstico y tiempo de ajuste.
3. **Comparación pareada y referencias.**
4. **Robustez de los finalistas** (Tarea C).
5. **Riesgos y límites:** qué no se sabe del test, qué quedó sin probar (XGBoost, LightGBM, validación anidada) y en qué casos cambiaría la recomendación.
6. **Próximos pasos** para el equipo, sin ejecutarlos: evaluar una sola vez en `test.csv`, reentrenar con train + test, armar `predictor.py` y probarlo en Colab.

Guardá las tablas intermedias en `experiments/seleccion/` (CSV) para poder rehacer el informe.

## 6. Forma de trabajo

- El equipo decide. Vos recomendás con argumentos y mostrás la evidencia, sin vender el resultado.
- Si encontrás algo raro (una métrica que no cierra, un archivo incompleto o una posible fuga), frená y reportalo antes de seguir.
- Separá siempre lo que medís de lo que interpretás.
- Sé escéptico con tus propias conclusiones: si dos modelos empatan, decilo, aunque eso haga la elección menos vistosa.
