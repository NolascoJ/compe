# Baseline, análisis de errores y estrategia de intervención

## Alcance y baseline (2.1–2.3)

Se declara como baseline del informe la **regresión logística calibrada sin GPU asignada**.
Es una referencia lineal interpretable, ya ajustada mediante búsqueda y confirmación sobre TRAIN:
`C=0.016779852749843744`, `class_weight='balanced'`, `l1_ratio=0`.
Usa one-hot, logaritmos y escalado dentro del pipeline. No es el predictor de la primera entrega
(RF + CatBoost), que corresponde a otro desarrollo y no tiene OOF comparables en este análisis.

Este diagnóstico se agrega después de la búsqueda histórica. Las nuevas cohortes no se presentan
como causa de decisiones anteriores. No se entrena ni se cambia el candidato o su umbral.

### Dos métricas y su papel

- **Balanced accuracy (BA)** = (recall de cancelados + recall de completados) / 2.
  Pondera ambas clases por igual, evita premiar la predicción constante de la clase mayoritaria
  y coincide con el objetivo de competencia. Es el criterio principal para hiperparámetros y
  para el corte de decisión, que se aprende solo en el entrenamiento de cada fold externo.
- **ROC-AUC** mide el ordenamiento de los scores a través de distintos cortes. Se usa como
  diagnóstico secundario al comparar modelos y configuraciones: permite distinguir una mejora
  de ordenamiento de una mejora debida al corte. No determina el umbral ni garantiza probabilidades
  calibradas. Si BA y AUC discrepan se conserva la prioridad de BA y se explica la discrepancia.

La búsqueda histórica registró ambas métricas, pero su regla automática seleccionó por BA,
gap y simplicidad; no se afirma que AUC haya sido un segundo objetivo de optimización.
La comparación de cierre considera explícitamente ambas. No se introduce ahora una búsqueda nueva.

### Protocolo comparable

Solo TRAIN (38.274 filas), mismos cinco folds externos agrupados y mismos índices por modelo.
Cada corte se elige con tres folds internos del entrenamiento externo; preprocesamiento ajustado
dentro de su entrenamiento. Una predicción externa por fila. Todos los modelos de esta tabla
excluyen `assigned_gpu_type` y `gpu_changed`. BA/AUC globales se calculan concatenando OOF;
también se guarda el desvío entre folds (ddof=0), que no es un intervalo de confianza.
La selección histórica de hiperparámetros no fue totalmente anidada; persiste optimismo de selección.

| Modelo | BA | Desvío BA (pp) | ROC-AUC | Δ BA vs baseline (pp) | FP | FN |
|---|---|---|---|---|---|---|
| Logística (baseline) | 82.01% | 1.39 | 0.9039 | +0.00 | 4027 | 2862 |
| GradientBoosting | 84.27% | 1.50 | 0.9327 | +2.26 | 3159 | 2761 |
| HistGB (final) | 84.83% | 1.03 | 0.9355 | +2.82 | 3253 | 2516 |

HistGB mejora el baseline en ambas métricas. No se atribuye toda la mejora a una sola feature:
cambian familia, representación categórica y capacidad. El diagnóstico de curvas de aprendizaje
es consistente con limitación de capacidad del modelo lineal.

## Errores por cohortes exigidas (3.1)

**FP:** una reserva que se completa es marcada como cancelación; podría provocar contactos
innecesarios o una reasignación incorrecta de recursos. **FN:** una reserva cancelada se marca
como vigente; podría retener capacidad que finalmente no se utiliza. Son consecuencias posibles,
no costos medidos: no hay datos de utilización, penalidades ni política de acción para monetizarlas.
Las predicciones deben apoyar una revisión, no justificar cancelaciones automáticas.

Las cohortes se definen por los valores originales de `client_type`, `service_tier` y
`reserved_gpu_type`, antes del agrupamiento de categorías raras. Se incluyen todas las categorías.
`n0/n1` son completados/cancelados; BA y AUC quedan indefinidas si falta una clase.
El asterisco identifica menos de 30 observaciones de alguna clase: no priorizar por BA sin mirar soporte.
Las dimensiones se superponen; no sumar errores de dimensiones distintas.

### client_type

| Cohorte | n (n0/n1) | BA base | BA final | AUC base/final | FP base→final | FN base→final | Δ BA (pp) |
|---|---|---|---|---|---|---|---|
| cloud_marketplace | 18717 (11618/7099) | 74.25% | 77.49% | 0.808/0.862 | 3277→2593 | 1654→1611 | +3.25 |
| enterprise_client | 1350 (1093/257) | 74.02% | 78.47% | 0.868/0.899 | 32→24 | 126→105 | +4.45 |
| high_priority_partner * | 116 (91/25) | 55.25% | 67.96% | 0.789/0.776 | 5→11 | 21→13 | +12.70 |
| independent_client | 2896 (2419/477) | 61.56% | 64.64% | 0.760/0.803 | 110→88 | 345→320 | +3.08 |
| managed_account | 8306 (4733/3573) | 90.36% | 93.04% | 0.959/0.985 | 314→273 | 452→291 | +2.69 |
| research_consortium | 6655 (2150/4505) | 90.91% | 92.36% | 0.982/0.991 | 276→257 | 241→150 | +1.45 |
| sponsored_project * | 233 (206/27) | 56.10% | 52.00% | 0.576/0.534 | 13→7 | 22→25 | -4.10 |
| unknown * | 1 (0/1) | — | — | — | 0→0 | 1→1 | — |

Menor BA del baseline con soporte suficiente: **independent_client**, 61.56% (n=2896). Mayor cantidad de FN del baseline: **cloud_marketplace**, 1654 (57.79% de todos sus FN).

### service_tier

| Cohorte | n (n0/n1) | BA base | BA final | AUC base/final | FP base→final | FN base→final | Δ BA (pp) |
|---|---|---|---|---|---|---|---|
| basic | 5081 (3133/1948) | 74.08% | 76.64% | 0.805/0.857 | 923→751 | 436→443 | +2.57 |
| enterprise * | 9 (4/5) | 100.00% | 100.00% | 1.000/1.000 | 0→0 | 0→0 | +0.00 |
| priority | 3211 (2029/1182) | 84.08% | 88.55% | 0.914/0.957 | 265→190 | 222→160 | +4.47 |
| standard | 29973 (17144/12829) | 83.13% | 85.80% | 0.915/0.943 | 2839→2312 | 2204→1913 | +2.67 |

Menor BA del baseline con soporte suficiente: **basic**, 74.08% (n=5081). Mayor cantidad de FN del baseline: **standard**, 2204 (77.01% de todos sus FN).

### reserved_gpu_type

| Cohorte | n (n0/n1) | BA base | BA final | AUC base/final | FP base→final | FN base→final | Δ BA (pp) |
|---|---|---|---|---|---|---|---|
| GPU_T1 | 30118 (17031/13087) | 83.63% | 86.32% | 0.920/0.948 | 2786→2257 | 2143→1845 | +2.69 |
| GPU_T12 * | 6 (0/6) | — | — | — | 0→0 | 6→6 | — |
| GPU_T2 | 529 (339/190) | 72.83% | 77.15% | 0.788/0.852 | 70→55 | 64→56 | +4.32 |
| GPU_T3 * | 7 (3/4) | 100.00% | 100.00% | 1.000/1.000 | 0→0 | 0→0 | +0.00 |
| GPU_T4 | 5777 (3754/2023) | 75.44% | 78.96% | 0.821/0.873 | 907→728 | 505→459 | +3.52 |
| GPU_T5 | 740 (498/242) | 75.66% | 76.97% | 0.824/0.852 | 84→75 | 77→75 | +1.32 |
| GPU_T6 | 883 (524/359) | 76.11% | 78.49% | 0.821/0.862 | 173→132 | 53→64 | +2.38 |
| GPU_T7 | 214 (161/53) | 84.62% | 87.76% | 0.899/0.936 | 7→6 | 14→11 | +3.14 |

Menor BA del baseline con soporte suficiente: **GPU_T2**, 72.83% (n=529). Mayor cantidad de FN del baseline: **GPU_T1**, 2143 (74.88% de todos sus FN).

Los CSV adjuntos incluyen recall, especificidad, FPR, FNR, tasa de cancelación,
fracción de filas y participación en el total de FP/FN, además de las matrices completas.
Una tasa baja de error no implica BA alta cuando una clase es muy poco frecuente.

## Cohorte prioritaria e intervención (3.3)

Se prioriza **anticipación registrada entre 0 y 7 inclusive** (`hours_before_execution <= 7`).
La unidad se mantiene en **horas**, según el diccionario del enunciado; no se interpreta como
una semana ni se transforma a días basándose en la hipótesis del origen hotelero del dataset.
Es una cohorte adicional a las tres obligatorias, definida por una regla reproducible y disponible al predecir.

| Modelo | n | % filas | BA | Recall | FP | FN | % FN totales |
|---|---|---|---|---|---|---|---|
| Logística (baseline) | 5220 | 13.64% | 60.89% | 29.62% | 355 | 487 | 17.02% |
| GradientBoosting | 5220 | 13.64% | 60.45% | 22.40% | 68 | 537 | 19.45% |
| HistGB (final) | 5220 | 13.64% | 61.42% | 25.29% | 111 | 517 | 20.55% |

**Frecuencia e impacto:** la tabla cuantifica volumen y cancelaciones omitidas tanto por el
baseline como por el final. Se prioriza por recall bajo y presión operativa: con poca anticipación
hay menos margen para confirmar la reserva o volver a ofrecer capacidad liberada.
Esto es una hipótesis de operación; el dataset no mide capacidad desperdiciada.

**Causas posibles:** las variables estáticas pueden no representar el estado más reciente de
la reserva; faltan confirmaciones, preparación del workload y actividad del cliente.
También podrían cambiar las tasas de cancelación o existir menos positivos para aprender.
Los desgloses son asociaciones: no prueban ninguna de esas causas.

**Intervención propuesta, aún no implementada:** recopilar confirmación de reserva, carga de
contenedor/datos, pago acreditado y eventos de modificación con timestamp. Definir una fotografía
semanal de reservas activas con timestamp de predicción; usar solo eventos conocidos hasta ese
instante. Incorporar ID estable de reserva/cliente y fechas de creación/cancelación para definir
correctamente quién sigue activo y evitar eventos posteriores al corte.

1. Auditar significado y unidad de `hours_before_execution`: el campo describe creación→ejecución,
   no tiempo restante desde la fotografía semanal. Obtener fecha de creación y de ejecución para
   medir también ese tiempo restante sin confundirlos.
2. Medir cobertura y antigüedad de las nuevas señales en esta cohorte y en el resto; auditar faltantes.
3. Comparar HistGB fijo con/sin esas señales usando el mismo protocolo y nuevos datos reservados
   antes del análisis. Agrupar por cliente/reserva cuando haya IDs, además de controlar duplicados.
4. Reportar BA/AUC globales y por las tres cohortes obligatorias, recall, FPR y FN por 1.000 reservas.
   Exigir una mejora consistente entre folds y evaluar incertidumbre; no elegir por una categoría diminuta.
5. Simular una política de confirmación manual; medir contactos, cancelaciones detectadas y costo
   operativo antes de activar decisiones automáticas. La mejora esperada queda como hipótesis.

No se afirma que esta intervención haya mejorado el predictor: hoy no existen esas variables.
La consigna pide discutir la estrategia y los datos a adquirir, no inventar su implementación.

## Estrategias de optimización y decisión final (4–5)

**Selección de modelos:** comparar la referencia lineal con GradientBoosting e HistGB prueba si
la capacidad para interacciones no lineales mejora el sistema. La tabla inicial mantiene folds,
exclusión de assigned y procedimiento de corte comparables; ambos boosting superan la BA/AUC del baseline.

**Optimización de hiperparámetros:** se conservaron búsquedas amplias, refinamientos y
confirmaciones. Su comparación histórica puede reproducirse en `experiments/<modelo>/`.
No se mezclan métricas OOF con corte global y métricas externas con cortes internos para
atribuir una ganancia aislada al tuning. HistGB final conserva 100 iteraciones y 31 hojas por
rendimiento y simplicidad; más complejidad no produjo una ventaja concluyente.

**Ensembles:** promedio BA 84,73%, AUC 0,93525; stacking BA 84,66%, AUC 0,93582;
HistGB BA 84,83%, AUC 0,93554. Se compararon con el mismo protocolo externo. El stacking
ordena ligeramente mejor según AUC, pero no mejora BA y exige más componentes: se conserva HistGB.
Que una estrategia no mejore también es un resultado válido del experimento.

Las tablas por cohorte muestran si la ganancia es global o localizada; no se ocultan regresiones
de categorías particulares. Para comparar esas diferencias hay que considerar soporte y la
selección posterior de cohortes; no son pruebas confirmatorias de significancia.

**Modelo seleccionado:** HistGB sin GPU asignada; umbral de inferencia ya fijado
0,38294747099493937. BA externa 84,83%; BA histórica del holdout interno 85,09% y AUC 0,93774.
Este informe no vuelve a abrir el holdout. No es el test oculto de la competencia.
La expectativa depende de la población: en años excluidos HistGB obtuvo aproximadamente
77,47–84,32% BA; en extrapolación regional extrema llegó a 51,47%. El 85% local no garantiza el oculto.
El control posterior con agrupamiento de entradas efectivas obtuvo 84,68% BA: ver la auditoría existente.

## Reproducción y archivos

```sh
PYTHONPATH=src .venv/bin/python experiments/analisis_consigna/analizar.py
```

- `cohortes.csv`: todas las cohortes y tres modelos; matrices, tasas y soporte.
- `comparacion_modelos.csv` y `metricas_folds.csv`: métricas globales y por fold.
- `mejoras_cohortes.csv`: diferencias final−baseline, incluidos cambios en FP/FN.
- `manifest.json`: hashes, alcance y controles de alineación.

Fuentes relacionadas: [curva de aprendizaje](curva_aprendizaje.md),
[adquisición de datos](adquisicion_datos.md), [selección final](seleccion_final_stacking.md)
y [auditoría](../output/revision_modelos_20261116/informe.md).
