# Control comparable del gap, sin assigned

Se reentrenó GradientBoosting con su configuración fija y los mismos cinco folds agrupados (semilla 42), sin assigned ni gpu_changed. Los OOF reproducen los scores guardados con tolerancia absoluta 1e-12. No se abrió test.csv ni se buscaron hiperparámetros.

| Modelo | BA entrenamiento (diagnóstico) | BA OOF global* | Gap | BA con umbral interno |
|---|---:|---:|---:|---:|
| GradientBoosting | 87,68% | 84,39% | 3,29 puntos | 84,27% |
| HistGradientBoosting | 89,02% | 84,84% | 4,17 puntos | 84,83% |

*Umbral optimizado sobre los mismos OOF: BA levemente optimista. La columna de umbral interno procede de la evaluación anterior. No mezclar sus protocolos para calcular el gap.

GradientBoosting reduce el gap en 0,88 puntos, pero esa reducción combina menor rendimiento de entrenamiento y menor validación. Es evidencia de menor separación train-validación, no prueba automática de mejor generalización. La diferencia externa de 0,56 puntos a favor de HistGB es pequeña frente a la dispersión; no fundamenta sola la elección.

Mantengo HistGB como candidato para test: el control no cambia su ventaja observada ante PRT no visto (75,56% vs 55,26%), su menor dispersión externa (1,03 vs 1,50 puntos) y su mejor mínimo por subgrupo. GB conserva ventajas reales: menor gap, umbral más plano y ligera ventaja en PRT→resto (53,86% vs 51,47%), donde ambos fallan. No hay dominancia universal.

Este control completa la limitación de comparación de gaps señalada en los informes anteriores; el gap comparable de GB sin assigned es 3,29, no el 2,96 de su variante con assigned. No altera la configuración ni el umbral propuestos para HistGB. Queda pendiente la evaluación única del candidato en holdout.

Código y datos: experiments/seleccion/revision_stacking/control_gap_gb.py y comparacion_gap_sin_assigned.csv.
