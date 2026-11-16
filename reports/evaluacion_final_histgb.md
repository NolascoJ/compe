# Evaluación final de HistGradientBoosting en holdout interno

Modelo fijo sin assigned_gpu_type ni gpu_changed; pipeline original E, ajustado exclusivamente con las 38,274 filas de train. Target, execution_status y execution_status_date excluidos de las entradas. El estado se utiliza exclusivamente para análisis. Parámetros completos, versiones y hashes: [manifest.json](../experiments/evaluacion_final_histgb/manifest.json).

Umbral fijo **0.38294747099493937**; cancelación si score >= umbral. Una única llamada a predict_proba sobre test; sin selección ni ajustes posteriores. Índice test_index: posición original en test.csv, base cero, sin reordenamiento. Predictor y umbral guardados antes de abrir test.

**Balanced accuracy global: 85.0904%.**

| Grupo | n | n0 | n1 | TN | FP | FN | TP | balanced_accuracy | recall | specificity | precision | f1 | accuracy | roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global_ | 9568 | 5577 | 3991 | 4765 | 812 | 609 | 3382 | 0.850904 | 0.847407 | 0.854402 | 0.806390 | 0.826390 | 0.851484 | 0.937743 |
| sin_coincidencias | 9287 | 5563 | 3724 | 4761 | 802 | 513 | 3211 | 0.859039 | 0.862245 | 0.855833 | 0.800150 | 0.830037 | 0.858404 | 0.943213 |
| PRT | 3854 | 1332 | 2522 | 1099 | 233 | 250 | 2272 | 0.862974 | 0.900872 | 0.825075 | 0.906986 | 0.903919 | 0.874676 | 0.947538 |
| resto | 5714 | 4245 | 1469 | 3666 | 579 | 359 | 1110 | 0.809610 | 0.755616 | 0.863604 | 0.657194 | 0.702977 | 0.835842 | 0.907956 |
| sin_locked_allocation | 7840 | 5573 | 2267 | 4762 | 811 | 609 | 1658 | 0.792920 | 0.731363 | 0.854477 | 0.671527 | 0.700169 | 0.818878 | 0.890595 |
| con_locked_allocation | 1728 | 4 | 1724 | 3 | 1 | 0 | 1724 | 0.875000 | 1.000000 | 0.750000 | 0.999420 | 0.999710 | 0.999421 | 0.999710 |
| no_show | 111 | 0 | 111 | 0 | 0 | 57 | 54 | indefinida | 0.486486 | indefinida | 1.000000 | 0.654545 | 0.486486 | indefinida |
| anio_2015 | 1804 | 967 | 837 | 906 | 61 | 138 | 699 | 0.886022 | 0.835125 | 0.936918 | 0.919737 | 0.875391 | 0.889690 | 0.971618 |
| anio_2016 | 4386 | 2755 | 1631 | 2401 | 354 | 257 | 1374 | 0.856967 | 0.842428 | 0.871506 | 0.795139 | 0.818101 | 0.860693 | 0.934287 |
| anio_2017 | 3378 | 1855 | 1523 | 1458 | 397 | 214 | 1309 | 0.822736 | 0.859488 | 0.785984 | 0.767292 | 0.810777 | 0.819124 | 0.920991 |

Las cantidades n0 y n1 corresponden a completados y cancelados. BA y ROC-AUC son indefinidas si falta alguna clase; denominadores nulos también quedan indefinidos. no_show cuenta como cancelación; su recall es la proporción detectada por este mismo predictor.

Solapamiento conocido: previamente se encontraron 281/9.568 filas (2,94%) con entradas idénticas a train. Reproducción actual: **281/9568 (2.9369%)**, usando exclusivamente las 31 FEATURE_COLS originales (incluye assigned_gpu_type para esta definición), nulos iguales. No se usa la coincidencia como predictor ni se excluye del resultado global. El subconjunto sin coincidencias se informa adicionalmente; tampoco garantiza independencia de clientes ni generalización temporal.

La CV previa obtuvo aproximadamente 84,829% de BA externa con cinco umbrales internos diferentes, entrenamientos parciales y folds agrupados sin entradas idénticas entre entrenamiento y validación. **No corresponde atribuir esa cifra al umbral global fijo de esta evaluación.** La diferencia descriptiva del holdout es +0.261 puntos porcentuales. El modelo actual usa todo train y otra población, con el solapamiento declarado. La BA OOF global de 84,84% del control de gap utilizó un umbral elegido sobre esos mismos OOF y es levemente optimista. Estas diferencias no constituyen una prueba estadística ni justifican cambiar el predictor. No se probaron otros candidatos ni umbrales en test. Este holdout interno no es el test oculto de los profesores.

Reproducir/verificar: `PYTHONPATH=src .venv/bin/python experiments/evaluacion_final_histgb/evaluar.py`. Si existen resultados completos, verifica hashes y consistencia y regenera métricas e informe desde predictions.csv, sin entrenar ni predecir. Si hay archivos parciales o inconsistentes, aborta. Los experimentos anteriores permanecen intactos.
