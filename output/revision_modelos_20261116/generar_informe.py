from pathlib import Path
import pandas as pd,json
P=Path(__file__).resolve().parent
m=pd.read_csv(P/'comparacion_protocolos.csv');orig=pd.read_csv(P/'configuraciones_originales.csv');strict=pd.read_csv(P/'control_activo/resumen.csv');summary=json.loads((P/'resumen.json').read_text());comparison=json.loads((P/'control_activo/comparacion31_15.json').read_text())
name={'hist_gradient_boosting':'HistGradientBoosting','gradient_boosting':'GradientBoosting','random_forest':'Random Forest','bagging':'Bagging','logistic':'Logística','ridge':'Ridge','linear_svc':'Linear SVC','knn':'KNN','promedio':'Promedio de 3 modelos','stacking':'Stacking'}
def f(v,n=2):return 'No medido' if pd.isna(v) else f'{v:.{n}f}'.replace('.',',')
def table(cols,rows):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
noassigned=m[~m.assigned]
primary=table(['Modelo','BA externa','Desvío folds (pp)','ROC-AUC','Gap diagnóstico (pp)'],[[name[r.modelo],f(r.ba_externa*100)+'%',f(r.std_folds*100),f(r.roc_auc,4),f(r.gap_global_oof*100)] for r in noassigned.itertuples()])
allvariants=table(['Modelo','GPU asignada','BA externa','ROC-AUC','Gap OOF (pp)'],[[name[r.modelo],'Incluida' if r.assigned else 'Excluida',f(r.ba_externa*100)+'%',f(r.roc_auc,4),f(r.gap_global_oof*100)] for r in m.itertuples()])
originals=table(['Modelo original','BA OOF sin GPU asignada','ROC-AUC sin GPU asignada','Búsquedas'],[[name[r.modelo],f(r.ba_original_sin_assigned_oof*100)+'%',f(r.auc_original_sin_assigned,4),r.n_busqueda] for r in orig.itertuples()])
strict_t=table(['Candidato','BA externa','Desvío (pp)','ROC-AUC','BA train*','BA OOF global*','Gap* (pp)'],[[r.candidato,f(r.ba_externa*100)+'%',f(r.std_folds*100),f(r.roc_auc,4),f(r.ba_train_global*100)+'%',f(r.ba_oof_global*100)+'%',f(r.gap_global_oof*100)] for r in strict.itertuples()])
text=f'''# Revisión de modelos y sobreajuste · 16/11/2026

## Conclusión

No hay evidencia de que sea necesario abandonar HistGradientBoosting por «mucho overfitting» basándose solo en el gap de 4,17 puntos. El gap es una señal de optimismo de entrenamiento y sensibilidad a la capacidad, no un diagnóstico automático de fracaso fuera de muestra. Tampoco se ha demostrado ausencia de sobreajuste.

**Mantendría HistGradientBoosting de 31 hojas si el criterio principal sigue siendo maximizar BA con la evidencia disponible.** GradientBoosting reduce algo el gap, pero no mejora la BA ni lo supera en el control adicional. Los ensembles tampoco mejoraron de forma consistente. No apareció otro modelo demostrado mejor para sustituirlo.

**La alternativa conservadora más interesante es el mismo HistGB con 15 hojas.** En el control adicional reduce el gap de 4,00 a 2,58 puntos y el desvío entre folds de 0,93 a 0,71, a cambio de bajar la BA externa de 84,68% a 84,28%. Dentro de estas dos configuraciones de HistGB, 15 hojas cumple la regla heurística de quedar a un error estándar de la mejor, que ya se utilizaba en el proyecto. Esa regla favorece simplicidad; no demuestra igual rendimiento ni mejor generalización universal. Elegir 15 por esa preferencia es defendible, pero no debe venderse como mayor BA ni heredar el resultado de TEST del modelo de 31 hojas.

No se cambió el predictor, el umbral guardado ni las decisiones originales. No se volvió a abrir test.csv ni se calcularon nuevas predicciones de TEST. Los nuevos ajustes se limitaron a TRAIN.

## Alcance y verificación

- Se revisaron las ocho familias y las 669 configuraciones distintas de búsqueda/refinamiento guardadas; no son 669 observaciones independientes ni el número total de entrenamientos.
- Se recalcularon matrices y métricas de los 47 conjuntos de predicciones de la evaluación previa. Se verificaron etiquetas, índices, cobertura y predicción = score >= umbral.
- Se revisaron los controles de gap, las confirmaciones originales de tres semillas, las curvas y las evaluaciones internas de población conservadas en el directorio. Estas pruebas internas no se incorporan a la presentación.
- Se revisaron promedio y stacking estrictos, con sus predicciones externas.
- Se reconstruyó Random Forest sin GPU asignada desde su caché previo: cinco scores externos y cinco bloques OOF internos, con verificación de hashes, índices y parámetros. No hubo entrenamiento para esa reconstrucción.
- Se ejecutó un control nuevo, acotado a tres candidatos fijos: HistGB 31 hojas, HistGB 15 hojas y GB regularizado. Cinco folds externos, tres internos para el umbral: 60 ajustes, todos sobre TRAIN, sin advertencias. Se conservaron scores, parámetros, versiones y manifiesto.
- XGBoost y LightGBM están definidos en el registro, pero no tienen corridas completas guardadas. No se presentan como modelos evaluados.

## Comparación principal existente: GPU asignada excluida

{primary}

BA externa: predicciones concatenadas de cinco folds externos, con corte elegido mediante tres folds internos del entrenamiento correspondiente. El desvío es ddof=0 y no es un intervalo de confianza. El gap diagnóstico usa un corte global OOF y su propio control de train; no se obtiene restando la BA externa de esta tabla a una BA de entrenamiento de otro protocolo. «No medido» significa que no hay un gap de entrenamiento comparable guardado para esa configuración; no significa cero.

Random Forest recuperado: BA {f(summary['rf_sin_assigned']['ba']*100)}%, ROC-AUC {f(summary['rf_sin_assigned']['auc'],4)}. HistGB le gana en {summary['rf_comparacion']['folds_ganados_hgb']}/5 folds. Diferencia HGB−RF: {f(summary['rf_comparacion']['diferencia_hgb_menos_rf']*100)} pp, intervalo bootstrap condicional 95% [{f(summary['rf_comparacion']['ic_condicional025']*100)}, {f(summary['rf_comparacion']['ic_condicional975']*100)}] pp. El bootstrap remuestrea grupos completos y conserva sus filas; no incorpora nuevas búsquedas ni repeticiones de entrenamiento.

Promedio y stacking quedan apenas por debajo de HistGB y requieren más componentes. El AUC del stacking es ligeramente mayor, pero el objetivo de la competencia es BA; no corresponde cambiar de métrica después de ver los resultados.

## Todos los modelos con evaluación de umbral interno guardada

{allvariants}

Las variantes que incluyen GPU asignada no son candidatas equivalentes al predictor que decidió excluirla. Los gaps de lineales y Bagging regularizado son menores, pero no justifican por sí solos preferirlos frente a modelos con mayor BA. La cobertura de variables y evaluaciones no es idéntica para todos: no se demuestra superioridad universal sobre variantes que faltan.

## Primeras configuraciones, todas con scores OOF sin GPU asignada

{originals}

Estos números son otra etapa. Se eligió un corte global sobre los mismos OOF: son algo optimistas y no se comparan como si fueran evaluaciones independientes del procedimiento de umbral. El HistGB original de 868 iteraciones da 85,02% OOF sin GPU asignada, frente al 84,84% del elegido de 100 iteraciones; ese pequeño incremento no demuestra que convenga añadir la complejidad del original. Las confirmaciones de tres semillas corresponden a las configuraciones originales con GPU asignada; no se heredan automáticamente a las revisiones posteriores.

La búsqueda también contiene alternativas con menor gap. Por ejemplo, al limitar retrospectivamente el gap a 3 pp aparece HistGB con BA OOF 85,05% y gap 2,45 pp, pero incluye GPU asignada y fue elegido de la misma búsqueda. No es prueba de una mejora del predictor final sin esa variable. La tabla completa de mejores candidatos bajo cada límite se guarda en frontera_ba_gap_busqueda.csv.

## Hallazgo de agrupamiento y control adicional

Los grupos originales se construyen con 31 entradas, incluida assigned_gpu_type. Esa variable se descarta después en el predictor final. Al definir identidad usando solo las entradas crudas efectivas —sin assigned_gpu_type ni execution_week, que también se descarta— hay **1.259 filas de validación (3,29% de TRAIN)** cuyas entradas efectivas coinciden con alguna fila del entrenamiento de su fold original. Eliminar solo execution_week no produce coincidencias: el cambio se debe a assigned_gpu_type.

Esto no demuestra que sean el mismo cliente o reserva ni permite llamarlo automáticamente fuga de etiquetas. Sí limita la afirmación «ninguna entrada idéntica del predictor cruza folds». Se hizo un control con grupos de las 29 entradas crudas efectivas, dentro de los folds externos e internos. Quedaron 25.444 grupos. No se agrupa usando target ni estados.

{strict_t}

*Las tres columnas de diagnóstico usan un único corte global OOF por candidato, que se optimiza sobre esos mismos OOF. BA externa usa los cinco cortes internos. No mezclar columnas para calcular gaps distintos. Cambiar grupos también cambia las particiones: la diferencia contra las cifras anteriores no es una medición causal aislada de «inflado».

La BA externa de HistGB 31 permanece cercana a la anterior (84,68% frente a 84,83%). La mayor regularización de 15 hojas baja tanto el rendimiento de train como el externo. No aparece un colapso de validación del candidato elegido ni una mejora de BA por reemplazarlo por GB.

Comparación HGB31−HGB15 en el control: +{f(comparison['diferencia_hgb31_menos15']*100)} pp. Bootstrap pareado condicional 95%: [{f(comparison['ic_condicional025']*100)}, {f(comparison['ic_condicional975']*100)}] pp. No demuestra equivalencia ni incluye la incertidumbre de volver a entrenar. El error estándar heurístico de la BA de HGB31 es {f(comparison['regla_1se_de31']*100,3)} pp y la diferencia media es aproximadamente 0,398 pp: 15 hojas queda dentro de ese margen. Los folds comparten datos de entrenamiento; este error estándar no es un intervalo formal.

## Qué significa el gap

Un modelo ajustado se evalúa sobre las mismas filas utilizadas para entrenarlo; es esperable que ese resultado sea optimista respecto de datos fuera de muestra. No hay un umbral universal de gap que convierta 4,17 puntos en una condena del modelo. Conviene evaluar curvas de capacidad, resultados externos, estabilidad y alternativas más regularizadas. Reducir el gap bajando el rendimiento de train, sin mejorar validación, no prueba una mejora de generalización.

En la curva original con GPU asignada, pasar de 31 a 127 hojas aumenta BA train aproximadamente de 89,43% a 95,47%, mientras BA OOF pasa de 85,31% a 85,77%. Con 255 hojas, train llega a 97,91% y validación baja a 85,31%. Ahí sí hay evidencia clara de exceso de capacidad. El modelo elegido no usa esas configuraciones grandes. El control nuevo de 15 frente a 31 hojas muestra el compromiso entre una ganancia externa pequeña y mayor brecha de entrenamiento.

Limitaciones reales: búsqueda de hiperparámetros no anidada por completo; solo una semilla externa en el nuevo control; cinco folds no independientes; categorías y poblaciones nuevas pueden cambiar el rendimiento. Las pruebas internas de población ya guardadas muestran esa fragilidad, aunque no se exhiban en la defensa. No se afirma que el modelo esté libre de sobreajuste ni que se optimizó de forma definitiva la regularización.

El TEST interno de 85,09% del candidato original sigue siendo un resultado histórico favorable, con el solapamiento declarado, no un certificado general. Ya se consultó y no sirve como nueva confirmación independiente para una variante elegida ahora. Ningún resultado nuevo de este informe se midió en TEST.

## Revisión del TP2

Se inspeccionó la única rama, main, de [ml_tp2_grupo9](https://github.com/agkorman/ml_tp2_grupo9). La configuración final fija es RF de 100 árboles, profundidad 12, split mínimo 200 y hoja mínima 10. El código de [validación](https://github.com/agkorman/ml_tp2_grupo9/blob/main/modelos/validacion_cruzada.py) usa return_train_score=False para la comparación, y el de [modelo final](https://github.com/agkorman/ml_tp2_grupo9/blob/main/modelos/modelo_final.py) guarda métricas de TEST. [ajuste_rf.py](https://github.com/agkorman/ml_tp2_grupo9/blob/main/modelos/ajuste_rf.py) sí puede producir train/validación de la búsqueda, pero esos resultados no están versionados: resultados/ está excluido en .gitignore. Por tanto, el repo no aporta el valor numérico observado del gap del RF final. No se inventa ni se atribuye al final el gap exploratorio de 9,8 puntos de AUC. Ese gap sería de AUC, no directamente comparable con BA.

## Cómo defender la elección

«No seleccionamos el modelo por su rendimiento de entrenamiento ni por minimizar el gap. Comparamos BA fuera de fold, dispersión y complejidad. Reducir las hojas redujo la brecha, pero también la BA externa. HistGB mantuvo la mayor BA en los controles comparables; reconocemos la brecha y las limitaciones de validación.»

Si se adopta explícitamente el criterio conservador de un error estándar: «Entre configuraciones cercanas en BA, preferimos la de 15 hojas por menor complejidad y dispersión». Esa decisión requeriría tratar el predictor como una variante nueva y no asignarle el TEST de 85,09% del modelo anterior.

## Referencias metodológicas

- [scikit-learn: curvas de validación y aprendizaje](https://scikit-learn.org/stable/modules/learning_curve.html): diagnosticar capacidad con train y validación, y distinguir sesgo y varianza.
- [scikit-learn: validación anidada y no anidada](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): el optimismo de seleccionar hiperparámetros sobre la misma validación no se elimina solamente separando el umbral.

## Reproducibilidad

- auditar.py: reconstruye y verifica resultados existentes, sin entrenar ni abrir TEST.
- control_agrupamiento.py: 60 ajustes nuevos en TRAIN, tres candidatos fijos, dos procesos, sin TEST. Conserva un manifiesto, scores y avisos por fold.
- comparacion_protocolos.csv: métricas de configuraciones fijas y protocolos identificados.
- configuraciones_originales.csv: resultados de las primeras configuraciones.
- frontera_ba_gap_busqueda.csv: alternativas de búsqueda bajo límites retrospectivos de gap.
- control_activo/resumen.csv: resultados del control de agrupamiento y regularización.
- control_activo/comparacion31_15.json: diferencia pareada y bootstrap condicional.

Los artefactos originales se conservaron; todos los resultados nuevos están en este directorio de revisión.
'''
(P/'informe.md').write_text(text)
print(str(P/'informe.md'))
