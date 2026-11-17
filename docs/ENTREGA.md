# Estado de la entrega

## Archivos y alcance

- Predictor: `entrega/predictor_v2_Grupo_9.py`.
- Instrucciones y dependencias: `entrega/README.md`, `entrega/requirements.txt`.
- Manifiesto: `entrega/manifiesto.json`. Entrenado con 38.274 filas de train, sin nuevo entrenamiento con train+test.
- Evidencia: informes en `reports/`, resultados en `experiments/`, contexto ampliado en `docs/CONTEXTO.md`.
- EDA reproducible: `notebooks/01_eda.py` y `.ipynb`.
- Resultados y análisis de errores: `notebooks/02_resultados.py` y `.ipynb`, ejecutada completa sin errores.
- Presentación: queda para una etapa posterior en Canva, por indicación del usuario. No se creó ni modificó un deck.

El archivo actual expone `predictor_v1_Grupo_9`, `predictor_v2_Grupo_9` y `predictor`: todas devuelven el mismo ndarray 0/1 en el mismo orden. v1 es una función delegada a v2 para el nombre del PDF; no se cambió modelo, features, corte ni entrenamiento.

## Puntos completados al 17/11/2026

1. Curvas de aprendizaje: `reports/curva_aprendizaje.md`, puntos y figura existentes.
2. Cohortes del baseline y del final: `experiments/analisis_consigna/cohortes.csv`; matrices, BA, AUC, recall, especificidad, FP/FN, soporte y contribución al error. Incluye las tres variables obligatorias y verifica la reconstrucción de las matrices globales.
3. Intervención prioritaria: reservas creadas con 0–7 **horas** de anticipación, frecuencia, impacto operativo posible, causas como hipótesis, nuevas señales y plan de evaluación. Informe: `reports/analisis_consigna.md`. No se confunde este campo con tiempo restante al predecir ni con momento de cancelación.
4. Adquisición de datos: `reports/adquisicion_datos.md`, con la corrección de unidades y relación con el diagnóstico.
5. Baseline explícito: logística calibrada sin assigned. Comparación consistente con GB/HistGB, BA principal y AUC secundaria. La búsqueda histórica usó BA/gap/simplicidad; no se reescribe su historia como una optimización multiobjetivo.
6. Entorno limpio local Python 3.12.12 con dependencias exactas: `entrega/compatibilidad_py312.json`, 300 etiquetas y scores idénticos, sin warnings. Nombre v1 verificado equivalente a v2, y empaquetado verificado contra el pipeline original sobre 600 filas de train.

El nuevo análisis usa exclusivamente las OOF existentes de TRAIN, sin entrenar ni abrir el holdout. Es posterior a la selección original, no se afirma que haya guiado retrospectivamente las búsquedas. Por la instrucción posterior del usuario de pushear todo a main, sus archivos y resultados se incluyen en Git junto con el contexto.

## Pendientes externos

1. Ejecutar `entrega/validacion_colab.ipynb` en Colab y conservar `compatibilidad_colab.json`. La prueba local de Python 3.12 no sustituye esa ejecución; `colab_verified` sigue en false.
2. Confirmar esquema oficial: `high_value_client`, `total_processes` y `total_gpu_hours` son entradas activas ausentes del diccionario del PDF. Se rechazan si faltan; no se inventan ni aproximan. Si no existen en el esquema oficial, habrá que evaluar una variante coherente con ese esquema.
3. Incorporar los hallazgos a Canva y preparar 10 minutos de exposición. Enviar presentación, predictor y código 24 horas antes de la clase de defensa. No se enviaron correos ni archivos a terceros.

El reentrenamiento train+test forma parte del plan interno del proyecto, no del requisito textual del PDF revisado. No se ejecutó ni se atribuye el holdout histórico a un modelo nuevo.

## Aclaración posterior de la cátedra

El uso es semanal sobre reservas activas, conserva el orden y usa un modelo ya entrenado. El test no es exclusivamente un período futuro. Asignaciones GPU y modificaciones pueden ocurrir días antes de ejecutar el trabajo: no afirmar que assigned es fuga comprobada por registrarse necesariamente al ejecutar. El candidato mantiene su exclusión acordada; el aviso cambia la justificación de disponibilidad, no obliga a modificar el modelo.

## Contrato de inferencia

DataFrame con entradas originales → ndarray 0/1, una etiqueta por fila, mismo orden. El predictor incluye preprocesamiento y umbral fijo; no requiere los scripts de desarrollo para predecir. BA y ROC-AUC se calculan en el evaluador con etiquetas, no son la salida de la función.

## Alcance de esta publicación

Se versionan el contexto técnico, el estado de entrega, el análisis de cohortes, sus resultados, la notebook ejecutada y las pruebas del predictor. Esta publicación en GitHub no es el envío del TP a la cátedra ni certifica la ejecución en Colab. Las entradas activas ausentes del diccionario oficial siguen pendientes de confirmación. El modelo, sus datos de entrenamiento, el payload y el umbral no cambiaron; v1 delega a v2 y conserva las mismas etiquetas 0/1.

No forman parte de este commit los entornos locales, cachés, temporales, instrucciones internas históricas ni archivos de presentación/Canva; se mantienen las exclusiones de `.gitignore`.
