# Estado de la entrega

## Archivos y alcance

- Predictor: `entrega/predictor_v2_Grupo_9.py`.
- Instrucciones y dependencias: `entrega/README.md`, `entrega/requirements.txt`.
- Manifiesto: `entrega/manifiesto.json`. Entrenado con 38.274 filas de train, sin nuevo entrenamiento con train+test.
- Evidencia: informes en `reports/`, resultados en `experiments/`, contexto ampliado en `docs/CONTEXTO.md`.
- EDA reproducible: `notebooks/01_eda.py` y `.ipynb`.
- Presentación de trabajo: `output/defensa_tp3_v3/`; confirmar la versión definitiva fuera de las exportaciones históricas.

El nombre de función del PDF es `predictor_v1_Grupo_X`; la reentrega actual usa `predictor_v2_Grupo_9` y alias `predictor`. Confirmar el contrato de nombre requerido sin cambiarlo por suposición.

## Pendientes encontrados al leer el enunciado

1. Curvas de aprendizaje variando cantidad de datos, con discusión de limitación por datos/modelo. Las curvas de hiperparámetros no las sustituyen.
2. Errores del baseline por `client_type`, `service_tier` y `reserved_gpu_type`: cohortes, cantidades, FP y FN. El EDA descriptivo no equivale a ese análisis.
3. Selección de una cohorte para intervención: frecuencia, impacto operativo, posibles causas y próximos pasos.
4. Propuesta justificada de adquisición de datos adicionales.
5. Identificar explícitamente el baseline y justificar el papel de BA y ROC-AUC en las decisiones.
6. Compatibilidad de sklearn del predictor: hubo una ejecución funcional en Colab, con advertencias 1.9.1 → 1.6.1; aún no valida portabilidad.
7. Confirmar entradas: los totales y `high_value_client` usados no figuran en el diccionario del PDF. No rellenarlos arbitrariamente.
8. Preparar presentación de 10 minutos y enviar presentación, predictor y código 24 horas antes de la defensa, según el enunciado.

Estas tareas son diagnóstico de faltantes; esta reorganización no las implementa ni cambia el candidato.

## Aclaración posterior de la cátedra

El uso es semanal sobre reservas activas, conserva el orden y usa un modelo ya entrenado. El test no es exclusivamente un período futuro. Asignaciones GPU y modificaciones pueden ocurrir días antes de ejecutar el trabajo: no afirmar que assigned es fuga comprobada por registrarse necesariamente al ejecutar. El candidato mantiene su exclusión acordada; el aviso cambia la justificación de disponibilidad, no obliga a modificar el modelo.

## Contrato de inferencia

DataFrame con entradas originales → ndarray 0/1, una etiqueta por fila, mismo orden. El predictor incluye preprocesamiento y umbral fijo; no requiere los scripts de desarrollo para predecir. BA y ROC-AUC se calculan en el evaluador con etiquetas, no son la salida de la función.
