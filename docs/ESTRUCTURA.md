# Mapa del proyecto

## Código vigente

| Ruta | Función |
| --- | --- |
| `src/compe/config.py` | Columnas, rutas y semilla del protocolo original. |
| `src/compe/data.py` | Carga y grupos por igualdad de entradas. |
| `src/compe/features.py` | Preprocesamiento aprendido dentro de los folds. |
| `src/compe/pipeline.py` | Variantes de preprocesamiento por familia. |
| `src/compe/models.py` | Constructores, búsquedas y simplicidad. |
| `src/compe/validation.py` | Folds, scores, umbral, métricas y subgrupos. |
| `src/compe/run.py` | Protocolo original por modelo; ejecutarlo puede entrenar. |
| `src/compe/resumen.py` | Resumen de los experimentos originales. |
| `entrega/predictor_v2_Grupo_9.py` | Único .py necesario para inferencia, con dependencias instaladas. |
| `entrega/construir_predictor.py` | Herramienta de empaquetado; ejecutar explícitamente, no al importar. |
| `entrega/verificar_predictor.py` | Verificación local contra el artefacto evaluado; solo usa train. |
| `entrega/validar_entorno.py` | Prueba autónoma de versiones y predicciones sobre el ejemplo, sin train/test. |
| `entrega/validacion_colab.ipynb` | Instalación y prueba en Colab; ejecución definitiva pendiente. |
| `entrega/compatibilidad_py312.json` | Evidencia de ejecución local limpia en Python 3.12; no es Colab. |
| `entrega/probar_predictor.py` | CLI CSV → probabilidades y clases, sin entrenar. |

Las herramientas auxiliares de `entrega/` no son archivos que el predictor necesite al recibir reservas. Se mantienen a su lado para que los comandos existentes sigan funcionando. El CSV de 300 filas y las predicciones esperadas son material de prueba funcional, no una evaluación independiente.

## Reproducción y evidencia experimental

- `experiments/<modelo>/`: resultados originales. No reorganizar sus archivos: informes y hashes dependen de esas rutas.
- `experiments/seleccion/calibracion.py` y `README.md`: calibración interna y robustez con parámetros fijos.
- `experiments/seleccion/stacking.py` y `README_STACKING.md`: comparación estricta HistGB/promedio/stacking.
- `experiments/seleccion/metricas_externas.py`: matrices y métricas desde predicciones guardadas, sin ajustar modelos.
- `experiments/seleccion/auditar_resultados.py` y `auditar_stacking_final.py`: auditorías reproducibles.
- `experiments/seleccion/revision*/`: controles fijos y generadores de informes; son herramientas de investigación, no inferencia.
- `experiments/evaluacion_final_histgb/`: evaluación ya realizada, modelo, manifiesto y predicciones congeladas. No usar como otra selección de modelos.
- `output/revision_modelos_20261116/`: auditoría posterior sobre train; incluye un control agrupado por entradas efectivas. Ver `docs/CONTEXTO.md`.
- `scripts/restaurar_artefactos.py` y `artifacts/manifest.json`: conservación sin pérdida de archivos grandes.

## Histórico

- `experiments/seleccion/historico/analisis_borrador.py`: borrador inicial reemplazado por los módulos de calibración y stacking. Contiene trabajo a nivel de módulo y supresión global de advertencias. Se conserva para trazabilidad; no usar para reproducir el resultado final.
- `experiments/seleccion/analisis.py`: entrada desactivada que explica los reemplazos; importarla no ejecuta ajustes.
- `reports/prompt_*.md`, `TODO.md`: instrucciones y plan de etapas anteriores. No describen por sí solos el estado actual.
- `experiments/logs/`, cachés de calibración y stacking: trazabilidad y reanudación; conservados.
- `output/defensa_tp3/`, `_v2/` y otras exportaciones: versiones de presentación; no confundir con la versión final acordada.

## Limpieza realizada

Se archivó el borrador reemplazado, se protegieron los puntos de entrada de construcción/verificación del predictor y se eliminó un import sin uso. Se retiraron únicamente cachés `__pycache__` y archivos `.DS_Store`; pueden regenerarse. Posteriormente se agregó el nombre `predictor_v1_Grupo_9` como delegación a v2. El payload del modelo, el umbral y las columnas requeridas se verificaron idénticos a la versión anterior. No se movieron fuentes que participan en hashes del modelo.
