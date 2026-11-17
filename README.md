# TP3 · Machine Learning · Grupo 9

Predicción de cancelaciones de reservas GPU (`canceled_job`). Métrica principal: balanced accuracy (BA); métrica complementaria: ROC-AUC.

El predictor elegido es **HistGradientBoosting**, 100 iteraciones y 31 hojas, sin GPU asignada ni `gpu_changed`, con umbral fijo **0,38294747099493937**. BA externa histórica: **84,83%**; holdout interno ya evaluado: **85,09%**. El test oculto de la competencia es un conjunto distinto.

## Para la entrega

- **Archivo predictor:** [entrega/predictor_v2_Grupo_9.py](entrega/predictor_v2_Grupo_9.py). Autocontenido; recibe un DataFrame y devuelve un 0/1 por fila en orden. Incluye el modelo entrenado solo con train; no es un reentrenamiento con train+test.
- **Uso, dependencias y prueba:** [entrega/README.md](entrega/README.md).
- **Estado y pendientes del TP:** [docs/ENTREGA.md](docs/ENTREGA.md).
- **Mapa del código y archivos históricos:** [docs/ESTRUCTURA.md](docs/ESTRUCTURA.md).
- **Presentación de trabajo:** `output/defensa_tp3_v3/`; las variantes anteriores son históricas. Estos directorios están excluidos de Git por la configuración existente; comprobar que la versión enviada sea la correcta.

El archivo ya produjo predicciones en Colab con sklearn 1.6.1, pero emitió advertencias al cargar un modelo de sklearn 1.9.1. **La compatibilidad de versiones sigue pendiente.** El esquema de entradas también debe cotejarse con el evaluador: el predictor requiere los dos totales y `high_value_client`, no enumerados en el diccionario del PDF.

## Instalación para desarrollo

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/restaurar_artefactos.py
```

`pyproject.toml` declara dependencias de desarrollo con rangos; no fija el entorno de inferencia. `requirements-lock.txt` registra el entorno local histórico (Python 3.14.2); `entrega/requirements.txt` contiene las versiones del predictor empaquetado. No asumir compatibilidad de modelos serializados entre versiones.

Un CSV grande se conserva también como `.csv.gz`, sin pérdida. `scripts/restaurar_artefactos.py` verifica los hashes del manifiesto antes de restaurarlo; `--check` verifica sin restaurar. Si Git LFS gestiona esos archivos en otra copia del repo, conservar también su configuración.

## Reproducir o consultar resultados

```sh
# Listar modelos, sin entrenar:
PYTHONPATH=src python -m compe.run --listar

# Ver planes, sin ejecutar ajustes:
PYTHONPATH=src python experiments/seleccion/calibracion.py todo --dry-run
PYTHONPATH=src python experiments/seleccion/stacking.py todo --dry-run
```

Los comandos largos y de verificación están en las guías de cada etapa, enlazadas desde [docs/ESTRUCTURA.md](docs/ESTRUCTURA.md). Algunos scripts históricos regeneran informes o ajustan modelos al ejecutarse: revisar su alcance antes de usarlos.

## Evidencia y decisiones

- [Contexto y revisión adicional del 16/11](docs/CONTEXTO.md).
- [EDA](reports/eda_decisiones.md).
- [Selección y ensembles](reports/seleccion_final_stacking.md).
- [Control comparable del gap](reports/control_gap_sin_assigned.md).
- [Evaluación final histórica](reports/evaluacion_final_histgb.md).
- [Auditoría adicional sobre train](output/revision_modelos_20261116/informe.md).
- `CLAUDE.md`: reglas del proyecto. `TODO.md`: plan histórico; su estado no refleja todas las corridas terminadas.

El preprocesamiento se aprende dentro de cada entrenamiento. Target, estado y fecha de estado no entran al predictor. El holdout interno ya fue consultado: no volver a usarlo para seleccionar alternativas como si fuera una evaluación independiente nueva. Los resultados y manifiestos históricos se conservan como evidencia.
