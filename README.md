# TP3 · Competencia de Machine Learning · Grupo 9

Proyecto para predecir `canceled_job`, la cancelación de una reserva de GPU. La métrica principal es **balanced accuracy (BA)**.

El candidato seleccionado es **HistGradientBoosting**, con 100 iteraciones y 31 hojas, sin `assigned_gpu_type` ni `gpu_changed`. Su evaluación histórica obtuvo BA externa de **84,83%** y **85,09%** en el holdout interno. Ese holdout no es el test oculto de los profesores.

## Contexto y resultados

- [Contexto técnico y decisiones actuales](docs/CONTEXTO.md): punto de entrada para continuar el proyecto.
- [Revisión de modelos y sobreajuste](output/revision_modelos_20261116/informe.md): auditoría de ocho familias, 669 configuraciones, ensembles y controles adicionales sobre TRAIN.
- [Evaluación final histórica de HistGB](reports/evaluacion_final_histgb.md).
- [Decisiones del EDA](reports/eda_decisiones.md).
- [Selección de modelos](reports/seleccion_modelo.md) y [revisión de ensembles](reports/seleccion_final_stacking.md).
- [Guía de trabajo original](CLAUDE.md) y [plan original](TODO.md).

Los documentos históricos se conservan. Algunos describen etapas previas o tareas entonces pendientes: consultar `docs/CONTEXTO.md` para distinguirlas del estado actual. Las presentaciones y los archivos de trabajo de Canva no forman parte de este repositorio.

## Instalación y artefactos grandes

```sh
git clone https://github.com/NolascoJ/compe.git
cd compe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/restaurar_artefactos.py
```

Un CSV de predicciones de 109.669.477 bytes se conserva como `.csv.gz`, sin pérdida, para respetar el límite de GitHub. El script restaura su ruta original y comprueba hashes SHA-256; no sobrescribe archivos diferentes. El resto de los datos, resultados, checkpoints y predictor histórico se incluyen directamente.

```sh
python scripts/restaurar_artefactos.py --check
```

`requirements-lock.txt` registra las versiones del entorno local al publicar, con Python 3.14.2. Las corridas principales usaron pandas 3.1.1 y scikit-learn 1.9.1. La compatibilidad del predictor con Colab debe verificarse; no se presume portabilidad de archivos serializados entre versiones.

## Estructura

| Directorio | Contenido |
| --- | --- |
| `src/compe/` | Datos, features, pipelines, modelos, validación y ejecución. |
| `data/raw/` | Train, holdout interno, índices, metadatos y reglas del split. |
| `notebooks/` | EDA en notebook y script pareados. |
| `experiments/` | Búsquedas, confirmaciones, métricas, scores, checkpoints y evaluación histórica. |
| `reports/` | Decisiones, informes y figuras originales. |
| `output/revision_modelos_20261116/` | Auditoría adicional, código reproducible y controles sobre TRAIN. |
| `artifacts/` | Manifiesto de archivos comprimidos. |
| `scripts/` | Restauración y verificación de artefactos. |

## Ejecución

```sh
PYTHONPATH=src python -m compe.run --listar
PYTHONPATH=src python experiments/seleccion/test_calibracion.py
PYTHONPATH=src python experiments/seleccion/test_stacking.py
python output/revision_modelos_20261116/auditar.py
```

Los comandos de entrenamiento y controles nuevos están documentados en sus informes. Ejecutarlos puede requerir tiempo y generar nuevos resultados; no hay que tratar una búsqueda posterior como validación independiente.

## Reglas para continuar

- Mantener el preprocesamiento dentro del pipeline y fuera del fold evaluado.
- Excluir target, estado de ejecución y fecha de estado de las entradas del predictor.
- Conservar la exclusión de GPU asignada acordada para el candidato final.
- No reutilizar el holdout ya consultado para elegir modelos, hiperparámetros o umbrales.
- Distinguir BA OOF con corte global de BA externa con cortes internos.
- No modificar retrospectivamente los resultados guardados al agregar controles nuevos.
