# TP2 - Bank Marketing

## Instalación

Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecución completa

```bash
python ejecutar_tp2.py
```

Descarga el dataset, separa train/test, ejecuta la validación y las búsquedas
de hiperparámetros, compara los modelos y ajusta el umbral de Random Forest
con predicciones OOF. Finalmente evalúa en test.

La descarga requiere internet. Las búsquedas pueden tardar varios minutos.
Los resultados y gráficos se guardan en `resultados/`.

## Ejecutar por partes

Ver los comandos sin ejecutarlos:

```bash
python ejecutar_tp2.py --ver-plan
```

Con los datos ya preparados, ejecutar la comparación, el ajuste del umbral
y la evaluación final:

```bash
python ejecutar_tp2.py --desde comparacion
```

Ejecutar solo el OOF de Random Forest, el ajuste del umbral y test:

```bash
python -m modelos.ajuste_umbral_rf
```

Para retomar una ejecución, usar `--desde` con una de estas etapas:
`descarga`, `particion`, `validacion`, `curvas`, `ajuste_rf`, `ajuste_svm`,
`comparacion` o `umbral_rf`. Se puede limitar la ejecución con `--hasta`.

## EDA

Se ejecuta por separado, después de preparar los datos:

```bash
python -m datos.exportar_dataset
python -m datos.separar_datos
python -m analisis.eda_inicial
python -m analisis.analizar_outliers
```
