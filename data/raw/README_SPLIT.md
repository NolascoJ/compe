# División compartida del TP3

- Entrenamiento: **38,274 filas** (`train.csv`).
- Test reservado: **9,568 filas** (`test.csv`).
- Ambos archivos conservan las 34 columnas originales, incluido `canceled_job`.
- Semilla: `42`. Se aproxima un reparto 80/20 por clase sin separar filas idénticas.
- Se conservan las repeticiones: no hay grupos exactos compartidos entre conjuntos.
- Esta es una división aleatoria por grupos, no una evaluación temporal.

## Uso entre compañeros

1. Usar exactamente estos archivos; no volver a dividir el CSV completo.
2. Hacer EDA, ajuste de transformaciones, selección de modelos y ajuste del umbral
   solamente con `train.csv`, usando validación cruzada interna.
3. Dentro de entrenamiento, respetar `grupo_duplicados` al construir los folds.
   En `indices_split.csv`, filtrar `particion == 'train'`: sus filas mantienen
   el mismo orden que `train.csv`. El grupo sirve para separar, nunca como predictor.
4. Optimizar **balanced accuracy** en la validación interna. Reservar `test.csv`
   para una comparación final acordada; probar cada variante sobre test lo convertiría
   en validación y sesgaría su resultado.
5. Separar `canceled_job` como objetivo. Excluir `execution_status` y
   `execution_status_date` de las entradas del modelo: describen el resultado.
   Revisar además la disponibilidad de variables en el instante de predicción.

## Trazabilidad

`fila_original` es la posición desde cero en el CSV original, sin contar la cabecera.
`indices_split.csv` permite recuperar cada partición manteniendo el orden original.
`split_metadata.json` registra cantidades por clase, versiones y hashes SHA-256.
Los grupos se definen por igualdad de las 34 columnas, tratando nulos como iguales;
no identifican necesariamente una reserva ni un cliente únicos.

Para regenerar los archivos desde el repositorio original:

```sh
python dividir_dataset.py
```

La misma semilla y versión de pandas reproducen la selección; los archivos y hashes
compartidos son la referencia para comparar resultados.
