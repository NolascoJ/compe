"""Predict from a raw CSV without training. Example: python probar_predictor.py input.csv output.csv"""
import argparse
from pathlib import Path
import pandas as pd
from predictor_v2_Grupo_9 import probabilidades_v2_Grupo_9, UMBRAL

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('entrada',type=Path);p.add_argument('salida',type=Path);args=p.parse_args()
 if args.entrada.resolve()==args.salida.resolve():p.error('La salida no puede sobrescribir la entrada.')
 if args.salida.exists():p.error('La salida ya existe; elegí otra ruta.')
 x=pd.read_csv(args.entrada);scores=probabilidades_v2_Grupo_9(x)
 pd.DataFrame({'fila':range(len(x)),'probabilidad_cancelacion':scores,'prediccion':(scores>=UMBRAL).astype(int)}).to_csv(args.salida,index=False)
 print(f'{len(x)} predicciones guardadas en {args.salida}. Umbral fijo: {UMBRAL}')
if __name__=='__main__':main()
