"""Tabla comparativa de todos los modelos ya corridos: python -m compe.resumen"""
import json

import pandas as pd

from . import config as cfg

COLS = ["modelo", "familia", "ba_val", "ba_train", "gap", "ba_std", "roc_auc", "f1", "precision", "recall", "umbral",
        "ba_val_kfold_comun", "inflado_por_copias", "ba_val_sin_assigned", "combinaciones_probadas", "fit_seconds"]


def main():
    filas = [json.loads(p.read_text()) for p in sorted(cfg.EXPERIMENTS.glob("*/metricas.json"))]
    if not filas:
        print("Todavía no hay modelos corridos en experiments/.")
        return
    t = pd.DataFrame(filas)[COLS].sort_values("ba_val", ascending=False).round(4)
    t.to_csv(cfg.EXPERIMENTS / "resumen.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
