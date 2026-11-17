"""Curva de aprendizaje (consigna 3.2): rendimiento según la cantidad de filas de entrenamiento.

Protocolo (solo TRAIN; no lee test.csv):
- Los 5 folds externos agrupados de siempre (`make_folds(df, seed=42)`). El fold de validación se evalúa
  siempre completo; solo se achica el entrenamiento.
- Dentro del entrenamiento externo de cada fold se toma una submuestra anidada por grupos de X idénticos
  (`feature_groups`): el 10% está contenido en el 20%, y así hasta el 100%. Las copias no se separan.
  Fracciones < 100% se repiten con 3 permutaciones (semillas 42, 43, 44).
- El corte se elige igual que en la selección del modelo: OOF de 3 folds internos agrupados hechos solo con
  la submuestra, `best_threshold`, y después se ajusta con toda la submuestra y se aplica al fold externo.
- Modelos con configuración fija, sin `assigned_gpu_type` ni `gpu_changed`: HistGB elegido y la logística de
  referencia. No se busca ningún hiperparámetro.
- Control: con el 100% se tiene que reproducir la BA externa de la calibración (84,83% y 82,01%).

Uso, desde la raíz del repo:
    PYTHONPATH=src .venv/bin/python experiments/curva_aprendizaje/curva_aprendizaje.py calcular --jobs 8
    PYTHONPATH=src .venv/bin/python experiments/curva_aprendizaje/curva_aprendizaje.py resumir
Si se corta, relanzar `calcular` sigue con las tareas que faltan (cache/).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from compe import config as cfg
from compe.data import feature_groups, load_train
from compe.models import MODELS
from compe.pipeline import make_pipeline
from compe.validation import best_threshold, make_folds, model_scores

AQUI = Path(__file__).resolve().parent
CACHE = AQUI / "cache"
FRACCIONES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
SEMILLAS = [42, 43, 44]
N_INTERNOS = 3
PARAMS = json.loads((cfg.EXPERIMENTS / "seleccion" / "parametros.json").read_text())
MODELOS = ["hist_gradient_boosting", "logistic"]
REFERENCIA_100 = {"hist_gradient_boosting": 0.8482931676954456, "logistic": 0.8201098033102495}


def ba(y, pred) -> float:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(bool)
    return 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())


def crear(modelo: str):
    spec = MODELS[modelo]
    return make_pipeline(spec.build(PARAMS[modelo]), spec.variante, use_assigned=False)


def puntuar(modelo: str, train: pd.DataFrame, evaluar: list[pd.DataFrame]) -> list[np.ndarray]:
    est = crear(modelo)
    est.fit(train, train[cfg.TARGET].to_numpy())
    return [np.asarray(model_scores(est, e, MODELS[modelo].score), dtype=float) for e in evaluar]


def submuestra(tr: np.ndarray, grupos: np.ndarray, fraccion: float, semilla: int) -> np.ndarray:
    """Filas de `tr` de una submuestra anidada por grupos completos (orden original)."""
    if fraccion >= 1.0:
        return np.sort(tr)
    g = grupos[tr]
    unicos, tam = np.unique(g, return_counts=True)
    orden = np.random.default_rng(semilla).permutation(len(unicos))
    acumulado = np.cumsum(tam[orden])
    k = int(np.searchsorted(acumulado, fraccion * len(tr))) + 1
    elegidos = set(unicos[orden[:k]].tolist())
    return np.sort(tr[np.fromiter((x in elegidos for x in g), bool, len(g))])


def tareas(folds) -> list[dict]:
    out = []
    for modelo in MODELOS:
        for f in range(len(folds)):
            for fr in FRACCIONES:
                for s in ([SEMILLAS[0]] if fr >= 1.0 else SEMILLAS):
                    out.append({"modelo": modelo, "fold": f, "fraccion": fr, "semilla": s,
                                "clave": f"{modelo}__f{f}__p{int(round(fr * 100)):03d}__s{s}"})
    return out


def calcular_tarea(t: dict, df: pd.DataFrame, folds, grupos: np.ndarray) -> dict:
    with threadpool_limits(1):
        t0 = time.time()
        tr, va = folds[t["fold"]]
        idx = submuestra(tr, grupos, t["fraccion"], t["semilla"])
        train = df.iloc[idx].reset_index(drop=True)
        ytr = train[cfg.TARGET].to_numpy().astype(int)
        yva = df[cfg.TARGET].to_numpy()[va].astype(int)

        oof = np.full(len(train), np.nan)
        for itr, iva in make_folds(train, n_splits=N_INTERNOS, seed=cfg.SEED):
            oof[iva] = puntuar(t["modelo"], train.iloc[itr], [train.iloc[iva]])[0]
        umbral, ba_interna = best_threshold(ytr, oof)

        s_va, s_tr = puntuar(t["modelo"], train, [df.iloc[va], train])
        p_va = s_va >= umbral
        return {**{k: v for k, v in t.items() if k != "clave"},
                "n_train": int(len(train)), "tasa_train": float(ytr.mean()), "n_val": int(len(va)),
                "umbral": float(umbral), "ba_interna": float(ba_interna),
                "ba_val": ba(yva, p_va), "ba_train": ba(ytr, s_tr >= umbral),
                "recall_val": float(p_va[yva == 1].mean()), "especificidad_val": float((~p_va[yva == 0]).mean()),
                "tp": int((p_va & (yva == 1)).sum()), "fn": int((~p_va & (yva == 1)).sum()),
                "tn": int((~p_va & (yva == 0)).sum()), "fp": int((p_va & (yva == 0)).sum()),
                "auc_val": float(roc_auc_score(yva, s_va)), "auc_train": float(roc_auc_score(ytr, s_tr)),
                "segundos": time.time() - t0}


def calcular(jobs: int) -> None:
    df = load_train()
    folds = make_folds(df, seed=cfg.SEED)
    grupos = feature_groups(df).to_numpy()
    CACHE.mkdir(parents=True, exist_ok=True)
    todas = tareas(folds)
    pendientes = [t for t in todas if not (CACHE / f"{t['clave']}.json").exists()]
    print(f"{len(todas)} tareas, {len(pendientes)} pendientes", flush=True)
    t0 = time.time()
    gen = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(lambda t: (t, calcular_tarea(t, df, folds, grupos)))(t) for t in pendientes)
    for i, (t, r) in enumerate(gen, 1):
        tmp = CACHE / f"{t['clave']}.json.tmp"
        tmp.write_text(json.dumps(r))
        tmp.rename(CACHE / f"{t['clave']}.json")
        print(f"[{i}/{len(pendientes)}] {t['clave']} ba_val={r['ba_val']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    print("COMPLETO", flush=True)


def resumir() -> None:
    import sklearn
    filas = [json.loads(p.read_text()) for p in sorted(CACHE.glob("*.json"))]
    pts = pd.DataFrame(filas).sort_values(["modelo", "fraccion", "fold", "semilla"])
    esperadas = len(MODELOS) * 5 * (len(FRACCIONES) - 1) * len(SEMILLAS) + len(MODELOS) * 5
    assert len(pts) == esperadas, f"faltan tareas: {len(pts)} de {esperadas}"
    pts.to_csv(AQUI / "puntos.csv", index=False)

    # Primero se promedian las semillas dentro de cada fold; el desvío es entre los 5 folds.
    por_fold = pts.groupby(["modelo", "fraccion", "fold"]).mean(numeric_only=True).reset_index()
    met = ["ba_val", "ba_train", "auc_val", "auc_train", "ba_interna", "umbral", "recall_val", "especificidad_val"]
    res = por_fold.groupby(["modelo", "fraccion"]).agg(
        n_train=("n_train", "mean"), **{f"{m}": (m, "mean") for m in met},
        **{f"{m}_std": (m, lambda x: float(np.std(x))) for m in ["ba_val", "ba_train", "auc_val"]}).reset_index()
    res["gap_ba"] = res["ba_train"] - res["ba_val"]
    res.to_csv(AQUI / "resumen.csv", index=False)

    control = {}
    for m, ref in REFERENCIA_100.items():
        sub = pts[(pts.modelo == m) & (pts.fraccion == 1.0)]
        # BA global de los 5 folds juntos, igual que la calibración (no promedio de folds).
        tp, fn, tn, fp = (int(sub[c].sum()) for c in ["tp", "fn", "tn", "fp"])
        ba_global = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
        control[m] = {"referencia_calibracion": ref, "ba_global_100": ba_global,
                      "diferencia": ba_global - ref, "ba_media_folds": float(sub.ba_val.mean())}
    (AQUI / "manifest.json").write_text(json.dumps({
        "fracciones": FRACCIONES, "semillas_submuestra": SEMILLAS, "folds_externos": 5, "folds_internos": N_INTERNOS,
        "parametros": {m: PARAMS[m] for m in MODELOS}, "use_assigned": False, "test_csv_leido": False,
        "control_100": control, "python": platform.python_version(), "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__}, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print(json.dumps(control, indent=2))


C_TRAIN = "#00799c"   # petróleo de la presentación (validado: croma, CVD y contraste)
C_VAL = "#d9661a"     # naranja de la presentación
NOMBRES = {"hist_gradient_boosting": "HistGradientBoosting", "logistic": "Regresión logística"}


def graficar() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from compe.plots import INK, INK2, set_style

    set_style()
    res = pd.read_csv(AQUI / "resumen.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    miles = FuncFormatter(lambda v, _: f"{v / 1000:.0f}k")
    for ax, m in zip(axes, MODELOS):
        r = res[res.modelo == m].sort_values("n_train")
        x = r.n_train.to_numpy()
        cerca = abs(r.ba_train.iloc[-1] - r.ba_val.iloc[-1]) < 0.03   # separar etiquetas si se pisan
        for col, color, etiqueta, dy in [("ba_train", C_TRAIN, "Entrenamiento", 14 if cerca else 0),
                                         ("ba_val", C_VAL, "Validación", -14 if cerca else 0)]:
            y = 100 * r[col].to_numpy()
            s = 100 * r[f"{col}_std"].to_numpy()
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.15, lw=0)
            ax.plot(x, y, color=color, lw=2, marker="o", ms=4.5, markeredgecolor="white", markeredgewidth=1)
            ax.annotate(f"{etiqueta}\n{y[-1]:.1f}%".replace(".", ","), (x[-1], y[-1]), xytext=(8, dy),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        ax.set_title(NOMBRES[m], color=INK)
        ax.set_xlabel("Filas de entrenamiento")
        ax.xaxis.set_major_formatter(miles)
        ax.set_xlim(0, x[-1] * 1.22)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].set_ylim(70, 100)
    fig.text(0.0, -0.04, "5 folds externos agrupados; corte elegido con 3 folds internos de cada submuestra. "
             "Banda: ±1 desvío entre folds. Solo TRAIN, sin GPU asignada.", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(AQUI / "curva_aprendizaje.png")
    print(AQUI / "curva_aprendizaje.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["calcular", "resumir", "graficar"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    {"calcular": lambda: calcular(a.jobs), "resumir": resumir, "graficar": graficar}[a.etapa]()
    sys.exit(0)
