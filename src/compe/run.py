"""Ejecutor por modelo: corre el protocolo de ajuste y guarda todo en experiments/<modelo>/.

Uso:
    python -m compe.run <modelo>                  # protocolo completo
    python -m compe.run <modelo> --etapa curva    # una sola etapa
    python -m compe.run <modelo> --rapido         # prueba corta para verificar que corre
    python -m compe.run --listar

Etapas (en orden) y archivos que deja cada una:
    base        base.json                 metricas con los valores por defecto
    curva       curva_validacion.csv/png  entrenamiento vs validacion segun la complejidad
    busqueda    busqueda.csv              busqueda amplia (una fila por combinacion)
    refinar     refinamiento.csv          busqueda estrecha alrededor del 10% mejor
    confirmar   confirmacion.csv          las 3 mejores, con 3 particiones distintas
    final       mejores_parametros.json, metricas.json, subgrupos.csv, oof.npy

La busqueda se guarda combinacion por combinacion: si se corta, al relanzar sigue donde quedo.
"""
import argparse
import json
import math
import time
import warnings

import numpy as np
import pandas as pd

from . import config as cfg
from .data import load_train
from .models import MODELS, refine_space, sample_space
from .pipeline import make_pipeline
from .validation import cv_scores, make_folds, metrics, subgroup_table

ETAPAS = ["base", "curva", "busqueda", "refinar", "confirmar", "final"]
METRIC_COLS = ["ba_val", "ba_train", "gap", "ba_std", "umbral", "roc_auc", "f1", "precision", "recall",
               "completados_detectados", "fit_seconds"]


def _dumps(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def evaluate(spec, params: dict, df, folds, use_assigned: bool = True):
    pipe = make_pipeline(spec.build(params), spec.variante, use_assigned=use_assigned)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cv_scores(pipe, df, folds, spec.score, n_jobs=spec.fold_jobs)
    return metrics(df, res, folds), res


def select(rows: pd.DataFrame, spec, max_gap: float, n_folds: int = 5) -> pd.Series:
    """Regla de seleccion (paso 4): descartar diferencias grandes entre entrenamiento y validacion;
    entre las que quedan a menos de un error estandar de la mejor, elegir la mas simple."""
    ok = rows[rows["gap"] <= max_gap]
    if ok.empty:
        ok = rows
    best = ok.loc[ok["ba_val"].idxmax()]
    se = best["ba_std"] / math.sqrt(n_folds)
    cands = ok[ok["ba_val"] >= best["ba_val"] - se]
    key = cands["params"].map(lambda s: spec.simplicity(json.loads(s)))
    return cands.loc[key.map(lambda k: tuple(k)).sort_values().index[0]]


def _search(spec, combos, df, folds, path, etiqueta):
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["params", *METRIC_COLS])
    vistos = set(done["params"])
    pendientes = [p for p in combos if _dumps(p) not in vistos]
    print(f"[{spec.nombre}] {etiqueta}: {len(pendientes)} combinaciones por correr ({len(vistos)} ya guardadas)")
    for i, p in enumerate(pendientes, 1):
        m, _ = evaluate(spec, p, df, folds)
        fila = pd.DataFrame([{"params": _dumps(p), **{k: m[k] for k in METRIC_COLS}}])
        fila.to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"  {i}/{len(pendientes)}  val {m['ba_val']:.4f}  train {m['ba_train']:.4f}  dif {m['gap']:+.4f}  {m['fit_seconds']:.0f}s")
    return pd.read_csv(path) if path.exists() else done


def etapa_base(spec, df, folds, out, a):
    m, _ = evaluate(spec, {}, df, folds)
    (out / "base.json").write_text(json.dumps({"params": "valores por defecto", **m}, indent=2))
    print(f"[{spec.nombre}] base: val {m['ba_val']:.4f}  train {m['ba_train']:.4f}  dif {m['gap']:+.4f}")


def etapa_curva(spec, df, folds, out, a):
    nombre, valores = spec.curve
    if a.rapido:
        valores = valores[:3]
    filas = []
    for v in valores:
        m, _ = evaluate(spec, {nombre: v}, df, folds)
        filas.append({nombre: "sin limite" if v is None else v, **{k: m[k] for k in METRIC_COLS}})
        print(f"[{spec.nombre}] curva {nombre}={v}: val {m['ba_val']:.4f}  train {m['ba_train']:.4f}")
    t = pd.DataFrame(filas)
    t.to_csv(out / "curva_validacion.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import C_SINL, C_TODO, INK2, set_style
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(t))
    ax.plot(x, t["ba_train"], color=C_TODO, lw=2, marker="o", label="entrenamiento")
    ax.plot(x, t["ba_val"], color=C_SINL, lw=2, marker="o", label="validación (fuera de fold)")
    ax.fill_between(x, t["ba_val"] - t["ba_std"], t["ba_val"] + t["ba_std"], color=C_SINL, alpha=0.15, lw=0)
    i = int(t["ba_val"].idxmax())
    ax.annotate(f"máx. {t['ba_val'][i]:.3f}", (x[i], t["ba_val"][i]), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in t[nombre]])
    ax.set_xlabel(nombre); ax.set_ylabel("balanced accuracy"); ax.legend()
    ax.set_title(f"{spec.nombre}: curva de validación")
    fig.savefig(out / "curva_validacion.png"); plt.close(fig)


def etapa_busqueda(spec, df, folds, out, a):
    n = a.n_iter or (3 if a.rapido else spec.n_iter)
    _search(spec, sample_space(spec.space, n), df, folds, out / "busqueda.csv", "búsqueda amplia")


def etapa_refinar(spec, df, folds, out, a):
    amplia = pd.read_csv(out / "busqueda.csv")
    k = max(3, math.ceil(len(amplia) * 0.10))
    top = [json.loads(s) for s in amplia[amplia["gap"] <= a.max_gap].nlargest(k, "ba_val")["params"]] or \
          [json.loads(s) for s in amplia.nlargest(k, "ba_val")["params"]]
    n = a.n_iter or (2 if a.rapido else max(10, spec.n_iter // 3))
    ya = set(amplia["params"])
    combos = [p for p in sample_space(refine_space(spec.space, top), n, seed=cfg.SEED + 1) if _dumps(p) not in ya]
    if not combos:
        # Finite grids (Ridge is 25 alpha values x 2 class weights) may be
        # exhausted completely by the broad search.  In that case there is
        # nothing meaningful to refine: record the stage as an empty,
        # well-formed result and let confirmation use busqueda.csv.
        path = out / "refinamiento.csv"
        if not path.exists():
            pd.DataFrame(columns=["params", *METRIC_COLS]).to_csv(path, index=False)
        print(f"[{spec.nombre}] refinamiento omitido: la búsqueda amplia ya agotó el espacio candidato")
        return
    _search(spec, combos, df, folds, out / "refinamiento.csv", "refinamiento")


def _todas(out) -> pd.DataFrame:
    partes = []
    for f in ("busqueda.csv", "refinamiento.csv"):
        path = out / f
        if path.exists():
            parte = pd.read_csv(path)
            if not parte.empty:
                partes.append(parte)
    return pd.concat(partes, ignore_index=True).drop_duplicates("params") if partes else pd.DataFrame()


def etapa_confirmar(spec, df, folds, out, a):
    todas = _todas(out)
    ok = todas[todas["gap"] <= a.max_gap]
    top = (ok if len(ok) else todas).nlargest(2 if a.rapido else 3, "ba_val")
    semillas = [cfg.SEED, cfg.SEED + 1] if a.rapido else [cfg.SEED, cfg.SEED + 1, cfg.SEED + 2]
    filas = []
    for s in top["params"]:
        ms = [evaluate(spec, json.loads(s), df, make_folds(df, seed=sem))[0] for sem in semillas]
        fila = {"params": s, **{k: float(np.mean([m[k] for m in ms])) for k in METRIC_COLS}}
        fila["ba_entre_particiones_std"] = float(np.std([m["ba_val"] for m in ms]))
        filas.append(fila)
        print(f"[{spec.nombre}] confirmación: val {fila['ba_val']:.4f} ± {fila['ba_entre_particiones_std']:.4f}  dif {fila['gap']:+.4f}")
    pd.DataFrame(filas).to_csv(out / "confirmacion.csv", index=False)


def etapa_final(spec, df, folds, out, a):
    if (out / "confirmacion.csv").exists():
        cand, origen = pd.read_csv(out / "confirmacion.csv"), "confirmacion"
    else:
        cand, origen = _todas(out), "busqueda"
    if len(cand):
        elegido = select(cand, spec, a.max_gap)
        params = json.loads(elegido["params"])
    else:
        params, origen = {}, "valores por defecto"
    m, res = evaluate(spec, params, df, folds)
    np.save(out / "oof.npy", res.oof)
    subgroup_table(df, res.oof, m["umbral"]).round(4).to_csv(out / "subgrupos.csv", index=False)
    # Controles: K-Fold comun (copias repartidas entre folds) y sin assigned_gpu_type.
    m_kfold, _ = evaluate(spec, params, df, make_folds(df, grouped=False))
    m_sin, res_sin = evaluate(spec, params, df, folds, use_assigned=False)
    np.save(out / "oof_sin_assigned.npy", res_sin.oof)
    (out / "mejores_parametros.json").write_text(json.dumps({"modelo": spec.nombre, "origen": origen, "params": params}, indent=2))
    resumen = {
        "modelo": spec.nombre, "familia": spec.familia, "variante_preprocesamiento": spec.variante, "puntuacion": spec.score,
        **m,
        "ba_val_kfold_comun": m_kfold["ba_val"], "inflado_por_copias": m_kfold["ba_val"] - m["ba_val"],
        "ba_val_sin_assigned": m_sin["ba_val"], "umbral_sin_assigned": m_sin["umbral"],
        "combinaciones_probadas": int(len(_todas(out))), "max_gap": a.max_gap,
    }
    (out / "metricas.json").write_text(json.dumps(resumen, indent=2))
    print(f"[{spec.nombre}] FINAL ({origen}): val {m['ba_val']:.4f}  train {m['ba_train']:.4f}  dif {m['gap']:+.4f}  "
          f"auc {m['roc_auc']:.4f} | K-Fold común {m_kfold['ba_val']:.4f} | sin assigned {m_sin['ba_val']:.4f}")


FUNCIONES = {"base": etapa_base, "curva": etapa_curva, "busqueda": etapa_busqueda, "refinar": etapa_refinar,
             "confirmar": etapa_confirmar, "final": etapa_final}


def main():
    ap = argparse.ArgumentParser(description="Ajuste de un modelo siguiendo el protocolo del TODO.md")
    ap.add_argument("modelo", nargs="?", choices=sorted(MODELS))
    ap.add_argument("--etapa", choices=ETAPAS + ["todo"], default="todo")
    ap.add_argument("--n-iter", type=int, help="combinaciones a probar (por defecto, las del registro)")
    ap.add_argument("--max-gap", type=float, default=0.05, help="diferencia máxima entrenamiento-validación admitida")
    ap.add_argument("--rapido", action="store_true", help="prueba corta; guarda en experiments/_prueba/")
    ap.add_argument("--listar", action="store_true")
    a = ap.parse_args()
    if a.listar or not a.modelo:
        for s in MODELS.values():
            print(f"{s.nombre:24s} {s.familia:12s} variante {s.variante}  {s.n_iter} combinaciones  curva: {s.curve[0]}")
        return
    spec = MODELS[a.modelo]
    out = cfg.EXPERIMENTS / ("_prueba" if a.rapido else "") / spec.nombre
    out.mkdir(parents=True, exist_ok=True)
    df = load_train()
    folds = make_folds(df)
    t0 = time.time()
    for e in (ETAPAS if a.etapa == "todo" else [a.etapa]):
        FUNCIONES[e](spec, df, folds, out, a)
    print(f"[{spec.nombre}] listo en {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
