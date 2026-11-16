"""Punto de corte calibrado adentro de cada entrenamiento, y robustez, con configuraciones FIJAS.

No elige modelos ni busca hiperparametros: usa las configuraciones de `parametros.json`.
Nunca lee `data/raw/test.csv`. Instrucciones de uso en `README.md` (esta carpeta).

Unidades de trabajo (cada una se guarda por separado y se puede reanudar):
  ajuste   ajustar el pipeline en un entrenamiento y puntuar su conjunto de evaluacion (1 ajuste)
  interno  OOF internos de 3 folds agrupados dentro de un entrenamiento -> punto de corte (3 ajustes)

Escenarios:
  cv             los 5 folds externos de `make_folds(df, seed=42)` (los mismos del protocolo original)
  eval_anio_X    entrenar con los otros anios, evaluar en X
  eval_PRT       entrenar sin PRT, evaluar en PRT
  eval_resto     entrenar solo con PRT, evaluar en el resto
  sin_locked     5 folds agrupados armados sobre las filas sin locked_allocation (entrena y evalua sin ellas)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import time
import traceback
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from compe import config as cfg
from compe.data import load_train
from compe.models import MODELS
from compe.pipeline import make_pipeline
from compe.validation import best_threshold, make_folds, model_scores, subgroup_table

AQUI = Path(__file__).resolve().parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))
import metricas_externas  # noqa: E402
SALIDA = AQUI / "calibracion"
PARAMETROS = AQUI / "parametros.json"

# Subir este numero si cambia la forma de entrenar o puntuar en este archivo: invalida los caches.
VERSION_CALCULO = "1"
CODIGO_RELEVANTE = ["config.py", "data.py", "features.py", "pipeline.py", "models.py", "validation.py"]

MODELOS = ["logistic", "ridge", "linear_svc", "knn", "random_forest", "bagging", "gradient_boosting", "hist_gradient_boosting"]
FINALISTAS = ["logistic", "gradient_boosting", "hist_gradient_boosting"]
SEMILLA = cfg.SEED
N_EXTERNOS = 5
N_INTERNOS = 3
FILAS_REFERENCIA = 38274 * (N_EXTERNOS - 1) / N_EXTERNOS   # tamano de entrenamiento de un fold en los experimentos

log = logging.getLogger("calibracion")


# ---------------------------------------------------------------- utilidades

def variante(assigned: bool) -> str:
    return "con_assigned" if assigned else "sin_assigned"


def ba_segura(y: np.ndarray, pred: np.ndarray) -> float:
    """Balanced accuracy; NaN si falta alguna de las dos clases (queda indefinida)."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(bool)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    return float(0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean()))


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_indices(idx) -> str:
    return _hash_bytes(np.asarray(idx, dtype=np.int64).tobytes())[:16]


def hash_archivo(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def hash_codigo() -> str:
    src = Path(cfg.__file__).resolve().parent
    partes = [VERSION_CALCULO.encode()] + [(src / f).read_bytes() for f in CODIGO_RELEVANTE]
    return _hash_bytes(b"\0".join(partes))[:16]


def versiones() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__}


def folds_externos(df: pd.DataFrame):
    return make_folds(df, n_splits=N_EXTERNOS, seed=SEMILLA)


def folds_internos(train: pd.DataFrame):
    """Indices RELATIVOS a `train` (posiciones 0..len(train)-1), nunca al dataframe global."""
    return make_folds(train.reset_index(drop=True), n_splits=N_INTERNOS, seed=SEMILLA)


# ---------------------------------------------------------------- contexto y tareas

@dataclass(eq=False)
class Tarea:
    tipo: str                      # "ajuste" o "interno"
    modelo: str
    assigned: bool
    escenario: str
    fold: int | None
    train_idx: np.ndarray          # posiciones en el dataframe del contexto (orden de train.csv)
    eval_idx: np.ndarray | None    # solo para "ajuste"

    @property
    def clave(self) -> str:
        sufijo = "" if self.fold is None else f"__f{self.fold}"
        return f"{self.tipo}__{self.modelo}__{variante(self.assigned)}__{self.escenario}{sufijo}"

    @property
    def ajustes(self) -> int:
        return 1 if self.tipo == "ajuste" else N_INTERNOS


@dataclass(eq=False)
class Contexto:
    df: pd.DataFrame
    params: dict
    raiz: Path
    hash_train: str
    factory: Callable | None = None       # solo para pruebas: (modelo, assigned) -> estimador
    factory_nombre: str = "make_pipeline"
    comparar_experimentos: bool = True
    y: np.ndarray = field(init=False)
    folds: list = field(init=False)
    codigo: str = field(init=False)
    vers: dict = field(init=False)

    def __post_init__(self):
        self.y = self.df[cfg.TARGET].to_numpy().astype(int)
        self.folds = folds_externos(self.df)
        self.codigo = hash_codigo()
        self.vers = versiones()
        self.raiz = Path(self.raiz)

    @property
    def cache(self) -> Path:
        return self.raiz / "cache"

    @property
    def resultados(self) -> Path:
        return self.raiz / "resultados"

    def crear(self, modelo: str, assigned: bool):
        if self.factory is not None:
            return self.factory(modelo, assigned)
        spec = MODELS[modelo]
        return make_pipeline(spec.build(self.params[modelo]), spec.variante, use_assigned=assigned)


def escenarios_robustez(ctx: Contexto) -> list[tuple]:
    """(nombre, fold, train_idx, eval_idx) con posiciones globales."""
    df = ctx.df
    out = []
    for anio in sorted(df["execution_year"].unique()):
        m = (df["execution_year"] == anio).to_numpy()
        out.append((f"eval_anio_{int(anio)}", None, np.flatnonzero(~m), np.flatnonzero(m)))
    prt = (df["client_region"] == "PRT").to_numpy()
    out.append(("eval_PRT", None, np.flatnonzero(~prt), np.flatnonzero(prt)))
    out.append(("eval_resto", None, np.flatnonzero(prt), np.flatnonzero(~prt)))
    pos = np.flatnonzero((df["reservation_commitment"] != "locked_allocation").to_numpy())
    sub = df.iloc[pos].reset_index(drop=True)
    for k, (tr, va) in enumerate(make_folds(sub, n_splits=N_EXTERNOS, seed=SEMILLA)):
        out.append(("sin_locked", k, pos[tr], pos[va]))
    return out


def planificar(ctx: Contexto, etapa: str, modelos: list[str], modo_assigned: str) -> list[Tarea]:
    def vars_de(m, robustez=False):
        if modo_assigned == "con":
            return [True]
        if modo_assigned == "sin":
            return [False]
        if modo_assigned == "ambos" or robustez:
            return [True, False]
        return [True, False] if m in FINALISTAS else [True]   # auto: lo que pide el plan

    tareas = []
    if etapa in ("oof", "calibracion", "todo"):
        for m in modelos:
            for a in vars_de(m):
                for k, (tr, va) in enumerate(ctx.folds):
                    tareas.append(Tarea("ajuste", m, a, "cv", k, tr, va))
                    if etapa != "oof":
                        tareas.append(Tarea("interno", m, a, "cv", k, tr, None))
    if etapa in ("robustez", "todo"):
        esc = escenarios_robustez(ctx)
        for m in modelos:
            if etapa == "todo" and m not in FINALISTAS:
                continue
            for a in vars_de(m, robustez=True):
                for nombre, k, tr, va in esc:
                    tareas.append(Tarea("ajuste", m, a, nombre, k, tr, va))
                    tareas.append(Tarea("interno", m, a, nombre, k, tr, None))
    return tareas


# ---------------------------------------------------------------- calculo

def _ajustar_y_puntuar(ctx: Contexto, modelo: str, assigned: bool, train: pd.DataFrame, evaluar: pd.DataFrame) -> np.ndarray:
    est = ctx.crear(modelo, assigned)
    est.fit(train, train[cfg.TARGET].to_numpy())
    return np.asarray(model_scores(est, evaluar, MODELS[modelo].score), dtype=float)


def calcular(ctx: Contexto, t: Tarea) -> dict:
    """Calculo puro de una tarea. Solo ve ctx.df.iloc[t.train_idx] para aprender; eval_idx solo se puntua."""
    train = ctx.df.iloc[t.train_idx].reset_index(drop=True)
    if t.tipo == "ajuste":
        scores = _ajustar_y_puntuar(ctx, t.modelo, t.assigned, train, ctx.df.iloc[t.eval_idx])
        return {"scores": scores, "eval_idx": np.asarray(t.eval_idx, np.int64)}
    ytr = train[cfg.TARGET].to_numpy().astype(int)
    oof = np.full(len(train), np.nan)
    fold_interno = np.full(len(train), -1, dtype=np.int64)
    for j, (itr, iva) in enumerate(folds_internos(train)):
        oof[iva] = _ajustar_y_puntuar(ctx, t.modelo, t.assigned, train.iloc[itr], train.iloc[iva])
        fold_interno[iva] = j
    umbral, ba_interna = best_threshold(ytr, oof)
    return {"oof_interno": oof, "fold_interno": fold_interno, "umbral": np.float64(umbral),
            "ba_interna": np.float64(ba_interna), "train_idx": np.asarray(t.train_idx, np.int64)}


# ---------------------------------------------------------------- cache

def clave_manifiesto(ctx: Contexto, t: Tarea) -> dict:
    spec = MODELS[t.modelo]
    return {
        "version_calculo": VERSION_CALCULO, "tipo": t.tipo, "modelo": t.modelo,
        "params": json.dumps(ctx.params[t.modelo], sort_keys=True), "assigned": t.assigned,
        "variante_preprocesamiento": spec.variante, "puntuacion": spec.score,
        "escenario": t.escenario, "fold": t.fold, "semilla": SEMILLA,
        "n_folds_externos": N_EXTERNOS, "n_folds_internos": N_INTERNOS if t.tipo == "interno" else None,
        "filas_entrenamiento": int(len(t.train_idx)), "hash_train_idx": hash_indices(t.train_idx),
        "filas_evaluacion": None if t.eval_idx is None else int(len(t.eval_idx)),
        "hash_eval_idx": None if t.eval_idx is None else hash_indices(t.eval_idx),
        "hash_train_csv": ctx.hash_train, "hash_codigo": ctx.codigo, "versiones": ctx.vers,
        "constructor": ctx.factory_nombre,
    }


def _rutas(ctx: Contexto, t: Tarea) -> tuple[Path, Path]:
    base = ctx.cache / t.clave
    return base.with_suffix(".npz"), base.with_suffix(".json")


def estado(ctx: Contexto, t: Tarea) -> str:
    """'valido', 'incompatible' o 'pendiente'. Un archivo existente no alcanza: se valida todo."""
    npz, js = _rutas(ctx, t)
    if not npz.exists() and not js.exists():
        return "pendiente"
    if not (npz.exists() and js.exists()):
        return "incompatible"
    try:
        man = json.loads(js.read_text())
        if man.get("clave") != clave_manifiesto(ctx, t):
            return "incompatible"
        z = np.load(npz)
        if t.tipo == "ajuste":
            ok = (len(z["scores"]) == len(t.eval_idx) and np.array_equal(z["eval_idx"], t.eval_idx)
                  and np.isfinite(z["scores"]).all())
        else:
            ok = (len(z["oof_interno"]) == len(t.train_idx) and np.array_equal(z["train_idx"], t.train_idx)
                  and np.isfinite(z["oof_interno"]).all() and np.isfinite(z["umbral"]))
        return "valido" if ok else "incompatible"
    except Exception:
        return "incompatible"


def descartar(ctx: Contexto, t: Tarea) -> None:
    """Mueve un cache incompatible a cache/descartados/ (no se borra)."""
    destino = ctx.cache / "descartados"
    destino.mkdir(parents=True, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d-%H%M%S")
    for p in _rutas(ctx, t):
        if p.exists():
            shutil.move(str(p), destino / f"{p.stem}.{marca}{p.suffix}")


def _escribir_atomico(path: Path, escribir) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        escribir(f)
    os.replace(tmp, path)


def cargar(ctx: Contexto, t: Tarea) -> dict:
    npz, _ = _rutas(ctx, t)
    with np.load(npz) as z:
        return {k: z[k] for k in z.files}


def ejecutar(ctx: Contexto, t: Tarea) -> dict:
    """Calcula y guarda una tarea. Devuelve un resumen para el log; nunca lanza excepciones."""
    t0 = time.time()
    try:
        with warnings.catch_warnings(record=True) as capturadas:
            warnings.simplefilter("always")
            arrays = calcular(ctx, t)
        if t.tipo == "ajuste" and not np.isfinite(arrays["scores"]).all():
            raise ValueError("puntuaciones no finitas")
        if t.tipo == "interno" and not np.isfinite(arrays["oof_interno"]).all():
            raise ValueError("OOF internos no finitos o incompletos")
        advertencias = Counter(f"{w.category.__name__}: {str(w.message)[:200]}" for w in capturadas)
        segundos = time.time() - t0
        npz, js = _rutas(ctx, t)
        ctx.cache.mkdir(parents=True, exist_ok=True)
        _escribir_atomico(npz, lambda f: np.savez(f, **arrays))
        manifiesto = {"clave": clave_manifiesto(ctx, t), "segundos": segundos,
                      "advertencias": dict(advertencias), "creado": datetime.now().isoformat(timespec="seconds")}
        # El json se escribe al final: su presencia marca la tarea como completa.
        _escribir_atomico(js, lambda f: f.write(json.dumps(manifiesto, indent=2).encode()))
        extra = {"umbral": float(arrays["umbral"]), "ba_interna": float(arrays["ba_interna"])} if t.tipo == "interno" else {}
        return {"clave": t.clave, "ok": True, "segundos": segundos, "advertencias": dict(advertencias), **extra}
    except Exception:
        return {"clave": t.clave, "ok": False, "segundos": time.time() - t0, "error": traceback.format_exc()}


# ---------------------------------------------------------------- estimacion y ejecucion

def segundos_por_ajuste(modelo: str, params: dict) -> tuple[float, str]:
    """Segundos de UN ajuste sobre ~30.600 filas, sacados de los experimentos (sin contar predecir)."""
    d = cfg.EXPERIMENTS / modelo
    for f in ("confirmacion.csv", "refinamiento.csv", "busqueda.csv"):
        p = d / f
        if p.exists():
            t = pd.read_csv(p)
            if len(t):
                igual = t[t["params"].map(lambda s: json.loads(s) == params)]
                if len(igual):
                    return float(igual["fit_seconds"].iloc[0]) / N_EXTERNOS, f
    if params == {} and (d / "base.json").exists():
        return json.loads((d / "base.json").read_text())["fit_seconds"] / N_EXTERNOS, "base.json"
    return json.loads((d / "metricas.json").read_text())["fit_seconds"] / N_EXTERNOS, "metricas.json (otra configuracion)"


def _segundos_tarea(t: Tarea, por_ajuste: float) -> float:
    escala = len(t.train_idx) / FILAS_REFERENCIA
    if t.tipo == "ajuste":
        return por_ajuste * escala
    return por_ajuste * escala * N_INTERNOS * (N_INTERNOS - 1) / N_INTERNOS


def _paralelo(modelo: str, jobs: int) -> int:
    return max(1, min(jobs, MODELS[modelo].fold_jobs))


def mostrar_plan(ctx: Contexto, tareas: list[Tarea], jobs: int) -> None:
    grupos: dict[tuple, list[Tarea]] = {}
    for t in tareas:
        grupos.setdefault((t.modelo, t.assigned, "cv" if t.escenario == "cv" else "robustez"), []).append(t)
    filas = []
    for (m, a, esc_tipo), grupo in grupos.items():
        est = {t.clave: estado(ctx, t) for t in grupo}
        estados = Counter(est.values())
        pend = [t for t in grupo if est[t.clave] != "valido"]
        por_ajuste, fuente = segundos_por_ajuste(m, ctx.params[m])
        seg = sum(_segundos_tarea(t, por_ajuste) for t in pend) / _paralelo(m, jobs)
        filas.append({"modelo": m, "variante": variante(a), "bloque": esc_tipo, "tareas": len(grupo),
                      "validas": estados["valido"], "incompatibles": estados["incompatible"],
                      "pendientes": estados["pendiente"], "ajustes_pendientes": sum(t.ajustes for t in pend),
                      "en_paralelo": _paralelo(m, jobs), "min_estimados": round(seg / 60, 1), "tiempo_de": fuente})
    tabla = pd.DataFrame(filas)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tabla.to_string(index=False))
    print(f"\nTotal: {len(tareas)} tareas, {int(tabla['ajustes_pendientes'].sum())} ajustes pendientes, "
          f"~{tabla['min_estimados'].sum():.0f} min estimados.")
    print("Estimacion gruesa: escala lineal con las filas y NO incluye el tiempo de predecir "
          "(en knn y random_forest puede ser varias veces el ajuste).")
    print(f"Caches en {ctx.cache}/ ; resultados en {ctx.resultados}/ (etapa resumen).")


def correr(ctx: Contexto, tareas: list[Tarea], jobs: int) -> list[dict]:
    errores = []
    for m in dict.fromkeys(t.modelo for t in tareas):
        grupo = [t for t in tareas if t.modelo == m]
        pendientes = []
        for t in grupo:
            e = estado(ctx, t)
            if e == "incompatible":
                log.warning("cache incompatible, se aparta a descartados/ y se recalcula: %s", t.clave)
                descartar(ctx, t)
            if e != "valido":
                pendientes.append(t)
        if not pendientes:
            log.info("[%s] nada pendiente (%d tareas ya validas)", m, len(grupo))
            continue
        n = _paralelo(m, jobs)
        log.info("[%s] %d tareas pendientes de %d, %d en paralelo", m, len(pendientes), len(grupo), n)
        t0 = time.time()
        if n == 1 or len(pendientes) == 1:
            resultados = (ejecutar(ctx, t) for t in pendientes)
        else:
            resultados = Parallel(n_jobs=n, return_as="generator_unordered")(delayed(ejecutar)(ctx, t) for t in pendientes)
        for i, r in enumerate(resultados, 1):
            if r["ok"]:
                extra = f"  umbral {r['umbral']:.4f}  BA interna {r['ba_interna']:.4f}" if "umbral" in r else ""
                log.info("  %d/%d %s  %.0fs%s", i, len(pendientes), r["clave"], r["segundos"], extra)
                for msg, c in r["advertencias"].items():
                    log.warning("    advertencia x%d: %s", c, msg)
            else:
                log.error("  %d/%d %s FALLO\n%s", i, len(pendientes), r["clave"], r["error"])
                errores.append(r)
        log.info("[%s] listo en %.1f min", m, (time.time() - t0) / 60)
    return errores


# ---------------------------------------------------------------- resumen

def _valida_o_none(ctx, t):
    return cargar(ctx, t) if estado(ctx, t) == "valido" else None


def _parametros_experimento(modelo: str):
    p = cfg.EXPERIMENTS / modelo / "mejores_parametros.json"
    return json.loads(p.read_text())["params"] if p.exists() else None


def resumir(ctx: Contexto, modelos: list[str]) -> None:
    ctx.resultados.mkdir(parents=True, exist_ok=True)
    y, df = ctx.y, ctx.df
    sin_locked = (df["reservation_commitment"] != "locked_allocation").to_numpy()
    folds_rows, resumen_rows, sub_rows, rob_rows, pred_ext, usados = [], [], [], [], [], []
    preds_internas = {}

    for m in modelos:
        for a in (True, False):
            aj = [_valida_o_none(ctx, Tarea("ajuste", m, a, "cv", k, tr, va)) for k, (tr, va) in enumerate(ctx.folds)]
            if any(x is None for x in aj):
                continue
            it = [_valida_o_none(ctx, Tarea("interno", m, a, "cv", k, tr, None)) for k, (tr, _) in enumerate(ctx.folds)]
            usados += [Tarea("ajuste", m, a, "cv", k, tr, va).clave for k, (tr, va) in enumerate(ctx.folds)]
            oof = np.full(len(df), np.nan)
            fold_de = np.full(len(df), -1)
            for k, ((_, va), x) in enumerate(zip(ctx.folds, aj)):
                oof[va] = x["scores"]
                fold_de[va] = k
            assert np.isfinite(oof).all() and (fold_de >= 0).all(), "los folds externos no cubren todas las filas"
            umbral_g, ba_conv = best_threshold(y, oof)
            pred_g = oof >= umbral_g
            completo = all(x is not None for x in it)
            pred_i = np.zeros(len(df), bool)
            umbral_fila = np.full(len(df), np.nan)
            for k, ((tr, va), x) in enumerate(zip(ctx.folds, it)):
                fila = {"modelo": m, "variante": variante(a), "fold": k, "filas_eval": len(va),
                        "eval_clase_0": int((y[va] == 0).sum()), "eval_clase_1": int((y[va] == 1).sum()),
                        "umbral_global_oof": umbral_g, "ba_fold_umbral_global": ba_segura(y[va], pred_g[va])}
                if x is not None:
                    usados.append(Tarea("interno", m, a, "cv", k, tr, None).clave)
                    u = float(x["umbral"])
                    pred_i[va] = oof[va] >= u
                    umbral_fila[va] = u
                    fila.update(umbral_interno=u, ba_interna_oof=float(x["ba_interna"]),
                                ba_fold_umbral_interno=ba_segura(y[va], pred_i[va]))
                folds_rows.append(fila)
            fila_r = {"modelo": m, "variante": variante(a), "params": json.dumps(ctx.params[m], sort_keys=True),
                      "ba_umbral_global_oof": ba_conv, "umbral_global_oof": umbral_g, "roc_auc": float(roc_auc_score(y, oof)),
                      "calibracion_completa": completo}
            if completo:
                por_fold = [ba_segura(y[va], pred_i[va]) for _, va in ctx.folds]
                umbrales = [float(x["umbral"]) for x in it]
                fila_r.update(ba_umbral_interno=ba_segura(y, pred_i), ba_umbral_interno_media_folds=float(np.mean(por_fold)),
                              ba_umbral_interno_std_folds=float(np.std(por_fold)),
                              caida_global_menos_interno=ba_conv - ba_segura(y, pred_i),
                              umbral_interno_min=min(umbrales), umbral_interno_max=max(umbrales))
                preds_internas[m, a] = pred_i
                for r in subgroup_table(df, pred_i.astype(float), 0.5).to_dict("records"):
                    sub_rows.append({"modelo": m, "variante": variante(a), "metodo": "umbral_interno", **r})
            for r in subgroup_table(df, oof, umbral_g).to_dict("records"):
                sub_rows.append({"modelo": m, "variante": variante(a), "metodo": "umbral_global_oof", **r})
            # Control de reproducibilidad: solo si es EXACTAMENTE la misma configuracion del experimento.
            if ctx.comparar_experimentos and _parametros_experimento(m) == ctx.params[m]:
                met = json.loads((cfg.EXPERIMENTS / m / "metricas.json").read_text())
                ref = met["ba_val"] if a else met["ba_val_sin_assigned"]
                fila_r.update(ba_experimento_misma_config=ref, diferencia_vs_experimento=ba_conv - ref)
            resumen_rows.append(fila_r)
            if completo:
                pred_ext.append(pd.DataFrame({
                    "modelo": m, "assigned": a, "variante": variante(a), "escenario": "cv",
                    "fold_externo": pd.array(fold_de, dtype="Int64"), "fila_train": np.arange(len(df)),
                    "y_true": y, "score": oof, "y_pred": pred_i.astype(int), "umbral": umbral_fila,
                    # Solo de control: el metodo convencional (un punto de corte optimizado sobre todos los OOF).
                    "umbral_global_oof": umbral_g, "y_pred_umbral_global": pd.array(pred_g.astype(int), dtype="Int64")}))

    ref_cv = {(r["modelo"], r["variante"]): r.get("ba_umbral_interno", np.nan) for r in resumen_rows}
    escenarios = escenarios_robustez(ctx)
    for m in modelos:
        for a in (True, False):
            por_nombre: dict[str, list] = {}
            for nombre, k, tr, va in escenarios:
                aj = _valida_o_none(ctx, Tarea("ajuste", m, a, nombre, k, tr, va))
                it = _valida_o_none(ctx, Tarea("interno", m, a, nombre, k, tr, None))
                por_nombre.setdefault(nombre, []).append((k, tr, va, aj, it))
            for nombre, partes in por_nombre.items():
                if any(aj is None or it is None for *_, aj, it in partes):
                    continue
                eval_idx = np.concatenate([va for _, _, va, _, _ in partes])
                pred = np.concatenate([aj["scores"] >= float(it["umbral"]) for *_, aj, it in partes])
                umbrales = [float(it["umbral"]) for *_, it in partes]
                for k, tr, va, aj, it in partes:
                    usados += [Tarea("ajuste", m, a, nombre, k, tr, va).clave, Tarea("interno", m, a, nombre, k, tr, None).clave]
                    pred_ext.append(pd.DataFrame({
                        "modelo": m, "assigned": a, "variante": variante(a), "escenario": nombre,
                        "fold_externo": pd.array([pd.NA if k is None else k] * len(va), dtype="Int64"),
                        "fila_train": va, "y_true": y[va], "score": aj["scores"],
                        "y_pred": (aj["scores"] >= float(it["umbral"])).astype(int), "umbral": float(it["umbral"]),
                        "umbral_global_oof": np.nan, "y_pred_umbral_global": pd.array([pd.NA] * len(va), dtype="Int64")}))
                ye = y[eval_idx]
                ref = ref_cv.get((m, variante(a)), np.nan)
                mismo = preds_internas.get((m, a))
                ba_mismo = ba_segura(ye, mismo[eval_idx]) if mismo is not None else np.nan
                ba = ba_segura(ye, pred)
                rob_rows.append({
                    "modelo": m, "variante": variante(a), "escenario": nombre,
                    # En sin_locked hay 5 entrenamientos: se informa el promedio de filas.
                    "filas_entrenamiento": int(round(np.mean([len(tr) for _, tr, *_ in partes]))),
                    "filas_evaluacion": len(eval_idx), "eval_clase_0": int((ye == 0).sum()), "eval_clase_1": int((ye == 1).sum()),
                    "umbral_min": min(umbrales), "umbral_max": max(umbrales), "ba": ba,
                    "cancelados_detectados": float(pred[ye == 1].mean()) if (ye == 1).any() else np.nan,
                    "completados_detectados": float((~pred[ye == 0]).mean()) if (ye == 0).any() else np.nan,
                    "ba_cv_calibrada": ref, "caida_vs_cv": ref - ba,
                    # Las mismas filas evaluadas con el modelo de la CV normal (entrenado con todo lo demas).
                    "ba_mismas_filas_en_cv": ba_mismo, "caida_vs_mismas_filas": ba_mismo - ba,
                })
                if nombre == "sin_locked":
                    rob_rows[-1]["nota"] = "entrenado y evaluado sin locked_allocation; ba_mismas_filas_en_cv = modelo entrenado con todas las filas"

    out = ctx.resultados
    pd.DataFrame(folds_rows).to_csv(out / "calibracion_folds.csv", index=False)
    pd.DataFrame(resumen_rows).to_csv(out / "calibracion_resumen.csv", index=False)
    pd.DataFrame(sub_rows).to_csv(out / "subgrupos_calibrados.csv", index=False)
    pd.DataFrame(rob_rows).to_csv(out / "robustez_calibrada.csv", index=False)
    if pred_ext:
        p = pd.concat(pred_ext, ignore_index=True)
        p.to_csv(out / "predicciones_externas.csv", index=False)
        # Verificaciones de cobertura, sobre el archivo tal como quedo escrito.
        p = pd.read_csv(out / "predicciones_externas.csv", dtype={"fold_externo": "Int64", "y_pred_umbral_global": "Int64"})
        esperado = {"cv": np.arange(len(df)), "sin_locked": np.flatnonzero(sin_locked)}
        esperado.update({n: va for n, k, tr, va in escenarios if k is None})
        for (m, a, esc), g in p.groupby(["modelo", "assigned", "escenario"]):
            filas = np.sort(g["fila_train"].to_numpy())
            assert np.array_equal(filas, np.sort(esperado[esc])), f"{m} {a} {esc}: faltan o sobran observaciones"
            assert np.array_equal(g["y_true"].to_numpy(), y[g["fila_train"].to_numpy()]), f"{m} {a} {esc}: y_true no coincide"
        matriz, globales, _ = metricas_externas.escribir(p, out)
        # La BA de la matriz (solo predicciones guardadas) coincide con la de los CSV anteriores.
        ba = {(r["modelo"], bool(r["assigned"]), r["escenario"]): r["balanced_accuracy"] for r in globales.to_dict("records")}
        for r in resumen_rows:
            if r["calibracion_completa"]:
                assert np.isclose(ba[r["modelo"], r["variante"] == "con_assigned", "cv"], r["ba_umbral_interno"]), "BA no coincide"
        for r in rob_rows:
            assert np.isclose(ba[r["modelo"], r["variante"] == "con_assigned", r["escenario"]], r["ba"], equal_nan=True)

    manifiesto = {"creado": datetime.now().isoformat(timespec="seconds"), "hash_train_csv": ctx.hash_train,
                  "hash_codigo": ctx.codigo, "versiones": ctx.vers, "semilla": SEMILLA,
                  "n_folds_externos": N_EXTERNOS, "n_folds_internos": N_INTERNOS,
                  "params": {m: ctx.params[m] for m in modelos}, "tareas_usadas": sorted(set(usados))}
    (out / "manifiesto_resultados.json").write_text(json.dumps(manifiesto, indent=2))
    log.info("resumen: %d configuraciones de CV, %d calibradas, %d escenarios de robustez -> %s",
             len(resumen_rows), sum(r["calibracion_completa"] for r in resumen_rows), len(rob_rows), out)


# ---------------------------------------------------------------- CLI

def _verificar_train(path: Path) -> str:
    h = hash_archivo(path)
    meta = cfg.DATA_RAW / "split_metadata.json"
    if meta.exists():
        esperado = json.loads(meta.read_text()).get("sha256_archivos", {}).get("train.csv")
        if esperado and esperado != h:
            raise SystemExit(f"train.csv no coincide con el hash de split_metadata.json ({h} != {esperado})")
    return h


def _configurar_log(raiz: Path) -> None:
    raiz.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(raiz / "calibracion.log")):
        h.setFormatter(fmt)
        log.addHandler(h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["oof", "calibracion", "robustez", "resumen", "todo"])
    ap.add_argument("--modelos", nargs="+", help="por defecto: los 8 (oof, calibracion, resumen) o los 3 finalistas (robustez)")
    ap.add_argument("--assigned", choices=["auto", "con", "sin", "ambos"], default="auto",
                    help="auto: con assigned para todos y ademas sin assigned para los finalistas; robustez siempre ambos")
    ap.add_argument("--jobs", type=int, default=4, help="tareas en paralelo como maximo (los modelos que ya usan todos los nucleos van de a 1)")
    ap.add_argument("--dry-run", action="store_true", help="lista tareas, caches y tiempo estimado sin entrenar")
    ap.add_argument("--salida", type=Path, default=SALIDA)
    a = ap.parse_args(argv)

    modelos = a.modelos or (FINALISTAS if a.etapa == "robustez" else MODELOS)
    params = json.loads(PARAMETROS.read_text())
    faltan = [m for m in modelos if m not in params or m not in MODELS]
    if faltan:
        raise SystemExit(f"sin configuracion fija para: {faltan}")

    _configurar_log(a.salida)
    df = load_train()
    ctx = Contexto(df=df, params=params, raiz=a.salida, hash_train=_verificar_train(cfg.DATA_RAW / "train.csv"))
    log.info("etapa=%s modelos=%s assigned=%s jobs=%d codigo=%s", a.etapa, modelos, a.assigned, a.jobs, ctx.codigo)

    if a.etapa == "resumen":
        resumir(ctx, modelos)
        return 0
    tareas = planificar(ctx, a.etapa, modelos, a.assigned)
    if a.dry_run:
        mostrar_plan(ctx, tareas, a.jobs)
        return 0
    errores = correr(ctx, tareas, a.jobs)
    if a.etapa == "todo":
        resumir(ctx, modelos)
    if errores:
        log.error("%d tareas fallaron; relanzar el mismo comando reintenta solo esas", len(errores))
        return 1
    log.info("COMPLETO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
