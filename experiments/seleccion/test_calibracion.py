"""Pruebas cortas de calibracion.py sobre una submuestra de train (no usa test.csv).

Correr:  PYTHONPATH=src .venv/bin/python experiments/seleccion/test_calibracion.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibracion as C  # noqa: E402
import metricas_externas as M  # noqa: E402
from compe.data import feature_groups, load_train  # noqa: E402
from compe.features import Preprocessor  # noqa: E402

PARAMS = json.loads(C.PARAMETROS.read_text())
DF = load_train().sample(3000, random_state=0).reset_index(drop=True)


def ctx_en(tmp, df=DF, **kw):
    return C.Contexto(df=df, params=PARAMS, raiz=Path(tmp), hash_train="prueba", comparar_experimentos=False, **kw)


class Espia(BaseEstimator, ClassifierMixin):
    """Registra que filas (columna _id) ve al ajustar y al puntuar."""
    registro: list = []

    def fit(self, X, y):
        Espia.registro.append(("fit", X["_id"].to_numpy().copy()))
        self.classes_ = np.array([0, 1])
        return self

    def _r(self, X):
        Espia.registro.append(("score", X["_id"].to_numpy().copy()))
        return (X["_id"].to_numpy() * 0.618) % 1

    def predict_proba(self, X):
        r = self._r(X)
        return np.c_[1 - r, r]

    def decision_function(self, X):
        return self._r(X)


def test_cobertura_y_grupos():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = ctx_en(tmp)
        g = feature_groups(ctx.df).to_numpy()
        visto = np.zeros(len(ctx.df), int)
        for tr, va in ctx.folds:
            visto[va] += 1
            assert not set(tr) & set(va)
            assert not set(g[tr]) & set(g[va]), "grupo partido entre entrenamiento y validacion externa"
            interno = C.folds_internos(ctx.df.iloc[tr])
            cubre = np.zeros(len(tr), int)
            for itr, iva in interno:
                assert itr.max() < len(tr) and iva.max() < len(tr), "indices internos no son relativos"
                cubre[iva] += 1
                gtr, gva = g[tr[itr]], g[tr[iva]]
                assert not set(gtr) & set(gva), "grupo partido en folds internos"
                assert not set(tr[iva]) & set(va), "fila externa en folds internos"
            assert (cubre == 1).all()
        assert (visto == 1).all(), "cada fila tiene que evaluarse exactamente una vez"
        for nombre, k, tr, va in C.escenarios_robustez(ctx):
            assert not set(tr) & set(va), nombre
            assert not set(g[tr]) & set(g[va]), f"{nombre}: grupo partido"
        locked = (ctx.df["reservation_commitment"] == "locked_allocation").to_numpy()
        sl = [(tr, va) for n, k, tr, va in C.escenarios_robustez(ctx) if n == "sin_locked"]
        assert not locked[np.concatenate([np.r_[tr, va] for tr, va in sl])].any()
        assert np.sort(np.concatenate([va for _, va in sl])).tolist() == np.flatnonzero(~locked).tolist()


def test_espia_no_ve_validacion_externa():
    df = DF.assign(_id=np.arange(len(DF)))
    with tempfile.TemporaryDirectory() as tmp:
        ctx = ctx_en(tmp, df=df, factory=lambda m, a: Espia(), factory_nombre="espia")
        for k, (tr, va) in enumerate(ctx.folds):
            Espia.registro.clear()
            C.calcular(ctx, C.Tarea("interno", "logistic", True, "cv", k, tr, None))
            for tipo, ids in Espia.registro:
                assert not set(ids) & set(va), f"el fold externo aparecio en un {tipo} interno"
                assert set(ids) <= set(tr)
            Espia.registro.clear()
            C.calcular(ctx, C.Tarea("ajuste", "logistic", True, "cv", k, tr, va))
            (t1, ids_fit), (t2, ids_score) = Espia.registro
            assert t1 == "fit" and sorted(ids_fit) == sorted(tr)
            assert t2 == "score" and sorted(ids_score) == sorted(va)


def test_umbral_y_preprocesamiento_no_dependen_de_validacion():
    """Cambiar etiquetas y variables del fold externo no cambia el umbral ni las otras puntuaciones."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = ctx_en(tmp)
        tr, va = ctx.folds[0]
        t_int = C.Tarea("interno", "ridge", True, "cv", 0, tr, None)
        t_aj = C.Tarea("ajuste", "ridge", True, "cv", 0, tr, va)
        u1 = C.calcular(ctx, t_int)["umbral"]
        s1 = C.calcular(ctx, t_aj)["scores"]
        df2 = ctx.df.copy()
        mitad = va[: len(va) // 2]
        df2.loc[va, "canceled_job"] = 1 - df2.loc[va, "canceled_job"]
        df2.loc[mitad, "hours_before_execution"] *= 50
        df2.loc[mitad, "client_region"] = "ZZZ"
        ctx2 = ctx_en(tmp, df=df2)
        assert C.calcular(ctx2, t_int)["umbral"] == u1
        s2 = C.calcular(ctx2, t_aj)["scores"]
        otra = len(mitad)
        assert np.array_equal(s1[otra:], s2[otra:]), "el preprocesamiento aprendio de la validacion"


def test_sin_assigned_excluye_variable():
    tr, va = np.arange(0, 2400), np.arange(2400, 3000)
    rng = np.random.default_rng(0)
    barajado = DF.copy()
    barajado["assigned_gpu_type"] = rng.permutation(barajado["assigned_gpu_type"].to_numpy())
    with tempfile.TemporaryDirectory() as tmp:
        for m in ["logistic", "gradient_boosting", "hist_gradient_boosting"]:
            t = C.Tarea("ajuste", m, False, "cv", 0, tr, va)
            s1 = C.calcular(ctx_en(tmp), t)["scores"]
            s2 = C.calcular(ctx_en(tmp, df=barajado), t)["scores"]
            assert np.allclose(s1, s2, rtol=0, atol=1e-12), f"{m}: sin assigned depende de assigned_gpu_type"
        t = C.Tarea("ajuste", "hist_gradient_boosting", True, "cv", 0, tr, va)
        assert not np.allclose(C.calcular(ctx_en(tmp), t)["scores"], C.calcular(ctx_en(tmp, df=barajado), t)["scores"]), \
            "control: con assigned la variable deberia importar"
    cols = Preprocessor(use_assigned=False).fit(DF).transform(DF).columns
    assert "assigned_gpu_type" not in cols and "gpu_changed" not in cols
    cols = Preprocessor(use_assigned=True).fit(DF).transform(DF).columns
    assert "assigned_gpu_type" in cols and "gpu_changed" in cols


def test_ejecucion_resumen_y_ba():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = ctx_en(tmp)
        tareas = C.planificar(ctx, "calibracion", ["ridge"], "ambos") + C.planificar(ctx, "robustez", ["ridge"], "con")
        assert not C.correr(ctx, tareas, jobs=1)
        assert all(C.estado(ctx, t) == "valido" for t in tareas)
        C.resumir(ctx, ["ridge"])
        out = ctx.resultados
        for f in ["calibracion_folds.csv", "calibracion_resumen.csv", "subgrupos_calibrados.csv", "robustez_calibrada.csv",
                  "predicciones_externas.csv", "matriz_confusion_global.csv", "metricas_externas_globales.csv",
                  "metricas_externas_por_fold.csv", "manifiesto_resultados.json"]:
            assert (out / f).exists(), f
        res = pd.read_csv(out / "calibracion_resumen.csv")
        assert len(res) == 2 and res["calibracion_completa"].all()
        p = pd.read_csv(out / "predicciones_externas.csv", dtype={"fold_externo": "Int64"})
        assert set(M.OBLIGATORIAS) <= set(p.columns)
        cv = p[p.escenario == "cv"]
        for _, r in res.iterrows():
            s = cv[(cv.modelo == r.modelo) & (cv.variante == r.variante)].sort_values("fila_train")
            assert s["fila_train"].tolist() == list(range(len(DF))), "CV normal: cada fila exactamente una vez"
            assert np.isfinite(s["score"]).all() and set(s["y_pred"]) <= {0, 1}
            assert (s["y_true"].to_numpy() == DF["canceled_job"].to_numpy()).all()
            assert s["fold_externo"].notna().all() and s.groupby("fold_externo")["umbral"].nunique().eq(1).all()
            assert np.isclose(C.ba_segura(s["y_true"], s["y_pred"]), r.ba_umbral_interno)
            assert np.isclose(C.ba_segura(s["y_true"], s["y_pred_umbral_global"]), r.ba_umbral_global_oof)
        folds = pd.read_csv(out / "calibracion_folds.csv")
        assert len(folds) == 10 and folds["umbral_interno"].notna().all()
        rob = pd.read_csv(out / "robustez_calibrada.csv")
        assert set(rob.escenario) == {"eval_anio_2015", "eval_anio_2016", "eval_anio_2017", "eval_PRT", "eval_resto", "sin_locked"}
        locked = (DF["reservation_commitment"] == "locked_allocation").to_numpy()
        for _, r in rob.iterrows():
            s = p[(p.escenario == r.escenario) & (p.variante == r.variante) & (p.modelo == r.modelo)]
            assert len(s) == r.filas_evaluacion and s["fila_train"].is_unique
            assert ((s["score"] >= s["umbral"]).astype(int) == s["y_pred"]).all()
            assert np.isclose(C.ba_segura(s["y_true"], s["y_pred"]), r.ba, equal_nan=True)
        sl = p[p.escenario == "sin_locked"]
        assert sorted(sl["fila_train"]) == np.flatnonzero(~locked).tolist(), "sin_locked: cada fila del subconjunto una vez"
        mat = pd.read_csv(out / "matriz_confusion_global.csv")
        assert (mat[["TN", "FP", "FN", "TP"]].sum(axis=1) == mat["n"]).all()
        conteo = p.groupby(["modelo", "assigned", "escenario"]).size().rename("filas").reset_index()
        assert (mat.merge(conteo, on=["modelo", "assigned", "escenario"]).eval("n == filas")).all()
        # Regenerar solo desde el archivo de predicciones da exactamente lo mismo.
        with tempfile.TemporaryDirectory() as otra:
            M.main([str(out / "predicciones_externas.csv"), otra])
            for f in ["matriz_confusion_global.csv", "metricas_externas_globales.csv", "metricas_externas_por_fold.csv"]:
                pd.testing.assert_frame_equal(pd.read_csv(out / f), pd.read_csv(Path(otra) / f))


def test_metricas_a_mano():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([1, 1, 0, 0, 0, 1, 0, 1])
    m = M.metricas(y, p, p.astype(float))
    assert (m["TN"], m["FP"], m["FN"], m["TP"]) == (3, 1, 1, 3)
    assert m["sensibilidad"] == 0.75 and m["especificidad"] == 0.75 and m["balanced_accuracy"] == 0.75
    assert m["accuracy"] == 0.75 and m["precision"] == 0.75 and m["f1"] == 0.75
    assert np.isclose(m["balanced_accuracy"], C.ba_segura(y, p))
    # Denominadores en cero -> NA
    m0 = M.metricas(np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert np.isnan(m0["precision"]) and m0["sensibilidad"] == 0 and m0["f1"] == 0
    m1 = M.metricas(np.array([1, 1]), np.array([1, 0]), np.array([0.9, 0.1]))
    assert np.isnan(m1["especificidad"]) and np.isnan(m1["balanced_accuracy"]) and np.isnan(m1["roc_auc"])
    # Global concatenado != promedio de folds; ddof=0
    pred = pd.DataFrame({"modelo": "x", "assigned": True, "escenario": "cv", "fold_externo": [0, 0, 0, 0, 1, 1],
                         "fila_train": range(6), "y_true": [1, 0, 1, 0, 1, 0], "score": [.9, .1, .2, .3, .8, .7],
                         "y_pred": [1, 0, 0, 0, 1, 1], "umbral": [.5, .5, .5, .5, .6, .6]})
    _, g, f = M.calcular(pred)
    assert np.isclose(g["balanced_accuracy"].iloc[0], C.ba_segura(pred.y_true, pred.y_pred))
    assert np.isclose(g["balanced_accuracy_std_folds_ddof0"].iloc[0], np.std(f["balanced_accuracy"], ddof=0))
    for malo in [pred.assign(y_pred=[0, 0, 0, 0, 1, 1]), pred.assign(fila_train=[0, 0, 1, 2, 3, 4])]:
        try:
            M.calcular(malo)
            raise AssertionError("tenia que rechazar predicciones inconsistentes")
        except ValueError:
            pass


def test_cache_rechaza_incompatibles():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = ctx_en(tmp)
        tr, va = ctx.folds[0]
        t = C.Tarea("ajuste", "ridge", True, "cv", 0, tr, va)
        assert C.estado(ctx, t) == "pendiente"
        assert not C.correr(ctx, [t], jobs=1)
        assert C.estado(ctx, t) == "valido"
        assert C.estado(ctx, C.Tarea("ajuste", "ridge", False, "cv", 0, tr, va)) == "pendiente"
        otro = dict(PARAMS, ridge={**PARAMS["ridge"], "alpha": 1.0})
        assert C.estado(C.Contexto(df=DF, params=otro, raiz=Path(tmp), hash_train="prueba"), t) == "incompatible"
        assert C.estado(C.Contexto(df=DF, params=PARAMS, raiz=Path(tmp), hash_train="otro"), t) == "incompatible"
        t_mal = C.Tarea("ajuste", "ridge", True, "cv", 0, tr, va[::-1])   # mismo nombre, otros indices
        assert C.estado(ctx, t_mal) == "incompatible"
        npz, js = C._rutas(ctx, t)
        js.unlink()
        assert C.estado(ctx, t) == "incompatible", "npz sin manifiesto no se reutiliza"
        assert not C.correr(ctx, [t], jobs=1)
        assert C.estado(ctx, t) == "valido"
        assert list((ctx.cache / "descartados").glob("*.npz")), "el cache viejo se aparta, no se borra"


def test_paralelo_igual_a_secuencial():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        ca, cb = ctx_en(a), ctx_en(b)
        ta = C.planificar(ca, "oof", ["ridge"], "con")
        assert not C.correr(ca, ta, jobs=1)
        assert not C.correr(cb, C.planificar(cb, "oof", ["ridge"], "con"), jobs=3)
        for t in ta:
            assert np.array_equal(C.cargar(ca, t)["scores"], C.cargar(cb, t)["scores"])


def test_ba_segura():
    assert np.isnan(C.ba_segura(np.ones(5), np.ones(5)))
    assert C.ba_segura(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])) == 0.75


if __name__ == "__main__":
    fallas = 0
    for nombre, f in [(n, f) for n, f in globals().items() if n.startswith("test_")]:
        t0 = time.time()
        try:
            f()
            print(f"ok    {nombre}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            fallas += 1
            import traceback
            print(f"FALLA {nombre}: {e!r}")
            traceback.print_exc()
    print("todo ok" if not fallas else f"{fallas} pruebas fallaron")
    sys.exit(1 if fallas else 0)
