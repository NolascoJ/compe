"""Three fixed strategies; strict nested threshold calibration. Never evaluates test.csv."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import calibracion as C
import metricas_externas as M
from compe.data import feature_groups, load_train
from compe.validation import best_threshold, make_folds, model_scores

HERE = Path(__file__).resolve().parent
BASES = ('gradient_boosting', 'hist_gradient_boosting', 'random_forest')
STRATEGIES = ('histgb', 'promedio', 'stacking')
META = dict(C=1.0, solver='lbfgs', max_iter=1000, class_weight=None, random_state=42)
# Default penalty is L2. Avoid the penalty argument deprecated in sklearn 1.9.

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        writer(f)
    os.replace(tmp, path)


class Experiment:
    def __init__(self, df, params, root, train_hash, legacy=None, factory=None):
        self.df = df.reset_index(drop=True)
        self.y = self.df[C.cfg.TARGET].to_numpy()
        self.groups = feature_groups(self.df).to_numpy()
        self.params = {m: params[m] for m in BASES}
        self.root = Path(root)
        self.factory = factory
        self.ctx = C.Contexto(self.df, self.params, Path(legacy or HERE/'calibracion'), train_hash)
        self.identity = dict(train_hash=train_hash, params=self.params, meta=META,
                             assigned=False, seed=42, external=5, internal=3, deepest=3,
                             code=C.hash_codigo(), versions=C.versiones(),
                             implementation=C.hash_archivo(Path(__file__)),
                             metrics_code=C.hash_archivo(Path(M.__file__)),
                             calibration_code=C.hash_archivo(Path(C.__file__)),
                             factory='production' if factory is None else 'test')
        self.run_id = digest(self.identity)[:16]
        self.run = self.root / self.run_id
        self.legacy_index = {}
        if factory is None:
            for js in self.ctx.cache.glob('*.json'):
                try:
                    key = json.loads(js.read_text())['clave']
                    if key['modelo'] in BASES and key['assigned'] is False:
                        self.legacy_index[(key['tipo'], key['modelo'], key['hash_train_idx'], key['hash_eval_idx'])] = key
                except (ValueError, KeyError):
                    continue

    def check_split(self, tr, va):
        assert not np.intersect1d(tr, va).size, 'row overlap'
        assert not np.intersect1d(self.groups[tr], self.groups[va]).size, 'group overlap'
        assert len(np.unique(tr)) == len(tr) and len(np.unique(va)) == len(va)

    def splits(self, idx, n=3):
        parts = make_folds(self.df.iloc[idx].reset_index(drop=True), n_splits=n, seed=42)
        coverage = np.zeros(len(idx), int)
        for tr, va in parts:
            self.check_split(idx[tr], idx[va]); coverage[va] += 1
        assert np.all(coverage == 1)
        return parts

    def key(self, kind, model, tr, va=None):
        return dict(identity=self.run_id, kind=kind, model=model,
                    train=C.hash_indices(tr), evaluate=None if va is None else C.hash_indices(va))

    def paths(self, key):
        stem = self.run/'cache'/f"{key['kind']}_{key['model']}_{digest(key)[:24]}"
        return stem.with_suffix('.npz'), stem.with_suffix('.json')

    def cached(self, key):
        npz, js = self.paths(key)
        if not npz.exists() or not js.exists(): return None
        info = json.loads(js.read_text())
        if info['key'] != key or info['sha256_npz'] != C.hash_archivo(npz):
            raise ValueError(f'Cache corrupto/incompatible: {npz}; conservar y revisar antes de seguir')
        with np.load(npz) as z:
            return {k:z[k] for k in z.files}

    def store(self, key, arrays, source, seconds=0., notices=None):
        for arr in arrays.values():
            if np.issubdtype(np.asarray(arr).dtype, np.number):
                assert np.isfinite(arr).all(), f'nonfinite cache: {key}'
        npz, js = self.paths(key)
        atomic(npz, lambda f:np.savez_compressed(f, **arrays))
        info = dict(key=key, sha256_npz=C.hash_archivo(npz), source=source,
                    seconds=seconds, warnings=notices or {})
        atomic(js, lambda f:f.write(json.dumps(info, indent=2).encode()))
        return arrays

    def legacy(self, kind, model, tr, va=None):
        key = self.legacy_index.get((kind, model, C.hash_indices(tr), None if va is None else C.hash_indices(va)))
        if key is None: return None
        task=C.Tarea(kind, model, False, key['escenario'], key['fold'], tr, va)
        if C.estado(self.ctx, task) != 'valido': return None
        result=C.cargar(self.ctx, task)
        if kind=='interno':
            fid=np.full(len(tr), -1)
            for j, (_, iv) in enumerate(self.splits(tr)): fid[iv]=j
            if not np.array_equal(fid,result['fold_interno']):
                raise ValueError('Legacy internal folds are not the requested grouped folds')
        return result

    def base(self, model, tr, va):
        self.check_split(tr, va)
        key=self.key('base', model, tr, va)
        z=self.cached(key)
        if z is not None:
            assert np.array_equal(z['eval_idx'],va) and len(z['score'])==len(va)
            return z['score']
        old=self.legacy('ajuste', model, tr, va)
        if old is not None:
            return self.store(key,dict(score=old['scores'],eval_idx=va),'legacy ajuste')['score']
        t=time.time()
        with warnings.catch_warnings(record=True) as ws, threadpool_limits(limits=1):
            warnings.simplefilter('always')
            est=self.factory(model) if self.factory else self.ctx.crear(model,False)
            # Outer jobs own parallelism; changing n_jobs does not change model hyperparameters.
            if 'model__n_jobs' in est.get_params(): est.set_params(model__n_jobs=1)
            est.fit(self.df.iloc[tr],self.y[tr])
            score=np.asarray(model_scores(est,self.df.iloc[va],'proba'))
        notices=dict(Counter(f'{w.category.__name__}: {w.message}' for w in ws))
        if notices: print('WARN',model,notices,flush=True)
        assert score.shape==(len(va),) and np.isfinite(score).all()
        return self.store(key,dict(score=score,eval_idx=va),'fit',time.time()-t,notices)['score']

    def oof(self, model, tr):
        key=self.key('oof',model,tr)
        z=self.cached(key)
        if z is not None:
            assert np.array_equal(z['train_idx'],tr)
            return z['score']
        parts=self.splits(tr)
        old=self.legacy('interno',model,tr)
        if old is not None:
            score=old['oof_interno']
            # These blocks are valid predictions of models fitted on each inner training.
            for itr,iva in parts:
                self.store(self.key('base',model,tr[itr],tr[iva]),
                           dict(score=score[iva],eval_idx=tr[iva]),'legacy interno block')
        else:
            score=np.full(len(tr),np.nan)
            for itr,iva in parts: score[iva]=self.base(model,tr[itr],tr[iva])
        assert np.isfinite(score).all()
        return self.store(key,dict(score=score,train_idx=tr),'legacy interno' if old is not None else 'assembled')['score']

    def fit_meta(self, Z, y):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            meta=LogisticRegression(**META).fit(Z,y)
        if ws:
            notices=[f'{w.category.__name__}: {w.message}' for w in ws]
            # No hiding a convergence problem; checkpoints of base fits remain reusable.
            raise RuntimeError('Metamodel warnings; inspect before continuing: '+str(notices))
        return meta

    def compute(self, tr, va):
        """Calibrate all three strategies without ever fitting on va (including its labels)."""
        self.check_split(tr,va)
        Z=np.column_stack([self.oof(m,tr) for m in BASES])
        # True OOF of the ENTIRE stacking procedure, not CV on precomputed Z alone.
        stack_oof=np.full(len(tr),np.nan)
        for itr,iva in self.splits(tr):
            train_inner=tr[itr]
            Z_deep=np.column_stack([self.oof(m,train_inner) for m in BASES])
            meta=self.fit_meta(Z_deep,self.y[train_inner])
            # Z[iva] was generated by bases fitted on precisely train_inner.
            stack_oof[iva]=meta.predict_proba(Z[iva])[:,1]
        internal=np.column_stack([Z[:,1],Z.mean(axis=1),stack_oof])
        thresholds=np.array([best_threshold(self.y[tr],internal[:,j])[0] for j in range(3)])
        final_meta=self.fit_meta(Z,self.y[tr])
        external_base=np.column_stack([self.base(m,tr,va) for m in BASES])
        external=np.column_stack([external_base[:,1],external_base.mean(axis=1),
                                  final_meta.predict_proba(external_base)[:,1]])
        return dict(train_idx=tr,eval_idx=va,score=external,threshold=thresholds,
                    inner_score=internal,base_external=external_base,
                    meta_coef=final_meta.coef_,meta_intercept=final_meta.intercept_)

    def task(self, scenario, fold, tr, va):
        key=self.key('result','three',tr,va)
        result=self.cached(key)
        if result is None:
            print('START',scenario,fold,flush=True);t=time.time()
            result=self.compute(tr,va)
            self.store(key,result,'strict nested stacking',time.time()-t)
            print('DONE',scenario,fold,f'{time.time()-t:.1f}s',flush=True)
        else: print('CACHED',scenario,fold,flush=True)
        return scenario,fold

    def tasks(self, phase):
        cv=[('cv',k,tr,va) for k,(tr,va) in enumerate(self.ctx.folds)]
        rob=C.escenarios_robustez(self.ctx)
        return cv if phase=='cv' else rob if phase=='robustez' else cv+rob

    def initialize(self):
        self.run.mkdir(parents=True,exist_ok=True)
        path=self.run/'manifiesto.json'
        if path.exists(): assert json.loads(path.read_text())==self.identity
        else: atomic(path,lambda f:f.write(json.dumps(self.identity,indent=2).encode()))


def summarize(exp):
    rows=[];coeff=[];status=[]
    for scenario,fold,tr,va in exp.tasks('todo'):
        z=exp.cached(exp.key('result','three',tr,va));status.append(dict(escenario=scenario,fold=fold,completo=z is not None))
        if z is None: continue
        assert np.array_equal(z['eval_idx'],va)
        assert z['score'].shape == (len(va),3) and z['threshold'].shape == (3,)
        coeff.append(dict(escenario=scenario,fold=fold,intercept=float(z['meta_intercept'][0]),
                          **{n:float(v) for n,v in zip(BASES,z['meta_coef'][0])}))
        for j,name in enumerate(STRATEGIES):
            score=z['score'][:,j];thr=float(z['threshold'][j])
            rows.append(pd.DataFrame(dict(modelo=name,assigned=False,escenario=scenario,
                fold_externo=fold,fila_train=va,y_true=exp.y[va],score=score,
                y_pred=(score>=thr).astype(int),umbral=thr)))
    out=exp.run/'resultados';out.mkdir(exist_ok=True)
    pd.DataFrame(status).to_csv(out/'estado.csv',index=False)
    if not rows: print('Sin resultados completos por fold; no se entrenó nada.');return
    pred=pd.concat(rows,ignore_index=True)
    for (name, scenario), d in pred.groupby(['modelo','escenario']):
        expected=np.concatenate([va for sc,fo,tr,va in exp.tasks('todo')
                                 if sc==scenario and exp.cached(exp.key('result','three',tr,va)) is not None])
        assert d.fila_train.is_unique and np.array_equal(np.sort(d.fila_train),np.sort(expected))
    pred.to_csv(out/'predicciones_externas.csv',index=False)
    _,globales,perfold=M.escribir(pred,out)
    # Mark partial aggregates visibly: they must never be mistaken for five-fold estimates.
    complete={s:all(r['completo'] for r in status if r['escenario']==s) for s in {r['escenario'] for r in status}}
    globales['escenario_completo']=globales.escenario.map(complete)
    globales.to_csv(out/'metricas_externas_globales.csv',index=False)
    pd.DataFrame(coeff).to_csv(out/'coeficientes_meta.csv',index=False)
    # Store the provenance of the estimator predictions; never tune cutoffs here.
    sub=[]
    for (name,scenario),d in pred.groupby(['modelo','escenario']):
        ix=d.fila_train.to_numpy();dd=exp.df.iloc[ix]
        masks={'todo':np.ones(len(d),bool),'sin_locked':(dd.reservation_commitment!='locked_allocation').to_numpy(),
               'PRT':(dd.client_region=='PRT').to_numpy(),'resto':(dd.client_region!='PRT').to_numpy()}
        masks.update({f'anio_{y}':(dd.execution_year==y).to_numpy() for y in sorted(dd.execution_year.unique())})
        for label,mask in masks.items():
            if not mask.any(): continue
            v=d.loc[mask];sub.append(dict(modelo=name,escenario=scenario,subgrupo=label,
                escenario_completo=complete[scenario],**M.metricas(v.y_true,v.y_pred,v.score)))
    pd.DataFrame(sub).to_csv(out/'subgrupos.csv',index=False)
    # Paired robust-scenario and fold comparisons, with same-row CV reference.
    comparisons=[]
    for (name,scenario),d in pred.groupby(['modelo','escenario']):
        ref=pred[(pred.modelo=='histgb')&(pred.escenario==scenario)].sort_values('fila_train')
        d=d.sort_values('fila_train');assert np.array_equal(d.fila_train,ref.fila_train)
        val=M.metricas(d.y_true,d.y_pred,d.score)['balanced_accuracy']
        base=M.metricas(ref.y_true,ref.y_pred,ref.score)['balanced_accuracy']
        same=pred[(pred.modelo==name)&(pred.escenario=='cv')&pred.fila_train.isin(d.fila_train)]
        ba_same=M.metricas(same.y_true,same.y_pred,same.score)['balanced_accuracy'] if len(same)==len(d) else np.nan
        comparisons.append(dict(modelo=name,escenario=scenario,completo=complete[scenario],ba=val,
                                diferencia_histgb=val-base,ba_mismas_filas_cv=ba_same,caida_vs_mismas_filas=ba_same-val))
    pd.DataFrame(comparisons).to_csv(out/'comparacion_escenarios.csv',index=False)
    if complete.get('cv',False): paired_summary(exp,pred,perfold,out)
    print('Resultados:',out,flush=True)


def paired_summary(exp,pred,perfold,out):
    cv=pred[pred.escenario=='cv'];ref=cv[cv.modelo=='histgb'].sort_values('fila_train');y=ref.y_true.to_numpy();p0=ref.y_pred.to_numpy()
    assert np.array_equal(ref.fila_train,np.arange(len(exp.df)))
    g=exp.groups;G=g.max()+1;den=np.stack([np.bincount(g,weights=y==c,minlength=G) for c in [0,1]],axis=1)
    deltas=[];rows=[]
    for name in ('promedio','stacking'):
        d=cv[cv.modelo==name].sort_values('fila_train');p=d.y_pred.to_numpy()
        dif=[]
        for k,(_,va) in enumerate(exp.ctx.folds):
            v=C.ba_segura(y[va],p[va])-C.ba_segura(y[va],p0[va]);dif.append(v)
            deltas.append(dict(modelo=name,fold=k,diferencia=v))
        num=np.stack([np.bincount(g,weights=((p==c).astype(float)-(p0==c))*(y==c),minlength=G) for c in [0,1]]) .T
        rng=np.random.default_rng(42);boot=[]
        for _ in range(2000):
            ix=rng.integers(G,size=G);counts=den[ix].sum(axis=0)
            if np.all(counts>0):boot.append(.5*np.sum(num[ix].sum(axis=0)/counts))
        rows.append(dict(modelo=name,folds_ganados=int(np.sum(np.array(dif)>0)),diferencia_media=np.mean(dif),
                         diferencia_global=C.ba_segura(y,p)-C.ba_segura(y,p0),ic025=np.quantile(boot,.025),ic975=np.quantile(boot,.975),
                         pasa_filtro_cv=bool(np.sum(np.array(dif)>0)>=4 and np.mean(dif)>.003),
                         nota='Filtro exploratorio; no elige ganador ni sustituye robustez. IC condicional a predicciones fijas.'))
    pd.DataFrame(deltas).to_csv(out/'diferencias_folds.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'comparacion_cv.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('etapa',choices=['cv','robustez','todo','resumen'])
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--salida',type=Path,default=HERE/'stacking_resultados')
    args=ap.parse_args()
    if args.jobs<1:ap.error('--jobs debe ser positivo')
    params=json.loads(C.PARAMETROS.read_text())
    train_hash=C._verificar_train(C.cfg.DATA_RAW/'train.csv')
    exp=Experiment(load_train(),params,args.salida,train_hash)
    tasks=exp.tasks('todo' if args.etapa=='resumen' else args.etapa)
    missing=[t for t in tasks if exp.cached(exp.key('result','three',t[2],t[3])) is None]
    print('Run:',exp.run,'| tareas:',len(tasks),'| pendientes:',len(missing),flush=True)
    if args.dry_run:
        print('No se ajusta ni escribe nada. Cada tarea nueva: hasta 39 ajustes base + 4 logísticas; caches reducen el costo.')
        for s,k,tr,va in missing: print(s,k,'train',len(tr),'eval',len(va))
        print('CV inicial: estimación 15–30 min; todo con robustez: 45–90 min, según CPU/cache.')
        return
    exp.initialize()
    if args.etapa!='resumen':
        try:
            with parallel_config(backend='loky',inner_max_num_threads=1):
                Parallel(n_jobs=args.jobs)(delayed(exp.task)(*t) for t in missing)
        finally:
            summarize(exp)
    else:summarize(exp)
    print('RESUMEN REGENERADO' if args.etapa=='resumen' else 'COMPLETO', flush=True)

if __name__=='__main__':main()
