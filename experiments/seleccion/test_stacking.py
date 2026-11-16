"""Short synthetic tests; no long production training and no test.csv."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,ClassifierMixin
import stacking as S
from compe import config as cfg


def synthetic():
    n=180
    d=pd.DataFrame({c:np.zeros(n) for c in cfg.FEATURE_COLS})
    for c in cfg.CAT_COLS:d[c]='standard'
    d['hours_before_execution']=np.arange(n)
    d['execution_year']=2015+np.arange(n)%3
    d['execution_month']=1;d['execution_day']=1
    d['client_region']=np.where(np.arange(n)%2,'PRT','ARG')
    d['reservation_commitment']=np.where(np.arange(n)%7==0,'locked_allocation','flexible')
    d['canceled_job']=((np.arange(n)%11)<5).astype(int)
    d['execution_status']=np.where(d.canceled_job,'canceled','completed')
    d=pd.concat([d,d],ignore_index=True)
    d['_id']=np.arange(len(d))
    return d


class Spy(ClassifierMixin,BaseEstimator):
    fits=[]
    def __init__(self,kind='gradient_boosting'):self.kind=kind
    def fit(self,X,y):
        self.classes_=np.array([0,1]);self.mean=float(np.mean(y));self.train=X._id.to_numpy()
        Spy.fits.append(self.train.copy())
        return self
    def predict_proba(self,X):
        assert not set(self.train)&set(X._id), 'spy predicts its own training rows'
        offset=S.BASES.index(self.kind)*.17
        z=np.sin(X.hours_before_execution.to_numpy()*.3+offset)+self.mean-.5
        p=1/(1+np.exp(-z))
        return np.column_stack([1-p,p])


def context(root,df=None,params=None):
    return S.Experiment(synthetic() if df is None else df,params or {m:{} for m in S.BASES},root,
                        'synthetic',factory=Spy)


def test_nested_and_reusable():
    with tempfile.TemporaryDirectory() as tmp:
        e=context(tmp);tr,va=e.ctx.folds[0];Spy.fits=[]
        r=e.compute(tr,va)
        assert len(Spy.fits)==39, len(Spy.fits)
        assert all(not set(ids)&set(va) for ids in Spy.fits)
        # Each calibration validation fold has deep base fits whose training excludes it.
        for itr,iva in e.splits(tr):
            for deeptr,deepva in e.splits(tr[itr]):
                want=tr[itr][deeptr]
                assert sum(np.array_equal(ids,want) for ids in Spy.fits)==3
                assert not np.intersect1d(want,tr[iva]).size
        assert r['score'].shape==(len(va),3)
        assert r['inner_score'].shape==(len(tr),3)
        assert np.isfinite(r['score']).all()
        for j in range(3):assert r['threshold'][j]==S.best_threshold(e.y[tr],r['inner_score'][:,j])[0]
        Spy.fits=[];r2=e.compute(tr,va)
        assert not Spy.fits
        np.testing.assert_array_equal(r['score'],r2['score'])
        np.testing.assert_array_equal(r['threshold'],r2['threshold'])


def test_external_labels_do_not_affect_training():
    with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
        e=context(a);tr,va=e.ctx.folds[0];r=e.compute(tr,va)
        df=synthetic();df.loc[va,'canceled_job']=1-df.loc[va,'canceled_job']
        e2=context(b,df=df);r2=e2.compute(tr,va)
        np.testing.assert_array_equal(r['threshold'],r2['threshold'])
        np.testing.assert_array_equal(r['score'],r2['score'])
        np.testing.assert_array_equal(r['meta_coef'],r2['meta_coef'])


def test_cache_identity_and_corruption():
    with tempfile.TemporaryDirectory() as tmp:
        e=context(tmp);tr,va=e.ctx.folds[0];e.base(S.BASES[0],tr,va)
        other=context(tmp,params={m:{'n_estimators':2} for m in S.BASES})
        assert e.run_id!=other.run_id
        assert other.cached(other.key('base',S.BASES[0],tr,va)) is None
        p,_=e.paths(e.key('base',S.BASES[0],tr,va));p.write_bytes(b'corrupt')
        try:e.base(S.BASES[0],tr,va)
        except ValueError:pass
        else:raise AssertionError('corrupt cache accepted')


def test_summary_coverage_and_rebuild():
    with tempfile.TemporaryDirectory() as tmp:
        e=context(tmp);e.initialize()
        for task in e.tasks('cv'):e.task(*task)
        Spy.fits=[];S.summarize(e);assert not Spy.fits
        out=e.run/'resultados';p=pd.read_csv(out/'predicciones_externas.csv')
        assert len(p)==3*len(e.df)
        for _,d in p.groupby('modelo'):
            assert d.fila_train.is_unique and len(d)==len(e.df)
        met=pd.read_csv(out/'metricas_externas_globales.csv')
        assert met.escenario_completo.all()
        assert np.all(met[['TN','FP','FN','TP']].sum(axis=1)==len(e.df))
        ref=(out/'comparacion_cv.csv').read_bytes();S.summarize(e)
        assert (out/'comparacion_cv.csv').read_bytes()==ref
        # Every robust scenario also separates complete groups.
        for _,_,tr,va in e.tasks('robustez'):e.check_split(tr,va)


def test_valid_legacy_reuse():
    # Read-only production cache checks, no fits. Ensures current HistGB reference can be reused.
    params=json.loads(S.C.PARAMETROS.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        e=S.Experiment(S.load_train(),params,tmp,S.C._verificar_train(cfg.DATA_RAW/'train.csv'))
        for k,(tr,va) in enumerate(e.ctx.folds):
            old=e.legacy('interno','hist_gradient_boosting',tr)
            ext=e.legacy('ajuste','hist_gradient_boosting',tr,va)
            assert old is not None and ext is not None
            expected=np.load(e.ctx.cache/f'interno__hist_gradient_boosting__sin_assigned__cv__f{k}.npz')['umbral']
            np.testing.assert_allclose(S.best_threshold(e.y[tr],old['oof_interno'])[0],expected)


if __name__=='__main__':
    for fn in [test_nested_and_reusable,test_external_labels_do_not_affect_training,
               test_cache_identity_and_corruption,test_summary_coverage_and_rebuild,test_valid_legacy_reuse]:
        fn();print('PASS',fn.__name__,flush=True)

# Additional production-pipeline smoke test on synthetic data only.
def test_real_pipelines():
    params={m:{} for m in S.BASES}
    params['gradient_boosting']={'n_estimators':2,'max_depth':2}
    params['hist_gradient_boosting']={'max_iter':2,'min_samples_leaf':5}
    params['random_forest']={'n_estimators':3,'max_depth':2}
    with tempfile.TemporaryDirectory() as tmp:
        e=S.Experiment(synthetic().drop(columns='_id'),params,tmp,'synthetic-real')
        tr,va=e.ctx.folds[0]
        r=e.compute(tr,va)
        assert np.isfinite(r['score']).all()
        assert np.all((r['score']>=0)&(r['score']<=1))
        # None of the preprocessing pipelines uses assigned_gpu_type.
        changed=e.df.copy();changed['assigned_gpu_type']='never_seen'
        e2=S.Experiment(changed,params,Path(tmp)/'changed','synthetic-changed')
        r2=e2.compute(tr,va)
        np.testing.assert_array_equal(r['score'],r2['score'])


def test_parallel_matches_serial():
    with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
        e=context(a);f=context(b);tasks=e.tasks('cv')[:2]
        for t in tasks:e.task(*t)
        with S.parallel_config(backend='loky',inner_max_num_threads=1):
            S.Parallel(n_jobs=2)(S.delayed(f.task)(*t) for t in tasks)
        for _,_,tr,va in tasks:
            r=e.cached(e.key('result','three',tr,va));r2=f.cached(f.key('result','three',tr,va))
            np.testing.assert_array_equal(r['score'],r2['score'])
            np.testing.assert_array_equal(r['threshold'],r2['threshold'])

if __name__=='__main__':
    for fn in [test_real_pipelines,test_parallel_matches_serial]:
        fn();print('PASS',fn.__name__,flush=True)
