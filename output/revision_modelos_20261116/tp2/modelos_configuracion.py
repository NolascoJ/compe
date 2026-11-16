"""Configuraciones seleccionadas y utilizadas en la presentación del TP2.

Las búsquedas de ajuste_rf.py, ajuste_svm.py y ajuste_hiperparametros.py
documentan la selección. Estas constantes reproducen los candidatos elegidos;
no se vuelven a seleccionar modelos usando test.
"""

SEMILLA = 42
FOLDS = 5
METRICAS = ("roc_auc", "recall")

PARAMETROS_RF = {
    "clasificador__n_estimators": 100,
    "clasificador__max_depth": 12,
    "clasificador__min_samples_split": 200,
    "clasificador__min_samples_leaf": 10,
    "clasificador__max_features": "sqrt",
}
PARAMETROS_CANDIDATOS = {
    "Naive Bayes": {},
    "SVM": {
        "clasificador__kernel": "rbf",
        "clasificador__C": 0.1,
        "clasificador__gamma": 0.001,
    },
    "KNN": {"clasificador__n_neighbors": 31},
    "RF": PARAMETROS_RF,
}
