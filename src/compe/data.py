"""Carga de datos. Todo el proyecto lee los CSV por aca."""
import pandas as pd

from .config import DATA_RAW, FEATURE_COLS, TARGET


def load_train() -> pd.DataFrame:
    """train.csv completo (34 columnas, incluye target y columnas con fuga)."""
    return pd.read_csv(DATA_RAW / "train.csv")


def load_test() -> pd.DataFrame:
    """Holdout interno. Usar solo para la comparacion final."""
    return pd.read_csv(DATA_RAW / "test.csv")


def load_split_groups(particion: str = "train") -> pd.Series:
    """`grupo_duplicados` del split, alineado fila a fila con el CSV de la particion."""
    idx = pd.read_csv(DATA_RAW / "indices_split.csv")
    return idx.loc[idx["particion"] == particion, "grupo_duplicados"].reset_index(drop=True)


def feature_groups(df: pd.DataFrame) -> pd.Series:
    """Grupo por igualdad de features (sin target ni fugas), nulos tratados como iguales.

    Mas estricto que `grupo_duplicados`, que tambien exige igual target y fecha de estado.
    Es el grupo correcto para validacion cruzada: dos filas con el mismo X nunca
    quedan una en entrenamiento y otra en validacion.
    """
    return df.groupby(FEATURE_COLS, dropna=False, sort=False).ngroup().rename("feature_group")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS].copy(), df[TARGET].copy()
