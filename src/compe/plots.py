"""Estilo comun de graficos y helpers reutilizables del EDA."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES, TARGET

C_OK = "#2a78d6"      # completado (0)
C_CANCEL = "#eb6834"  # cancelado (1)
C_NEUTRAL = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9,
    })


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")


def rate_table(df: pd.DataFrame, col: str, top: int | None = None, min_n: int = 0) -> pd.DataFrame:
    """Filas y tasa de cancelacion por valor de `col` (nulos como '(nulo)')."""
    s = df[col].astype("object").where(df[col].notna(), "(nulo)")
    t = df.groupby(s)[TARGET].agg(n="size", tasa="mean")
    t["pct_filas"] = t["n"] / len(df)
    t = t[t["n"] >= min_n].sort_values("n", ascending=False)
    return t.head(top) if top else t


def plot_rate(df: pd.DataFrame, col: str, ax=None, top: int = 12, sort_by: str = "n", title: str | None = None):
    """Barras horizontales: tasa de cancelacion por categoria, con n anotado."""
    t = rate_table(df, col, top=top)
    if sort_by == "tasa":
        t = t.sort_values("tasa", ascending=False)
    t = t.iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.32 * len(t) + 1))
    base = df[TARGET].mean()
    ax.barh([str(i) for i in t.index], t["tasa"], color=C_CANCEL, height=0.62)
    ax.axvline(base, color=INK2, lw=1, ls="--")
    for y, (r, n) in enumerate(zip(t["tasa"], t["n"])):
        ax.text(r + 0.012, y, f"{r:.0%}  (n={n:,})".replace(",", "."), va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 1.25)
    ax.set_xticks(np.arange(0, 1.01, 0.25), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    ax.grid(axis="y", visible=False)
    ax.set_title(title or col)
    ax.set_xlabel("tasa de cancelación (línea punteada = global)")
    return ax


def plot_binned_rate(df: pd.DataFrame, col: str, bins, ax=None, title: str | None = None):
    """Tasa de cancelacion por tramo de una variable numerica."""
    b = pd.cut(df[col], bins=bins, include_lowest=True, right=True)
    t = df.groupby(b, observed=True)[TARGET].agg(n="size", tasa="mean")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    labels = [f"{int(i.left) if i.left > 0 else 0}–{int(i.right)}" for i in t.index]
    ax.bar(labels, t["tasa"], color=C_CANCEL, width=0.7)
    ax.axhline(df[TARGET].mean(), color=INK2, lw=1, ls="--")
    for x, (r, n) in enumerate(zip(t["tasa"], t["n"])):
        ax.text(x, r + 0.02, f"{r:.0%}\nn={n:,}".replace(",", "."), ha="center", fontsize=7.5, color=INK2)
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.01, 0.25), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    ax.grid(axis="x", visible=False)
    ax.set_title(title or col)
    ax.set_ylabel("tasa de cancelación")
    ax.tick_params(axis="x", rotation=30)
    return ax


C_TODO = "#2a78d6"    # serie "todo train"
C_SINL = "#eb6834"    # serie "sin locked_allocation"


def _key(d: pd.DataFrame, col: str, bins=None, labels=None) -> pd.Series:
    if bins is not None:
        return pd.cut(d[col], bins=bins, labels=labels, include_lowest=True)
    return d[col].astype("object").where(d[col].notna(), "(nulo)")


def rate_dual(df: pd.DataFrame, col: str, bins=None, labels=None, top: int | None = None) -> pd.DataFrame:
    """Tasa de cancelacion por valor de `col`: todo train y sin locked_allocation."""
    sin = df[df["reservation_commitment"] != "locked_allocation"]
    a = df.groupby(_key(df, col, bins, labels), observed=True)[TARGET].agg(["size", "mean"])
    b = sin.groupby(_key(sin, col, bins, labels), observed=True)[TARGET].agg(["size", "mean"])
    t = pd.DataFrame({"filas": a["size"], "tasa": a["mean"], "filas_sin_locked": b["size"], "tasa_sin_locked": b["mean"]})
    if bins is None:
        t = t.sort_values("filas", ascending=False)
    t = t.head(top) if top else t
    t["filas_sin_locked"] = t["filas_sin_locked"].fillna(0).astype(int)
    return t.round({"tasa": 3, "tasa_sin_locked": 3})


def plot_rate_dual(df: pd.DataFrame, col: str, bins=None, labels=None, top: int = 8, ax=None, title: str | None = None):
    """Barras agrupadas: tasa en todo train vs sin locked_allocation, por valor de `col`."""
    t = rate_dual(df, col, bins, labels, top).iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.5 * len(t) + 1.2))
    yy = np.arange(len(t))
    h = 0.36
    ax.barh(yy + h / 2 + 0.02, t["tasa"], height=h, color=C_TODO, label="todo train")
    ax.barh(yy - h / 2 - 0.02, t["tasa_sin_locked"].fillna(0), height=h, color=C_SINL, label="sin locked_allocation")
    for i, (a, b) in enumerate(zip(t["tasa"], t["tasa_sin_locked"])):
        ax.text(a + 0.012, i + h / 2 + 0.02, f"{a:.0%}", va="center", fontsize=7.5, color=INK2)
        if pd.notna(b):
            ax.text(b + 0.012, i - h / 2 - 0.02, f"{b:.0%}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(yy, [str(i) for i in t.index])
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.arange(0, 1.01, 0.25), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    ax.grid(axis="y", visible=False)
    ax.set_title(title or col)
    return ax
