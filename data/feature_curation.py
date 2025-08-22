import numpy as np
import pandas as pd
from typing import Iterable, Tuple

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception:  # pragma: no cover - optional dependency
    variance_inflation_factor = None

def correlation_prune(df: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, Iterable[str]]:
    """Remove features with absolute pairwise correlation above ``threshold``.

    Parameters
    ----------
    df: pd.DataFrame
        Input feature matrix.
    threshold: float, default 0.95
        Absolute correlation above which a feature is dropped.

    Returns
    -------
    pruned_df: pd.DataFrame
        DataFrame with highly correlated features removed.
    dropped: Iterable[str]
        Names of the columns that were dropped.
    """
    df = df.copy()
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

def compute_vif(df: pd.DataFrame) -> pd.Series:
    """Compute variance inflation factors for the columns in ``df``.

    Requires ``statsmodels`` to be installed. A ``RuntimeError`` is raised if
    the dependency is unavailable.
    """
    if variance_inflation_factor is None:
        raise RuntimeError("statsmodels is required for VIF computation")
    X = df.values
    vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return pd.Series(vifs, index=df.columns)

def vif_prune(df: pd.DataFrame, max_vif: float = 10.0) -> pd.DataFrame:
    """Iteratively drop columns with VIF greater than ``max_vif``."""
    if variance_inflation_factor is None:
        raise RuntimeError("statsmodels is required for VIF computation")
    df = df.copy()
    while True:
        vifs = compute_vif(df)
        worst = vifs.idxmax()
        if vifs.max() <= max_vif:
            break
        df = df.drop(columns=[worst])
    return df
