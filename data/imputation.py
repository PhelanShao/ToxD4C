import pandas as pd


def median_impute_with_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric columns with their median and append missing flags.

    Parameters
    ----------
    df: pd.DataFrame
        Input feature matrix which may contain missing values.

    Returns
    -------
    pd.DataFrame
        Concatenation of the imputed features and binary missing indicators
        for each original column (suffix ``"__isna"``).
    """
    df = df.copy()
    flags = df.isna().astype(int)
    imputed = df.fillna(df.median(numeric_only=True))
    flags.columns = [f"{c}__isna" for c in flags.columns]
    return pd.concat([imputed, flags], axis=1)
