"""I/O helpers for the project."""
from pathlib import Path
import pandas as pd
import pickle
import gzip


def _has_parquet_support() -> bool:
    """Check if parquet engine is available."""
    try:
        import pyarrow  # noqa
        return True
    except ImportError:
        try:
            import fastparquet  # noqa
            return True
        except ImportError:
            return False


HAS_PARQUET = _has_parquet_support()


def save_df(df: pd.DataFrame, path: str | Path) -> Path:
    """Sauvegarde un DataFrame.

    En production : parquet (rapide, compact, typé).
    En fallback : pickle.gz (si pyarrow non installé).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_PARQUET:
        if path.suffix != ".parquet":
            path = path.with_suffix(".parquet")
        df.to_parquet(path, index=False, compression="snappy")
    else:
        if path.suffix != ".pkl":
            path = path.with_suffix(".pkl.gz")
        with gzip.open(path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_df(path: str | Path) -> pd.DataFrame:
    """Charge un DataFrame (parquet ou pickle.gz selon extension)."""
    path = Path(path)
    if not path.exists():
        # Tenter les 2 extensions
        for ext in (".parquet", ".pkl.gz", ".pkl"):
            alt = path.with_suffix(ext)
            if alt.exists():
                path = alt
                break
        else:
            raise FileNotFoundError(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffixes[-2:] == [".pkl", ".gz"] or path.name.endswith(".pkl.gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")
