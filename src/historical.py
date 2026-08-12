"""
Adapter untuk mengambil data dari dataset historis terintegrasi (integrated_dataset.csv).
"""

from __future__ import annotations

import os
import pandas as pd
from config.settings import HISTORICAL_DATASET_PATH

_historical_cache = None

def get_historical_df() -> pd.DataFrame:
    """Load dan kembalikan dataset historis. Menggunakan caching internal agar tidak lambat."""
    global _historical_cache
    if _historical_cache is not None:
        return _historical_cache

    if not os.path.exists(HISTORICAL_DATASET_PATH):
        raise FileNotFoundError(
            f"Historical dataset file not found at: {HISTORICAL_DATASET_PATH}"
        )

    df = pd.read_csv(HISTORICAL_DATASET_PATH)
    # Parse tanggal dan normalisasi format
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    _historical_cache = df
    return _historical_cache

def get_historical_row(date_str: str) -> dict | None:
    """Lookup data historis untuk satu tanggal."""
    try:
        df = get_historical_df()
        if date_str in df.index:
            row = df.loc[date_str]
            if isinstance(row, pd.DataFrame):
                # Jika ada duplikasi (meskipun tidak harus ada), ambil baris pertama
                row = row.iloc[0]
            return row.to_dict()
    except Exception:
        pass
    return None
