"""
Pembentukan sequence D-30..D-1 -> (1, 30, 8) dan scaler.transform().

Direplikasi dari bagian Stage 10 yang dibawa ke deployment (lihat
STAGE_DEPLOYMENT_MAP.md baris Stage 10): HANYA scaling (`scaler.transform`)
dan pembentukan window (`window = features[i-lookback:i]`). Split
Train/Val/Test dan label encoding TIDAK dijalankan di sini (training-only).

Dilarang: scaler.fit(...) / scaler.fit_transform(...) (lihat
INFERENCE_CONTRACT.md Bagian 4 & 9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import FEATURE_COLUMNS, LOOKBACK


def extract_feature_matrix(df: pd.DataFrame, window_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Ambil baris untuk window_dates (ascending, D-lookback..D-1) dari df
    hasil preprocessing Stage 7, urut sesuai window_dates, dengan kolom
    persis FEATURE_COLUMNS (urutan WAJIB dipertahankan)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    indexed = df.set_index("date")

    window_dates = [pd.Timestamp(d).normalize() for d in window_dates]
    missing = [d for d in window_dates if d not in indexed.index]
    if missing:
        raise KeyError(
            f"Tanggal berikut tidak tersedia pada dataset kerja: "
            f"{[d.strftime('%Y-%m-%d') for d in missing]}"
        )

    ordered = indexed.loc[window_dates, FEATURE_COLUMNS]
    return ordered


def scale_features(feature_matrix: pd.DataFrame, scaler) -> np.ndarray:
    """scaler.transform() SAJA -- dilarang scaler.fit()/fit_transform().
    Urutan kolom yang dikirim ke scaler diverifikasi sama dengan
    scaler.feature_names_in_ sebelum transform (validation gate)."""
    expected = list(getattr(scaler, "feature_names_in_", FEATURE_COLUMNS))
    actual = list(feature_matrix.columns)
    if actual != expected:
        raise ValueError(
            f"Urutan fitur tidak cocok dengan scaler.feature_names_in_: "
            f"actual={actual}, expected={expected}"
        )
    # Kirim sebagai DataFrame (bukan ndarray polos) agar scaler yang di-fit
    # dengan feature_names_in_ tidak memicu UserWarning "X does not have
    # valid feature names" dari scikit-learn. Tidak mengubah nilai/urutan
    # transform, hanya representasi input.
    return scaler.transform(feature_matrix.astype(float))


def build_input_sequence(scaled_features: np.ndarray, lookback: int = LOOKBACK) -> np.ndarray:
    """Reshape (lookback, n_features) -> (1, lookback, n_features)."""
    if scaled_features.shape[0] != lookback:
        raise ValueError(
            f"Jumlah baris fitur ({scaled_features.shape[0]}) tidak sama dengan "
            f"lookback ({lookback})."
        )
    return scaled_features.reshape(1, lookback, scaled_features.shape[1])
