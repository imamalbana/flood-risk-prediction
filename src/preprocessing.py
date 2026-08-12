"""
Missing-value handling setara Stage 7 (WAJIB direplikasi persis, lihat
INFERENCE_CONTRACT.md Bagian 7 & INFERENCE_AUDIT_REPORT.md Bagian 1.3):

- Ogimet (rr, tavg, rh):
    df.set_index("date")[col].interpolate(method="time", limit_direction="both")
  dijalankan pada index tanggal harian lengkap.

- SounderPy (cin, kindex, li, tt, sweat):
    imputasi median bulanan (group by bulan kalender 1-12, lintas tahun),
    HANYA pada baris selection_status == "SELECTED".
    Median diambil dari config/stage7_monthly_medians.json (hasil ekstraksi
    langsung dari output eksekusi Stage 7, bukan dihitung ulang).

- NO_SOUNDING: kelima fitur atmosfer TETAP NaN, tidak diisi median.

CATATAN OPEN DECISION 1 (INFERENCE_CONTRACT.md Bagian 10): ukuran buffer
data sebelum D-30 yang dibutuhkan agar interpolasi time-based pada window
terbatas berperilaku setara dengan interpolasi pada deret penuh BELUM
ditentukan pada CHECKPOINT 01. Modul ini TIDAK mengarang nilai buffer --
fungsi `apply_stage7_missing_value_handling` beroperasi pada persis
dataframe yang diberikan (bisa berupa window D-30..D-1 saja, atau window
plus buffer tambahan jika caller menyuplainya secara eksplisit). Semakin
sedikit data yang tersedia mendekati ujung window, semakin besar risiko
hasil interpolasi berbeda dari training -- ini adalah keterbatasan yang
diwarisi dari OPEN DECISION 1, bukan bug pada modul ini.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from config.settings import (
    OGIMET_FEATURE_COLUMNS,
    SELECTED_STATUS,
    SOUNDING_FEATURE_COLUMNS,
    STAGE7_MEDIANS_PATH,
)


def load_monthly_medians(path: str = STAGE7_MEDIANS_PATH) -> dict:
    """Baca config/stage7_monthly_medians.json. Key bulan pada file adalah
    string "1".."12"; dikonversi ke int di sini untuk kemudahan lookup."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    monthly = raw["monthly_medians"]
    return {int(month_str): values for month_str, values in monthly.items()}


def _interpolate_ogimet(df: pd.DataFrame) -> pd.DataFrame:
    """Identik dengan Stage 7: interpolate(method="time", limit_direction="both")
    per kolom Ogimet, pada index tanggal."""
    df = df.copy()
    indexed = df.set_index("date")
    for col in OGIMET_FEATURE_COLUMNS:
        indexed[col] = indexed[col].interpolate(method="time", limit_direction="both")
    return indexed.reset_index()


def _impute_sounding_medians(df: pd.DataFrame, monthly_medians: dict) -> pd.DataFrame:
    """Isi median bulanan HANYA pada baris selection_status == SELECTED.
    Baris NO_SOUNDING dibiarkan NaN (tidak disentuh)."""
    df = df.copy()
    selected_mask = df["selection_status"] == SELECTED_STATUS

    for col in SOUNDING_FEATURE_COLUMNS:
        month_series = df["date"].dt.month
        median_for_row = month_series.map(
            lambda m: monthly_medians.get(m, {}).get(col)
        )
        needs_fill = selected_mask & df[col].isna()
        df.loc[needs_fill, col] = median_for_row[needs_fill]

    return df


def apply_stage7_missing_value_handling(
    df: pd.DataFrame, monthly_medians: Optional[dict] = None
) -> pd.DataFrame:
    """Terapkan missing-value handling Stage 7 pada dataframe hasil
    integrasi Stage 5 (kolom: date, rr, tavg, rh, selection_status,
    selected_hour, cin, kindex, li, tt, sweat).

    Urutan (tidak boleh ditukar): interpolasi Ogimet dahulu, baru imputasi
    median bulanan SounderPy pada baris SELECTED. NO_SOUNDING tetap NaN.
    """
    if monthly_medians is None:
        monthly_medians = load_monthly_medians()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = _interpolate_ogimet(df)
    df = _impute_sounding_medians(df, monthly_medians)

    return df
