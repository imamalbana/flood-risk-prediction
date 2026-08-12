"""
Akuisisi & standardisasi data harian Ogimet.

Logika inti (fetch_month_data, find_column, placeholder standardisasi)
direplikasi dari `ogimet_downloader.py` dan Stage 2 (lihat
STAGE_DEPLOYMENT_MAP.md baris Stage 2), diparameterkan untuk rentang
tanggal sembarang (bukan hardcoded satu YEAR) sesuai catatan audit
Bagian 1.6 (INFERENCE_AUDIT_REPORT.md).

Placeholder standardisasi Ogimet (WAJIB, tidak boleh diubah):
    "Tr"    -> 0.0
    "----"  -> NaN
    "-----" -> NaN
sebelum konversi numerik.
"""

from __future__ import annotations

import calendar
import io
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config.settings import CACHE_DIR, OGIMET_PLACEHOLDER_MAP, STATION_ID

logger = logging.getLogger("inference_pipeline.ogimet")

SUMMARY_HOUR = 12  # identik dengan ogimet_downloader.py (SUMMARY_HOUR)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def find_column(df: pd.DataFrame, key1: str, key2: Optional[str] = None):
    """Identik dengan ogimet_downloader.py::find_column."""
    for col in df.columns:
        if isinstance(col, tuple):
            m1 = key1.lower() in str(col[0]).lower()
            if key2:
                m2 = key2.lower() in str(col[1]).lower()
                if m1 and m2:
                    return col
            else:
                if m1:
                    return col
        else:
            if key1.lower() in str(col).lower():
                return col
    return None


def fetch_month_data(
    year: int,
    month: int,
    station_id: str = STATION_ID,
    max_retries: int = 5,
    retry_delay: int = 5,
    session: Optional[requests.Session] = None,
) -> Optional[pd.DataFrame]:
    """Unduh tabel ringkasan harian mentah dari Ogimet untuk satu bulan.

    Identik dengan ogimet_downloader.py::fetch_month_data, kecuali
    `station_id` diparameterkan (bukan konstanta modul global) dan
    `session` opsional untuk reuse koneksi HTTP.
    """
    _, num_days = calendar.monthrange(year, month)
    url = (
        "https://www.ogimet.com/cgi-bin/gsynres"
        "?lang=en&ord=DIR&sum=YES"
        f"&ano={year}&mes={month:02d}&day={num_days}"
        f"&hora={SUMMARY_HOUR:02d}&ndays={num_days}&ndec=2&ind={station_id}"
    )

    http = session or requests

    for attempt in range(1, max_retries + 1):
        try:
            response = http.get(url, headers=HEADERS, timeout=40)
            if response.status_code != 200:
                raise Exception(f"HTTP Status {response.status_code}")

            html_content = response.text
            if any(
                phrase in html_content.lower()
                for phrase in ["no valid data found", "no data found", "sin datos"]
            ):
                logger.info("[%s-%02d] Tidak ada data di Ogimet.", year, month)
                return None

            tables = pd.read_html(io.StringIO(html_content))

            data_table = None
            for t in tables:
                has_date = any(
                    "date" in str(c).lower()
                    for col in t.columns
                    for c in (col if isinstance(col, tuple) else [col])
                )
                if has_date and t.shape[1] >= 10 and t.shape[0] >= 15:
                    data_table = t
                    break

            if data_table is None or data_table.empty:
                raise Exception("Tabel data cuaca utama tidak ditemukan pada respons HTML.")

            return data_table

        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s-%02d] Percobaan %d gagal: %s", year, month, attempt, exc)
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                raise Exception(
                    f"Gagal mengambil data untuk {year}-{month:02d} setelah {max_retries} percobaan."
                ) from exc

    return None


def _standardize_month(df_raw: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Bangun DataFrame standar (nama kolom baku) dari tabel mentah satu
    bulan, identik dengan blok `main()` ogimet_downloader.py."""
    col_date = find_column(df_raw, "date")
    col_prec = find_column(df_raw, "prec")
    col_temp = find_column(df_raw, "temperature", "avg")
    col_rh = find_column(df_raw, "hr")

    missing_cols = []
    if col_date is None:
        missing_cols.append("Date")
    if col_prec is None:
        missing_cols.append("Prec. (mm) -> Daily Rainfall (RR)")
    if col_temp is None:
        missing_cols.append("Temperature Avg -> Air Temperature")
    if col_rh is None:
        missing_cols.append("Hr. Avg -> Relative Humidity")
    if missing_cols:
        raise ValueError(
            f"[{year}-{month:02d}] Kolom hilang pada struktur HTML Ogimet: {missing_cols}"
        )

    df_month = pd.DataFrame()
    formatted_dates = []
    for val in df_raw[col_date]:
        s_val = str(val).strip()
        if "/" in s_val:
            try:
                parts = s_val.split("/")
                m_val = int(parts[0])
                d_val = int(parts[1])
                formatted_dates.append(f"{year:04d}-{m_val:02d}-{d_val:02d}")
            except Exception:
                formatted_dates.append(s_val)
        else:
            formatted_dates.append(s_val)

    df_month["date"] = pd.to_datetime(formatted_dates, errors="coerce")
    df_month["rr"] = df_raw[col_prec].values
    df_month["tavg"] = df_raw[col_temp].values
    df_month["rh"] = df_raw[col_rh].values

    return df_month


def _apply_ogimet_placeholder(series: pd.Series) -> pd.Series:
    """Placeholder standardisasi Stage 2 (WAJIB, urutan tidak boleh diubah):
    'Tr' -> 0.0, '----'/'-----' -> NaN, lalu konversi numerik."""
    series = series.astype(object)
    for token, replacement in OGIMET_PLACEHOLDER_MAP.items():
        series = series.where(series.astype(str).str.strip() != token, replacement)
    return pd.to_numeric(series, errors="coerce")


def get_ogimet_daily(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    station_id: str = STATION_ID,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Ambil data harian Ogimet (rr, tavg, rh) sudah distandardisasi.
    Jika data ada di historical dataset (integrated_dataset.csv), gunakan langsung.
    Jika tidak ada, fallback ke scraping Ogimet/cache on-disk.
    """
    from src.historical import get_historical_row

    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    
    dates = pd.date_range(start_date, end_date, freq="D")
    rows = []
    missing_dates = []

    # Coba lookup data historis per tanggal
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        hist_row = get_historical_row(date_str)
        if hist_row is not None:
            # Map kolom yang sesuai
            rows.append({
                "date": d,
                "rr": hist_row.get("rr"),
                "tavg": hist_row.get("tavg"),
                "rh": hist_row.get("rh"),
            })
        else:
            missing_dates.append(d)

    # Fallback ke existing logic untuk missing_dates
    if missing_dates:
        os.makedirs(CACHE_DIR, exist_ok=True)
        missing_start = min(missing_dates)
        missing_end = max(missing_dates)
        months = pd.period_range(missing_start, missing_end, freq="M")
        monthly_dfs = []

        with requests.Session() as session:
            for period in months:
                year, month = period.year, period.month
                cache_path = os.path.join(
                    CACHE_DIR, f"ogimet_{station_id}_{year}_{month:02d}.csv"
                )
                if use_cache and os.path.exists(cache_path):
                    df_month = pd.read_csv(cache_path, parse_dates=["date"])
                else:
                    df_raw = fetch_month_data(year, month, station_id=station_id, session=session)
                    if df_raw is None:
                        continue
                    df_month = _standardize_month(df_raw, year, month)
                    if use_cache:
                        df_month.to_csv(cache_path, index=False)
                    time.sleep(2.0)

                monthly_dfs.append(df_month)

        if monthly_dfs:
            full_df = pd.concat(monthly_dfs, ignore_index=True)
            full_df["rr"] = _apply_ogimet_placeholder(full_df["rr"])
            full_df["tavg"] = _apply_ogimet_placeholder(full_df["tavg"])
            full_df["rh"] = _apply_ogimet_placeholder(full_df["rh"])

            # Ambil yang sesuai dengan missing_dates
            mask = full_df["date"].isin(missing_dates)
            fallback_rows = full_df.loc[mask, ["date", "rr", "tavg", "rh"]].to_dict("records")
            rows.extend(fallback_rows)

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["date", "rr", "tavg", "rh"])

    result["date"] = pd.to_datetime(result["date"])
    # Terapkan konversi numerik untuk berjaga-jaga
    result["rr"] = pd.to_numeric(result["rr"], errors="coerce")
    result["tavg"] = pd.to_numeric(result["tavg"], errors="coerce")
    result["rh"] = pd.to_numeric(result["rh"], errors="coerce")

    return result.sort_values("date").reset_index(drop=True)

