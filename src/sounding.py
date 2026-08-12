"""
Seleksi sounding harian per NOMINAL DATE + perhitungan indeks atmosfer.

Source of truth untuk nominal/request date D:

    D 12Z
      ↓
    SUCCESS → SELECTED → STOP

    jika 12Z tidak tersedia:
      ↓
    D 00Z
      ↓
    SUCCESS → SELECTED → STOP

    jika keduanya tidak tersedia:
      ↓
    NO_SOUNDING

Ketentuan wajib:
- 12Z SUCCESS → JANGAN mencoba 00Z.
- fallback 00Z menggunakan request_date D (bukan D-1).
- satu nominal date → maksimal satu selected sounding.

Catatan definisi tanggal:
- `request_date` = NOMINAL DATE (kunci untuk cache dan integration).
- `observation_datetime` = waktu aktual yang tercatat pada sounding.
  Keduanya tidak harus sama. Contoh: sounding 00Z yang di-record pada
  D-1 23:30 tetap memiliki nominal_date = D (sesuai request_date).

VERIFIKASI CACHE (HARDENING): kedua percobaan (12Z dan fallback 00Z)
memakai `wyoming._try_hour_cached` (cache on-disk per (request_date, hour)
di `config.CACHE_DIR`), bukan `wyoming._try_hour` mentah.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import requests

from config.settings import NO_SOUNDING_STATUS, SELECTED_STATUS
from src.atmospheric_indices import compute_indices_for_sounding
from src.wyoming import _try_hour_cached

logger = logging.getLogger("inference_pipeline.sounding")


def _attempt_nominal_date(
    nominal_date: pd.Timestamp, session: requests.Session, use_cache: bool = True
) -> dict:
    """Coba dapatkan sounding untuk satu nominal_date, mengikuti urutan
    12Z(D) dulu, lalu fallback 00Z(D).

    Source of truth untuk nominal date D:
        12Z(D) → SUCCESS: SELECTED, STOP.
        12Z(D) → FAIL: coba 00Z(D) → SUCCESS: SELECTED, STOP.
        keduanya FAIL: NO_SOUNDING.

    request_date selalu = nominal_date (D) untuk kedua percobaan.
    observation_datetime boleh berbeda (mis. 00Z yang direkam D-1 23:30
    tetap memiliki nominal_date = D sesuai request_date).

    Menggunakan `_try_hour_cached` agar kombinasi (request_date, hour)
    yang sudah pernah diambil dibaca dari cache on-disk.
    """
    nominal_str = nominal_date.strftime("%Y-%m-%d")
    request_date_str = nominal_date.strftime("%Y-%m-%d")  # D untuk kedua percobaan

    # 12Z pada request_date = D
    result, _ = _try_hour_cached(
        request_date_str, "12", 12, "SUCCESS_12Z", session,
        use_cache=use_cache,
    )
    if result is not None:
        return {
            "nominal_date": nominal_str,
            "request_date": request_date_str,
            "selected_hour": 12,
            "selection_status": SELECTED_STATUS,
            "source": result.source,
            "profile_df": result.df,
        }

    # fallback: 00Z pada request_date = D (BUKAN D-1)
    result, _ = _try_hour_cached(
        request_date_str, "00", 0, "SUCCESS_00Z", session,
        use_cache=use_cache,
    )
    if result is not None:
        return {
            "nominal_date": nominal_str,
            "request_date": request_date_str,
            "selected_hour": 0,
            "selection_status": SELECTED_STATUS,
            "source": result.source,
            "profile_df": result.df,
        }

    return {
        "nominal_date": nominal_str,
        "request_date": None,
        "selected_hour": None,
        "selection_status": NO_SOUNDING_STATUS,
        "source": None,
        "profile_df": None,
    }


def get_sounding_row_for_date(
    nominal_date: pd.Timestamp,
    session: Optional[requests.Session] = None,
    use_cache: bool = True,
) -> dict:
    """Bangun satu baris sounding (status seleksi + 5 fitur atmosfer) untuk
    satu tanggal kalender (nominal_date).

    Menggunakan data historis dari integrated_dataset.csv terlebih dahulu jika ada.
    Jika tidak ada, fallback ke active Wyoming downloading/caching.
    """
    from src.historical import get_historical_row

    norm_date = pd.Timestamp(nominal_date).normalize()
    date_str = norm_date.strftime("%Y-%m-%d")

    # Coba data historis
    hist_row = get_historical_row(date_str)
    if hist_row is not None:
        sel_hour_raw = hist_row.get("selected_hour")
        if pd.isna(sel_hour_raw) or sel_hour_raw == "MISSING" or not sel_hour_raw:
            selected_hour = None
        else:
            # Parse '12Z' or '00Z' or '12' or '0'
            s_hour = str(sel_hour_raw).upper().strip()
            if "12" in s_hour:
                selected_hour = 12
            elif "0" in s_hour:
                selected_hour = 0
            else:
                selected_hour = None

        row = {
            "date": date_str,
            "selection_status": hist_row.get("selection_status"),
            "selected_hour": selected_hour,
            "cin": hist_row.get("cin"),
            "kindex": hist_row.get("kindex"),
            "li": hist_row.get("li"),
            "tt": hist_row.get("tt"),
            "sweat": hist_row.get("sweat"),
        }
        # Terapkan NaN jika status adalah NO_SOUNDING
        if row["selection_status"] == NO_SOUNDING_STATUS:
            row["selected_hour"] = None
            row["cin"] = None
            row["kindex"] = None
            row["li"] = None
            row["tt"] = None
            row["sweat"] = None

        # Cast to float or None
        for k in ["cin", "kindex", "li", "tt", "sweat"]:
            if row[k] is not None:
                val = float(row[k])
                row[k] = None if pd.isna(val) else val

        return row

    # Fallback ke Wyoming downloader
    own_session = session is None
    if own_session:
        session = requests.Session()
    try:
        selection = _attempt_nominal_date(
            norm_date, session, use_cache=use_cache
        )
    finally:
        if own_session:
            session.close()

    row = {
        "date": selection["nominal_date"],
        "selection_status": selection["selection_status"],
        "selected_hour": selection["selected_hour"],
        "cin": None,
        "kindex": None,
        "li": None,
        "tt": None,
        "sweat": None,
    }

    if selection["selection_status"] == SELECTED_STATUS and selection["profile_df"] is not None:
        try:
            indices = compute_indices_for_sounding(selection["profile_df"])
            row["cin"] = indices["cin"]
            row["kindex"] = indices["kindex"]
            row["li"] = indices["li"]
            row["tt"] = indices["tt"]
            row["sweat"] = indices["sweat"]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Gagal menghitung indeks atmosfer untuk %s: %s", row["date"], exc
            )

    return row



def get_sounding_rows_for_dates(
    dates: list[pd.Timestamp], use_cache: bool = True
) -> pd.DataFrame:
    """Bangun DataFrame sounding (satu baris per tanggal) untuk daftar
    tanggal kalender.

    Menggunakan ThreadPoolExecutor(max_workers=2) untuk mempercepat proses
    akuisisi secara paralel terbatas antar tanggal, namun tetap mempertahankan
    struktur pencarian sinkron per-tanggal. Session dibuat per worker task
    untuk menjaga thread-safety.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _fetch_worker(d: pd.Timestamp) -> dict:
        import threading
        thread_name = threading.current_thread().name
        # Reuse session for retries within the same task
        with requests.Session() as session:
            # Let's log brief output as required by setting settings.
            # (wyoming module logging output already prints CACHE HIT / CACHE MISS / SUCCESS)
            return get_sounding_row_for_date(d, session=session, use_cache=use_cache)

    # Parallel execution with max_workers=2
    with ThreadPoolExecutor(max_workers=2) as executor:
        # executor.map preserves the original input order of dates
        rows = list(executor.map(_fetch_worker, dates))

    return pd.DataFrame(rows)
