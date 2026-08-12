"""
Akuisisi sounding Wyoming Upper Air per tanggal, dengan prioritas jam dan
source identik dengan `wyouming_downloader.py::process_date`.

Urutan (WAJIB, lihat INFERENCE_CONTRACT.md Bagian 7 & Bagian 1.5
INFERENCE_AUDIT_REPORT.md):
    12Z (src=FM35) -> 12Z (default) -> 00Z (src=FM35) -> 00Z (default)
    -> NO_SOUNDING

Diparameterkan per-tanggal (dipanggil dari src/sounding.py per tanggal
window) agar dapat dipakai untuk rentang tanggal sembarang, bukan satu
YEAR hardcoded seperti skrip asli (lihat catatan audit Bagian 1.6).

CATATAN OPEN DECISION 2 (INFERENCE_CONTRACT.md Bagian 10): desain modul
akuisisi ini mereplikasi persis alur per-tanggal `process_date()` milik
downloader asli (coba 12Z lebih dulu, fallback 00Z, satu hasil per
tanggal) -- opsi paling konservatif yang konsisten dengan asumsi Stage 4
bahwa tidak pernah ada kombinasi jam ganda per tanggal.
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests

from config.settings import CACHE_DIR, SRC_ORDER, STATION_ID

logger = logging.getLogger("inference_pipeline.wyoming")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

MAX_RETRIES = 5
BACKOFF_BASE = 30.0
RETRY_BACKOFF_BASE = 5.0
TIMEOUT = 30
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class SoundingResult:
    date: str
    selected_hour: Optional[int]
    status: str  # "SUCCESS_12Z" | "SUCCESS_00Z" | "NO_SOUNDING"
    source: Optional[str]
    df: Optional[pd.DataFrame]
    message: str


def _build_url(date_str: str, hour_str: str, src: Optional[str]) -> str:
    src_param = f"&src={src}" if src else ""
    return (
        "https://weather.uwyo.edu/wsgi/sounding"
        f"?datetime={date_str}%20{hour_str}:00:00&id={STATION_ID}"
        f"&type=TEXT:CSV{src_param}"
    )


def _parse_response_text(text: str, src: Optional[str]):
    """Identik dengan wyouming_downloader.py::_parse_response_text."""
    if "Unable to retrieve" in text or "No sounding available" in text:
        return "MISSING", None, f"No sounding available in body (src={src})"

    if (
        "<html" in text.lower()
        or "<body" in text.lower()
        or not text.strip().startswith("time")
    ):
        return "FAILED", None, f"Invalid format (src={src})"

    try:
        df = pd.read_csv(io.StringIO(text.strip()))
    except Exception as exc:  # noqa: BLE001
        return "FAILED", None, f"CSV Parse Error: {exc} (src={src})"

    if df.empty:
        return "MISSING", None, f"Empty CSV (src={src})"

    return "SUCCESS", df, f"Success (src={src})"


def _compute_backoff(attempt: int, base: float) -> float:
    return base * (2 ** (attempt - 1))


def download_single_sounding(
    date_str: str, hour_str: str, src: Optional[str], session: requests.Session
):
    """Identik dengan wyouming_downloader.py::download_single_sounding."""
    url = _build_url(date_str, hour_str, src)

    for attempt in range(1, MAX_RETRIES + 1):
        t_req_start = time.perf_counter()
        try:
            response = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            status_code = response.status_code
            elapsed_req = time.perf_counter() - t_req_start

            if status_code == 404:
                print(f"[WYOMING DEBUG] {date_str} | {hour_str}Z | {src} | attempt={attempt} | HTTP 404 | {elapsed_req:.2f}s")
                return "MISSING", None, f"HTTP 404 (src={src})"

            if status_code == 429:
                print(f"[WYOMING DEBUG] {date_str} | {hour_str}Z | {src} | attempt={attempt} | HTTP 429 | {elapsed_req:.2f}s")
                if attempt == MAX_RETRIES:
                    return "FAILED", None, f"HTTP 429 after max retries (src={src})"
                backoff_time = _compute_backoff(attempt, BACKOFF_BASE)
                print(f"[WYOMING DEBUG] Backoff for {backoff_time}s")
                time.sleep(backoff_time)
                continue

            if status_code in RETRYABLE_STATUS_CODES:
                print(f"[WYOMING DEBUG] {date_str} | {hour_str}Z | {src} | attempt={attempt} | HTTP {status_code} | {elapsed_req:.2f}s")
                if attempt == MAX_RETRIES:
                    return "FAILED", None, f"HTTP {status_code} after max retries (src={src})"
                backoff_time = _compute_backoff(attempt, RETRY_BACKOFF_BASE)
                print(f"[WYOMING DEBUG] Backoff for {backoff_time}s")
                time.sleep(backoff_time)
                continue

            if status_code != 200:
                print(f"[WYOMING DEBUG] {date_str} | {hour_str}Z | {src} | attempt={attempt} | HTTP {status_code} | {elapsed_req:.2f}s")
                return "FAILED", None, f"HTTP {status_code} (src={src})"

            outcome, df, message = _parse_response_text(response.text, src)
            print(f"[WYOMING DEBUG] {date_str} | {hour_str}Z | {src} | attempt={attempt} | outcome={outcome} ({message}) | {elapsed_req:.2f}s")
            if outcome == "FAILED" and attempt < MAX_RETRIES:
                backoff_time = _compute_backoff(attempt, RETRY_BACKOFF_BASE)
                print(f"[WYOMING DEBUG] Backoff for {backoff_time}s")
                time.sleep(backoff_time)
                continue
            return outcome, df, message

        except requests.exceptions.Timeout as exc:
            elapsed_req = time.perf_counter() - t_req_start
            print(f"[WYOMING DEBUG] {date_str} | {hour_str}Z | {src} | attempt={attempt} | TIMEOUT | {elapsed_req:.2f}s")
            if attempt == MAX_RETRIES:
                return "FAILED", None, f"Timeout: {exc} (src={src})"
            backoff_time = _compute_backoff(attempt, RETRY_BACKOFF_BASE)
            time.sleep(backoff_time)
        except requests.exceptions.ConnectionError as exc:
            elapsed_req = time.perf_counter() - t_req_start
            print(f"[WYOMING DEBUG] {date_str} | {hour_str}Z | {src} | attempt={attempt} | CONNECTION_ERROR | {elapsed_req:.2f}s")
            if attempt == MAX_RETRIES:
                return "FAILED", None, f"Connection Error: {exc} (src={src})"
            backoff_time = _compute_backoff(attempt, RETRY_BACKOFF_BASE)
            time.sleep(backoff_time)
        except requests.exceptions.RequestException as exc:
            elapsed_req = time.perf_counter() - t_req_start
            print(f"[WYOMING DEBUG] {date_str} | {hour_str}Z | {src} | attempt={attempt} | REQUEST_ERROR | {elapsed_req:.2f}s")
            if attempt == MAX_RETRIES:
                return "FAILED", None, f"Request Error: {exc} (src={src})"
            backoff_time = _compute_backoff(attempt, RETRY_BACKOFF_BASE)
            time.sleep(backoff_time)

    return "FAILED", None, f"Failed all attempts (src={src})"


def _try_hour(date_str: str, hour_str: str, hour_int: int, status_label: str, session):
    messages = []
    for src in SRC_ORDER:
        t_start = time.perf_counter()
        outcome, df, message = download_single_sounding(date_str, hour_str, src, session)
        elapsed = time.perf_counter() - t_start
        messages.append(f"{hour_str}Z: {message}")
        if outcome == "SUCCESS":
            df = df.copy()
            df.insert(0, "request_date", date_str)
            df["observation_datetime"] = df["time"]
            df["observation_hour"] = hour_int
            print(f"[WYOMING] {date_str} | {hour_str}Z | {src} | CACHE MISS | {elapsed:.2f}s | SUCCESS")
            return (
                SoundingResult(
                    date=date_str,
                    selected_hour=hour_int,
                    status=status_label,
                    source="default" if src is None else src,
                    df=df,
                    message=" | ".join(messages),
                ),
                messages,
            )
        else:
            print(f"[WYOMING] {date_str} | {hour_str}Z | {src} | CACHE MISS | {elapsed:.2f}s | FAILED/MISSING")
    return None, messages


def _hour_cache_paths(date_str: str, hour_str: str) -> tuple[str, str]:
    """Path cache on-disk untuk satu kombinasi (request_date, hour)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = f"wyoming_{date_str}_{hour_str}Z"
    return (
        os.path.join(CACHE_DIR, f"{base}.csv"),
        os.path.join(CACHE_DIR, f"{base}.meta"),
    )


def _try_hour_cached(
    date_str: str,
    hour_str: str,
    hour_int: int,
    status_label: str,
    session,
    use_cache: bool = True,
):
    """Sama seperti _try_hour, tapi dengan cache on-disk per (date_str, hour_str)."""
    cache_data_path, cache_meta_path = _hour_cache_paths(date_str, hour_str)

    if use_cache and os.path.exists(cache_meta_path):
        t_start = time.perf_counter()
        with open(cache_meta_path, "r", encoding="utf-8") as f:
            meta_line = f.read().strip()
        status, selected_hour_str, source = (meta_line.split("|") + [None, None, None])[:3]
        elapsed = time.perf_counter() - t_start
        if status == "NO_HOUR":
            print(f"[WYOMING] {date_str} | {hour_str}Z | None | CACHE HIT (NO_HOUR) | {elapsed:.4f}s | FAILED")
            return None, [f"{hour_str}Z: (dari cache, tidak tersedia)"]
        selected_hour = int(selected_hour_str) if selected_hour_str not in (None, "", "None") else None
        df = pd.read_csv(cache_data_path) if os.path.exists(cache_data_path) else None
        print(f"[WYOMING] {date_str} | {hour_str}Z | {source} | CACHE HIT | {elapsed:.4f}s | SUCCESS")
        return (
            SoundingResult(
                date=date_str,
                selected_hour=selected_hour,
                status=status,
                source=source if source not in (None, "", "None") else None,
                df=df,
                message="(dari cache)",
            ),
            [f"{hour_str}Z: (dari cache)"],
        )

    result, messages = _try_hour(date_str, hour_str, hour_int, status_label, session)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if result is not None:
            if result.df is not None:
                result.df.to_csv(cache_data_path, index=False)
            with open(cache_meta_path, "w", encoding="utf-8") as f:
                f.write(f"{result.status}|{result.selected_hour}|{result.source}")
        else:
            with open(cache_meta_path, "w", encoding="utf-8") as f:
                f.write("NO_HOUR||")

    return result, messages


def process_date(date_str: str, session: requests.Session) -> SoundingResult:
    """Identik dengan wyouming_downloader.py::process_date, ditambah status
    akhir `NO_SOUNDING` (bukan `MISSING`/`FAILED`) untuk konsistensi
    penamaan dengan Stage 4/5/7 (`selection_status`)."""
    all_messages = []

    result_12z, msgs_12z = _try_hour(date_str, "12", 12, "SUCCESS_12Z", session)
    all_messages.extend(msgs_12z)
    if result_12z is not None:
        return result_12z

    result_00z, msgs_00z = _try_hour(date_str, "00", 0, "SUCCESS_00Z", session)
    all_messages.extend(msgs_00z)
    if result_00z is not None:
        return result_00z

    return SoundingResult(
        date=date_str,
        selected_hour=None,
        status="NO_SOUNDING",
        source=None,
        df=None,
        message=" | ".join(all_messages),
    )


def get_sounding_for_date(
    date: pd.Timestamp, use_cache: bool = True, session: Optional[requests.Session] = None
) -> SoundingResult:
    """Ambil (atau baca dari cache) satu sounding untuk tanggal tertentu.

    Cache disimpan sebagai CSV mentah per tanggal di CACHE_DIR
    (`wyoming_{date}.csv`) plus metadata status di `wyoming_{date}.meta`.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    cache_data_path = os.path.join(CACHE_DIR, f"wyoming_{date_str}.csv")
    cache_meta_path = os.path.join(CACHE_DIR, f"wyoming_{date_str}.meta")

    if use_cache and os.path.exists(cache_meta_path):
        with open(cache_meta_path, "r", encoding="utf-8") as f:
            meta_line = f.read().strip()
        status, selected_hour_str, source = (meta_line.split("|") + [None, None, None])[:3]
        selected_hour = int(selected_hour_str) if selected_hour_str not in (None, "", "None") else None
        df = pd.read_csv(cache_data_path) if os.path.exists(cache_data_path) else None
        return SoundingResult(
            date=date_str,
            selected_hour=selected_hour,
            status=status,
            source=source if source not in (None, "", "None") else None,
            df=df,
            message="(dari cache)",
        )

    own_session = session is None
    if own_session:
        session = requests.Session()
    try:
        result = process_date(date_str, session)
    finally:
        if own_session:
            session.close()

    if use_cache:
        if result.df is not None:
            result.df.to_csv(cache_data_path, index=False)
        with open(cache_meta_path, "w", encoding="utf-8") as f:
            f.write(f"{result.status}|{result.selected_hour}|{result.source}")

    return result
