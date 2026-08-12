"""
Replikasi konsep Stage 3 (master calendar & date alignment).

Sumber: STAGE_DEPLOYMENT_MAP.md baris Stage 3 -- "pd.date_range(freq='D')",
key kalender adalah `date` (bukan `observation_datetime`).

Nama file sengaja `calendar_utils.py`, BUKAN `calendar.py`, agar tidak
membayangi modul stdlib Python `calendar`.
"""

from __future__ import annotations

import datetime as _dt
from typing import Iterable, List

import pandas as pd


def parse_target_date(target_date) -> pd.Timestamp:
    """Parse tanggal target ke pandas.Timestamp (tanpa komponen jam).

    Menerima str ('YYYY-MM-DD'), datetime.date, datetime.datetime, atau
    pandas.Timestamp. Melempar ValueError jika tidak dapat diparse --
    validation gate (src/pipeline.py) akan menangkap ini sebagai kegagalan
    tanggal target tidak valid.
    """
    if isinstance(target_date, pd.Timestamp):
        return target_date.normalize()
    if isinstance(target_date, _dt.datetime):
        return pd.Timestamp(target_date).normalize()
    if isinstance(target_date, _dt.date):
        return pd.Timestamp(target_date)
    if isinstance(target_date, str):
        try:
            return pd.Timestamp(target_date).normalize()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Tanggal target tidak dapat diparse: {target_date!r}") from exc
    raise ValueError(
        f"Tipe tanggal target tidak didukung: {type(target_date)!r}"
    )


def build_window_dates(target_date: pd.Timestamp, lookback: int) -> List[pd.Timestamp]:
    """Bangun daftar tanggal D-lookback ... D-1 (ascending), tanggal D
    (target) TIDAK termasuk -- sesuai INFERENCE_CONTRACT.md Bagian 3."""
    target_date = pd.Timestamp(target_date).normalize()
    return [target_date - pd.Timedelta(days=n) for n in range(lookback, 0, -1)]


def build_master_calendar(
    start_date, end_date, freq: str = "D"
) -> pd.DatetimeIndex:
    """Master calendar harian penuh, konsep identik dengan Stage 3
    (`pd.date_range(freq='D')`), dipakai sebagai backbone LEFT JOIN
    (Stage 5) dan sebagai basis validasi kontinuitas window inference."""
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def validate_window_continuity(dates: Iterable[pd.Timestamp]) -> dict:
    """Validasi bahwa daftar tanggal:
    - tidak duplikat
    - ascending
    - berurutan harian tanpa celah

    Mengembalikan dict laporan (bukan raise) agar caller (validation gate
    di src/pipeline.py) dapat mengumpulkan seluruh pelanggaran sekaligus,
    bukan berhenti di pelanggaran pertama.
    """
    dates = [pd.Timestamp(d).normalize() for d in dates]
    report = {"valid": True, "reasons": []}

    if len(dates) != len(set(dates)):
        report["valid"] = False
        report["reasons"].append("terdapat tanggal duplikat pada window")

    if dates != sorted(dates):
        report["valid"] = False
        report["reasons"].append("urutan tanggal tidak ascending")
    else:
        for prev, curr in zip(dates, dates[1:]):
            if (curr - prev).days != 1:
                report["valid"] = False
                report["reasons"].append(
                    f"celah tanggal: {prev.date()} -> {curr.date()} "
                    f"(selisih {(curr - prev).days} hari, seharusnya 1)"
                )

    report["n_dates"] = len(dates)
    return report
