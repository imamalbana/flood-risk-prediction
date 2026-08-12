"""
Integrasi Stage 5: LEFT JOIN backbone_calendar -> Ogimet -> SounderPy, on
`date` (key kalender, BUKAN observation_datetime).

Identik urutan join dengan STAGE_DEPLOYMENT_MAP.md baris Stage 5: backbone
(kalender penuh) tidak pernah berkurang jumlah barisnya.
"""

from __future__ import annotations

import pandas as pd

from src.calendar_utils import build_master_calendar


def integrate(
    ogimet_df: pd.DataFrame,
    sounding_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Bangun dataset terintegrasi harian untuk rentang [start_date, end_date].

    backbone (kalender harian) LEFT JOIN ogimet_df (on date)
             LEFT JOIN sounding_df (on date)

    Parameters
    ----------
    ogimet_df : DataFrame dengan kolom date, rr, tavg, rh
    sounding_df : DataFrame dengan kolom date, selection_status,
        selected_hour, cin, kindex, li, tt, sweat
    """
    backbone = pd.DataFrame({
        "date": build_master_calendar(start_date, end_date),
    })

    ogimet_df = ogimet_df.copy()
    ogimet_df["date"] = pd.to_datetime(ogimet_df["date"]).dt.normalize()

    sounding_df = sounding_df.copy()
    sounding_df["date"] = pd.to_datetime(sounding_df["date"]).dt.normalize()

    merged = backbone.merge(ogimet_df, on="date", how="left")
    merged = merged.merge(sounding_df, on="date", how="left")

    if len(merged) != len(backbone):
        raise AssertionError(
            "Integrasi Stage 5 menghasilkan jumlah baris berbeda dari backbone "
            f"({len(merged)} != {len(backbone)}) -- kemungkinan duplicate date "
            "pada ogimet_df/sounding_df."
        )

    return merged
