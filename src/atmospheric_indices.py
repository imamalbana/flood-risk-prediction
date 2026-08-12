"""
FM35 -> SounderPy/SHARPpy Atmospheric Indices (inference wrapper).

Algoritma pada modul ini TIDAK diubah dari `atmospheric_indices.py` artefak
Stage (lihat INFERENCE_AUDIT_REPORT.md Bagian 1.7):
    - Tidak ada interpolasi level yang hilang.
    - Tidak ada penghapusan level (kecuali level dengan geopotentiaexitl
      height NaN, atau pressure duplikat -- identik logika sumber).
    - Tidak ada fabrikasi nilai observasi yang hilang.
    - CAPE/CIN/PWAT/dll: delegasi penuh ke sounderpy.calc.sounding_params.
    - LI/TT/K-Index/SWEAT: delegasi penuh ke fungsi resmi SHARPpy
      (params.k_index, params.t_totals, params.sweat, params.parcelx.li5),
      karena tidak diekspos oleh API resmi SounderPy.

Perbedaan dari artefak asli: modul ini dibungkus untuk kebutuhan inference
per-sounding (satu profile per pemanggilan, dipanggil dari
src/sounding.py), bukan pipeline batch load_dataset -> export_results per
CSV tahunan. Fungsi build_clean_data/validate_clean_data/
calculate_indices/calculate_sharppy_direct_indices dipertahankan identik.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from metpy.units import units

logger = logging.getLogger("inference_pipeline.atmospheric_indices")

SHARPPY_MISSING = -9999.0

STATION_INFO = {
    "site-id": "WIMG",
    "site-name": "PADANG/TABING",
    "site-lctn": "ID",
    "site-latlon": (-0.88, 100.35),
    "site-elv": 3.0,
}

RAW_STRING_NUMERIC_COLS = [
    "geopotential height_m",
    "dew point temperature_C",
    "ice point temperature_C",
    "relative humidity_%",
    "humidity wrt ice_%",
    "mixing ratio_g/kg",
    "wind direction_degree",
    "wind speed_m/s",
]


def clean_raw_sounding_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Identik dengan atmospheric_indices.py::load_dataset, bagian
    pembersihan kolom numerik string ber-whitespace (tanpa membaca file)."""
    df = df.copy()
    for col in RAW_STRING_NUMERIC_COLS:
        if col in df.columns and df[col].dtype != "float64":
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    return df


def build_clean_data(
    profile_df: pd.DataFrame,
    station_info: dict = STATION_INFO,
    missing_flag: float = SHARPPY_MISSING,
) -> dict:
    """Identik dengan atmospheric_indices.py::build_clean_data."""
    g = profile_df.sort_values("pressure_hPa", ascending=False).reset_index(drop=True)

    obs_dt_raw = g["observation_datetime"].iloc[0]

    pres = g["pressure_hPa"].to_numpy(dtype=float)
    hgt = pd.to_numeric(g["geopotential height_m"], errors="coerce").to_numpy(
        dtype=float
    )
    tmp = g["temperature_C"].to_numpy(dtype=float)
    dwp = g["dew point temperature_C"].to_numpy(dtype=float)
    wdir = g["wind direction_degree"].to_numpy(dtype=float)
    wspd_ms = g["wind speed_m/s"].to_numpy(dtype=float)

    hgt_nan_mask = np.isnan(hgt)
    n_hgt_dropped = int(hgt_nan_mask.sum())
    if n_hgt_dropped > 0:
        logger.warning(
            "Sounding %s: membuang %d level karena geopotential height_m NaN",
            obs_dt_raw,
            n_hgt_dropped,
        )
        keep_mask = ~hgt_nan_mask
        pres, hgt, tmp, dwp, wdir, wspd_ms = (
            arr[keep_mask] for arr in (pres, hgt, tmp, dwp, wdir, wspd_ms)
        )

    non_dups = np.concatenate(([True], np.diff(pres) != 0))
    pres, hgt, tmp, dwp, wdir, wspd_ms = (
        arr[non_dups] for arr in (pres, hgt, tmp, dwp, wdir, wspd_ms)
    )

    wspd_kt = wspd_ms * 1.94384
    wdir_rad = np.deg2rad(wdir)
    u = -wspd_kt * np.sin(wdir_rad)
    v = -wspd_kt * np.cos(wdir_rad)

    dwp_flagged = np.where(np.isnan(dwp), missing_flag, dwp)
    u_flagged = np.where(np.isnan(u), missing_flag, u)
    v_flagged = np.where(np.isnan(v), missing_flag, v)

    clean_data = {
        "p": pres * units.hPa,
        "z": hgt * units.meter,
        "T": tmp * units.degC,
        "Td": dwp_flagged * units.degC,
        "u": u_flagged * units.kt,
        "v": v_flagged * units.kt,
    }

    # --- site_info / titles ------------------------------------------------
    # Kontrak struktur `clean_data` resmi SounderPy (identik dengan
    # atmospheric_indices.py::build_clean_data pada artefak sumber) mewajibkan
    # key 'site_info' -- tanpanya sounderpy.calc.sounding_params(...).calc()
    # gagal dengan KeyError: 'site_info'. Wrapper inference ini sebelumnya
    # menghilangkan key tersebut; ditambahkan kembali di sini, bukan
    # workaround, karena merupakan bagian dari struktur sumber.
    obs_dt = pd.to_datetime(g["observation_datetime"].iloc[0])
    hour = str(int(g["observation_hour"].iloc[0])).zfill(2)

    clean_data["site_info"] = {
        "site-id": station_info["site-id"],
        "site-name": station_info["site-name"],
        "site-lctn": station_info["site-lctn"],
        "site-latlon": station_info["site-latlon"],
        "site-elv": station_info["site-elv"],
        "source": "RAOB OBSERVED PROFILE (FM35 CSV)",
        "model": "no-model",
        "fcst-hour": "no-fcst-hour",
        "run-time": ["none", "none", "none", "none"],
        "valid-time": [str(obs_dt.year), str(obs_dt.month), str(obs_dt.day), hour],
    }

    clean_data["titles"] = {
        "top_title": "FM35 OBSERVED VERTICAL PROFILE",
        "left_title": f"VALID: {obs_dt.month}-{obs_dt.day}-{obs_dt.year} {hour}Z",
        "right_title": f"{station_info['site-id']} - {station_info['site-name']}, "
        f"{station_info['site-lctn']} | {station_info['site-latlon'][0]}, "
        f"{station_info['site-latlon'][1]}",
    }

    return clean_data


def validate_clean_data(clean_data: dict) -> dict:
    """Identik dengan atmospheric_indices.py::validate_clean_data."""
    report = {"valid": True, "reasons": []}

    required = ["p", "z", "T", "Td", "u", "v"]
    for key in required:
        if key not in clean_data:
            report["valid"] = False
            report["reasons"].append(f"missing required key '{key}'")

    if not report["valid"]:
        return report

    lengths = {k: len(clean_data[k].m) for k in required}
    report["lengths"] = lengths
    if len(set(lengths.values())) != 1:
        report["valid"] = False
        report["reasons"].append(f"array length mismatch: {lengths}")

    n_levels = lengths.get("p")
    report["n_levels"] = n_levels

    report["n_nan_unflagged"] = {
        k: int(np.sum(np.isnan(clean_data[k].m))) for k in required
    }
    if any(v > 0 for v in report["n_nan_unflagged"].values()):
        report["valid"] = False
        report["reasons"].append(f"unflagged NaN present: {report['n_nan_unflagged']}")

    if n_levels is not None and n_levels < 2:
        report["valid"] = False
        report["reasons"].append(
            "fewer than 2 levels -- SHARPpy Profile requires length > 1"
        )

    return report


def calculate_indices(clean_data: dict) -> dict:
    """Identik dengan atmospheric_indices.py::calculate_indices (delegasi
    penuh ke sounderpy.calc.sounding_params)."""
    from sounderpy.calc import sounding_params

    result = {"success": False, "error": None, "general": {}, "thermo": {}, "kinem": {}}
    cd_for_calc = {k: v for k, v in clean_data.items() if k != "_meta"}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            general, thermo, kinem, intrp = sounding_params(cd_for_calc).calc()
        result["success"] = True
        result["general"] = general
        result["thermo"] = thermo
        result["kinem"] = kinem
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def calculate_sharppy_direct_indices(clean_data: dict) -> dict:
    """Identik dengan atmospheric_indices.py::calculate_sharppy_direct_indices
    (LI, TT, K-Index, SWEAT via pemanggilan langsung fungsi resmi SHARPpy)."""
    import metpy.calc as mpcalc
    from sounderpy.SHARPPYMAIN.sharppy.sharptab.profile import create_profile
    from sounderpy.SHARPPYMAIN.sharppy.sharptab.params import (
        k_index,
        t_totals,
        sweat,
        parcelx,
    )

    result = {"success": False, "error": None, "values": {}, "source": {}}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            T = clean_data["T"]
            Td = clean_data["Td"]
            p = clean_data["p"]
            z = clean_data["z"]
            u = clean_data["u"]
            v = clean_data["v"]
            wd = mpcalc.wind_direction(u, v)
            ws = mpcalc.wind_speed(u, v)

            prof = create_profile(
                profile="default",
                pres=p.m,
                hght=z.m,
                tmpc=T.m,
                dwpc=Td.m,
                wspd=ws,
                wdir=wd,
                missing=-9999,
                strictQC=False,
            )

            result["values"]["k_index"] = k_index(prof)
            result["values"]["t_totals"] = t_totals(prof)
            result["source"]["k_index"] = "sharppy.sharptab.params.k_index(prof)"
            result["source"]["t_totals"] = "sharppy.sharptab.params.t_totals(prof)"

            result["values"]["sweat"] = sweat(prof)
            result["source"]["sweat"] = "sharppy.sharptab.params.sweat(prof)"

            sbpcl = parcelx(prof, flag=1, pres=p[0].m, tmpc=T[0].m, dwpc=Td[0].m)
            result["values"]["li_sb_500"] = sbpcl.li5
            result["source"][
                "li_sb_500"
            ] = "sharppy.sharptab.params.parcelx(prof, flag=1, ...).li5"

            result["success"] = True
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _scalar(v):
    """Identik dengan atmospheric_indices.py::_scalar."""
    try:
        import numpy.ma as ma

        if v is ma.masked:
            return None
    except Exception:
        pass
    if hasattr(v, "m"):
        v = v.m
    if isinstance(v, np.ndarray):
        if v.size == 1:
            v = v.item()
        else:
            return None
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
    except Exception:
        pass
    return v


def compute_indices_for_sounding(profile_df: pd.DataFrame) -> dict:
    """Wrapper inference: hitung 5 fitur atmosfer model final (cin, kindex,
    li, tt, sweat) untuk satu profile sounding mentah (satu tanggal/jam
    terpilih), tanpa menulis CSV.

    Mapping (identik INFERENCE_CONTRACT.md Bagian 7 & audit Bagian 1.7):
        cin    <- SBCIN
        kindex <- KINDEX
        li     <- LI_SB_500
        tt     <- t_totals
        sweat  <- sweat

    Return dict berisi kelima key di atas (nilai None jika gagal dihitung
    -- caller (src/sounding.py / src/integration.py) yang memutuskan
    dampaknya terhadap status baris, BUKAN modul ini yang mengarang nilai
    pengganti).
    """
    profile_df = clean_raw_sounding_columns(profile_df)
    clean_data = build_clean_data(profile_df)
    validation = validate_clean_data(clean_data)

    out = {
        "cin": None,
        "kindex": None,
        "li": None,
        "tt": None,
        "sweat": None,
        "status": None,
        "reason": "",
    }

    if not validation["valid"]:
        out["status"] = "FAILED"
        out["reason"] = "; ".join(validation["reasons"])
        return out

    calc_result = calculate_indices(clean_data)
    if calc_result["success"]:
        out["cin"] = _scalar(calc_result.get("thermo", {}).get("sbcin"))
        out["status"] = "SUCCESS"
    else:
        out["status"] = "FAILED"
        out["reason"] = calc_result.get("error", "")

    sharppy_direct = calculate_sharppy_direct_indices(clean_data)
    sd_values = sharppy_direct.get("values", {}) or {}
    out["kindex"] = _scalar(sd_values.get("k_index"))
    out["tt"] = _scalar(sd_values.get("t_totals"))
    out["sweat"] = _scalar(sd_values.get("sweat"))
    out["li"] = _scalar(sd_values.get("li_sb_500"))
    if sharppy_direct.get("error") and not out["reason"]:
        out["reason"] = sharppy_direct["error"]

    return out
