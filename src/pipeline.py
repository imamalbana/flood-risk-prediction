"""
Orkestrasi inference end-to-end: `predict_for_date(target_date)`.

Alur (INFERENCE_CONTRACT.md Bagian 7, WAJIB urutan ini):

    target
    -> acquisition/cache (Ogimet + Wyoming/SounderPy untuk window D-30..D-1)
    -> standardisasi (placeholder Stage 2, dilakukan di dalam src/ogimet.py)
    -> date alignment (master calendar, src/calendar_utils.py)
    -> seleksi sounding harian per tanggal (src/sounding.py)
    -> hitung indeks atmosfer (di dalam src/sounding.py, delegasi ke
       src/atmospheric_indices.py)
    -> integrasi (src/integration.py, LEFT JOIN)
    -> Stage 7 preprocessing (src/preprocessing.py)
    -> validation gate (WAJIB lulus sebelum predict, lihat di bawah)
    -> ambil 8 fitur (src/sequence.py)
    -> scaler.transform() (src/sequence.py, TIDAK fit)
    -> reshape (1, 30, 8) (src/sequence.py)
    -> model.predict() (src/predictor.py) -> argmax -> class mapping

VALIDATION GATE (INFERENCE_CONTRACT.md Bagian 8, WAJIB LULUS SEBELUM
model.predict()). Jika gagal: JANGAN PREDIKSI, kembalikan alasan spesifik.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import FEATURE_COLUMNS, LOOKBACK
from src.calendar_utils import build_window_dates, parse_target_date, validate_window_continuity
from src.integration import integrate
from src.ogimet import get_ogimet_daily
from src.predictor import load_model, load_scaler, predict as run_predict
from src.preprocessing import apply_stage7_missing_value_handling, load_monthly_medians
from src.sequence import build_input_sequence, extract_feature_matrix, scale_features
from src.sounding import get_sounding_rows_for_dates

logger = logging.getLogger("inference_pipeline.pipeline")


class ValidationGateError(Exception):
    """Dilempar saat validation gate gagal. Prediksi TIDAK dijalankan."""

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("Validation gate gagal:\n- " + "\n- ".join(reasons))


def _run_validation_gate(
    window_dates: list[pd.Timestamp], feature_matrix: pd.DataFrame, scaler
) -> list[str]:
    """Jalankan seluruh 9 pemeriksaan INFERENCE_CONTRACT.md Bagian 8.
    Mengembalikan daftar alasan kegagalan (list kosong = lulus)."""
    reasons: list[str] = []

    # [ ] seluruh 30 tanggal D-30..D-1 tersedia (sudah dijamin oleh
    #     extract_feature_matrix yang melempar KeyError -- ditangani di
    #     caller sebelum fungsi ini dipanggil), di sini kita cek ulang
    #     jumlah baris.
    if len(feature_matrix) != LOOKBACK:
        reasons.append(
            f"Jumlah tanggal tersedia ({len(feature_matrix)}) != LOOKBACK ({LOOKBACK})"
        )

    # [ ] tidak ada duplicate date, [ ] ascending tanpa celah
    continuity = validate_window_continuity(window_dates)
    if not continuity["valid"]:
        reasons.extend(continuity["reasons"])

    # [ ] 8 fitur tersedia untuk setiap tanggal
    missing_cols = [c for c in FEATURE_COLUMNS if c not in feature_matrix.columns]
    if missing_cols:
        reasons.append(f"Kolom fitur tidak tersedia: {missing_cols}")
        return reasons  # pemeriksaan berikutnya butuh seluruh kolom ada

    # [ ] 8 fitur bertipe numerik
    non_numeric = [
        c for c in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(feature_matrix[c])
    ]
    if non_numeric:
        reasons.append(f"Kolom fitur bukan numerik: {non_numeric}")

    # [ ] tidak ada NaN pada array (30, 8) setelah seluruh preprocessing
    nan_mask = feature_matrix[FEATURE_COLUMNS].isna()
    if nan_mask.to_numpy().any():
        nan_dates = feature_matrix.index[nan_mask.any(axis=1)]
        nan_cols_per_date = {
            str(d): [c for c in FEATURE_COLUMNS if pd.isna(feature_matrix.loc[d, c])]
            for d in nan_dates
        }
        reasons.append(
            f"Terdapat NaN pada fitur setelah preprocessing (kemungkinan "
            f"NO_SOUNDING di dalam window): {nan_cols_per_date}"
        )

    # [ ] scaler memiliki n_features_in_ == 8
    n_features_in = getattr(scaler, "n_features_in_", None)
    if n_features_in != len(FEATURE_COLUMNS):
        reasons.append(
            f"scaler.n_features_in_ ({n_features_in}) != {len(FEATURE_COLUMNS)}"
        )

    # [ ] urutan fitur yang dikirim ke scaler == feature_names_in_ scaler
    expected_order = list(getattr(scaler, "feature_names_in_", []))
    if expected_order and list(feature_matrix.columns) != expected_order:
        reasons.append(
            f"Urutan fitur ({list(feature_matrix.columns)}) != "
            f"scaler.feature_names_in_ ({expected_order})"
        )

    return reasons


def predict_for_date(
    target_date,
    model=None,
    scaler=None,
    monthly_medians: Optional[dict] = None,
    integration_start_buffer_days: int = 0,
    use_cache: bool = True,
    verbose: bool = False,
) -> dict:
    """Jalankan inference lengkap untuk satu tanggal target.

    Parameters
    ----------
    target_date : str | date | datetime | Timestamp
    model, scaler : sudah dimuat (opsional -- dimuat otomatis jika None,
        tapi memuat sekali di luar lalu mengoper ke sini lebih efisien
        untuk pemanggilan berulang).
    monthly_medians : dict median bulanan Stage 7 (opsional, dimuat
        otomatis dari config/stage7_monthly_medians.json jika None).
    integration_start_buffer_days : jumlah hari tambahan SEBELUM D-30 yang
        ikut diambil untuk keperluan interpolasi Ogimet. Default 0 (tidak
        ada buffer) karena ukuran buffer yang "benar" adalah OPEN DECISION 1
        yang BELUM diputuskan (lihat INFERENCE_CONTRACT.md Bagian 10) --
        modul ini tidak mengarang nilai default yang belum divalidasi.
        Caller yang ingin mengeksplorasi buffer dapat mengoper nilai > 0
        secara eksplisit dan sadar risikonya.
    verbose : bool
        Jika True, tampilkan output debug / verbose intermediate values
        dan simpan ke CSV.

    Returns
    -------
    dict dengan salah satu bentuk:
        - sukses: {"status": "SUCCESS", "target_date": ..., "predicted_class_index": ...,
                    "predicted_class_name": ..., "probabilities": {...}}
        - gagal validasi: {"status": "REJECTED", "target_date": ..., "reasons": [...]}
    Tidak pernah memanggil model.predict() jika status akan REJECTED.
    """
    total_start = time.perf_counter()
    try:
        target_ts = parse_target_date(target_date)
    except ValueError as exc:
        return {"status": "REJECTED", "target_date": str(target_date), "reasons": [str(exc)]}

    window_dates = build_window_dates(target_ts, LOOKBACK)
    fetch_start = window_dates[0] - pd.Timedelta(days=integration_start_buffer_days)
    fetch_end = window_dates[-1]

    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()
    if monthly_medians is None:
        monthly_medians = load_monthly_medians()

    # [1/7] Get Ogimet data
    t0 = time.perf_counter()
    try:
        ogimet_df = get_ogimet_daily(fetch_start, fetch_end, use_cache=use_cache)
        elapsed = time.perf_counter() - t0
        print(f"[1/7] Get Ogimet data ................. OK ({elapsed:.2f}s)")
        print(f"      rows={len(ogimet_df)}")
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        print(f"[1/7] Get Ogimet data ................. FAILED ({elapsed:.2f}s)")
        print(f"      {type(exc).__name__}: {exc}")
        print("\nPipeline stopped at: Get Ogimet data")
        total_elapsed = time.perf_counter() - total_start
        print(f"\nTotal runtime: {total_elapsed:.2f}s")
        return {
            "status": "REJECTED",
            "target_date": target_ts.strftime("%Y-%m-%d"),
            "reasons": [f"Gagal akuisisi data Ogimet: {exc}"],
        }

    # [2/7] Get Wyoming sounding
    t0 = time.perf_counter()
    try:
        all_dates = pd.date_range(fetch_start, fetch_end, freq="D")
        sounding_df = get_sounding_rows_for_dates(list(all_dates), use_cache=use_cache)
        elapsed = time.perf_counter() - t0
        print(f"[2/7] Get Wyoming sounding ............ OK ({elapsed:.2f}s)")
        print(f"      profiles={len(sounding_df)}")
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        print(f"[2/7] Get Wyoming sounding ............ FAILED ({elapsed:.2f}s)")
        print(f"      {type(exc).__name__}: {exc}")
        print("\nPipeline stopped at: Get Wyoming sounding")
        total_elapsed = time.perf_counter() - total_start
        print(f"\nTotal runtime: {total_elapsed:.2f}s")
        return {
            "status": "REJECTED",
            "target_date": target_ts.strftime("%Y-%m-%d"),
            "reasons": [f"Gagal akuisisi data sounding: {exc}"],
        }

    # [3/7] Atmospheric indices
    t0 = time.perf_counter()
    try:
        selected_count = int((sounding_df["selection_status"] == "SELECTED").sum())
        no_sounding_count = int((sounding_df["selection_status"] == "NO_SOUNDING").sum())
        elapsed = time.perf_counter() - t0
        print(f"[3/7] Atmospheric indices ............. OK ({elapsed:.2f}s)")
        print(f"      selected={selected_count}")
        print(f"      no_sounding={no_sounding_count}")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"[3/7] Atmospheric indices ............. FAILED ({elapsed:.2f}s)")
        print(f"      {type(exc).__name__}: {exc}")
        print("\nPipeline stopped at: Atmospheric indices")
        total_elapsed = time.perf_counter() - total_start
        print(f"\nTotal runtime: {total_elapsed:.2f}s")
        raise exc

    # [4/7] Data integration
    t0 = time.perf_counter()
    try:
        integrated = integrate(ogimet_df, sounding_df, fetch_start, fetch_end)
        elapsed = time.perf_counter() - t0
        print(f"[4/7] Data integration ................ OK ({elapsed:.2f}s)")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"[4/7] Data integration ................ FAILED ({elapsed:.2f}s)")
        print(f"      {type(exc).__name__}: {exc}")
        print("\nPipeline stopped at: Data integration")
        total_elapsed = time.perf_counter() - total_start
        print(f"\nTotal runtime: {total_elapsed:.2f}s")
        raise exc

    # [5/7] Preprocessing
    t0 = time.perf_counter()
    try:
        preprocessed = apply_stage7_missing_value_handling(integrated, monthly_medians)
        elapsed = time.perf_counter() - t0
        print(f"[5/7] Preprocessing ................... OK ({elapsed:.2f}s)")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"[5/7] Preprocessing ................... FAILED ({elapsed:.2f}s)")
        print(f"      {type(exc).__name__}: {exc}")
        print("\nPipeline stopped at: Preprocessing")
        total_elapsed = time.perf_counter() - total_start
        print(f"\nTotal runtime: {total_elapsed:.2f}s")
        raise exc

    # [6/7] LB30 sequence
    t0 = time.perf_counter()
    try:
        feature_matrix = extract_feature_matrix(preprocessed, window_dates)
        reasons = _run_validation_gate(window_dates, feature_matrix, scaler)
        if reasons:
            raise ValidationGateError(reasons)
        scaled = scale_features(feature_matrix, scaler)
        X_input = build_input_sequence(scaled, LOOKBACK)
        elapsed = time.perf_counter() - t0
        print(f"[6/7] LB30 sequence ................... OK ({elapsed:.2f}s)")
        print(f"      shape={X_input.shape}")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"[6/7] LB30 sequence ................... FAILED ({elapsed:.2f}s)")
        print(f"      {type(exc).__name__}: {exc}")
        print("\nPipeline stopped at: LB30 sequence")
        total_elapsed = time.perf_counter() - total_start
        print(f"\nTotal runtime: {total_elapsed:.2f}s")
        if isinstance(exc, ValidationGateError):
            return {
                "status": "REJECTED",
                "target_date": target_ts.strftime("%Y-%m-%d"),
                "reasons": exc.reasons,
            }
        elif isinstance(exc, KeyError):
            return {
                "status": "REJECTED",
                "target_date": target_ts.strftime("%Y-%m-%d"),
                "reasons": [str(exc)],
            }
        raise exc

    # [7/7] Model inference
    t0 = time.perf_counter()
    try:
        result = run_predict(model, X_input)
        result["status"] = "SUCCESS"
        result["target_date"] = target_ts.strftime("%Y-%m-%d")
        elapsed = time.perf_counter() - t0
        print(f"[7/7] Model inference ................ OK ({elapsed:.2f}s)")
        print("\nInference SUCCESS")
        print(f"Prediction: {result['predicted_class_name']}")
        
        total_elapsed = time.perf_counter() - total_start
        print(f"\nTotal runtime: {total_elapsed:.2f}s")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"[7/7] Model inference ................ FAILED ({elapsed:.2f}s)")
        print(f"      {type(exc).__name__}: {exc}")
        print("\nPipeline stopped at: Model inference")
        total_elapsed = time.perf_counter() - total_start
        print(f"\nTotal runtime: {total_elapsed:.2f}s")
        raise exc

    if verbose:
        window_start = window_dates[0].strftime("%Y-%m-%d")
        window_end = window_dates[-1].strftime("%Y-%m-%d")
        
        print("============================================================")
        print("INFERENCE DEBUG")
        print("============================================================")
        print(f"Target date : {target_ts.strftime('%Y-%m-%d')}")
        print(f"Lookback    : {LOOKBACK} days")
        print(f"Window      : {window_start} -> {window_end}")
        print("============================================================\n")
        
        print("ATMOSPHERIC / MODEL FEATURES\n")
        print("Date        Status      RR      Tavg    RH      CIN     KINDEX   LI      TT      SWEAT")
        print("-----------------------------------------------------------------------------------------")
        
        prep_indexed = preprocessed.set_index("date")
        prep_indexed.index = pd.to_datetime(prep_indexed.index)
        
        csv_rows = []
        for d in window_dates:
            d_ts = pd.Timestamp(d)
            row_data = prep_indexed.loc[d_ts]
            
            status = str(row_data.get("selection_status", "UNKNOWN"))
            selected_hour = row_data.get("selected_hour", None)
            
            rr = row_data.get("rr", None)
            tavg = row_data.get("tavg", None)
            rh = row_data.get("rh", None)
            cin = row_data.get("cin", None)
            kindex = row_data.get("kindex", None)
            li = row_data.get("li", None)
            tt = row_data.get("tt", None)
            sweat = row_data.get("sweat", None)
            
            def fmt(val):
                if val is None or pd.isna(val):
                    return "NaN"
                if isinstance(val, (int, float, np.number)):
                    return f"{val:.2f}"
                return str(val)

            print(f"{d_ts.strftime('%Y-%m-%d'):<12}"
                  f"{status:<12}"
                  f"{fmt(rr):<8}"
                  f"{fmt(tavg):<8}"
                  f"{fmt(rh):<8}"
                  f"{fmt(cin):<8}"
                  f"{fmt(kindex):<9}"
                  f"{fmt(li):<8}"
                  f"{fmt(tt):<8}"
                  f"{fmt(sweat):<8}")
            
            csv_rows.append({
                "date": d_ts.strftime("%Y-%m-%d"),
                "selection_status": status,
                "selected_hour": int(selected_hour) if (selected_hour is not None and not pd.isna(selected_hour)) else "",
                "rr": rr,
                "tavg": tavg,
                "rh": rh,
                "cin": cin,
                "kindex": kindex,
                "li": li,
                "tt": tt,
                "sweat": sweat
            })
            
        print("\n============================================================")
        print("FINAL MODEL INPUT")
        print("============================================================")
        print("\nFeature order:")
        print(FEATURE_COLUMNS)
        print("\nShape before batch:")
        print(feature_matrix.shape)
        print("\nShape after batch:")
        print(X_input.shape)
        
        # Save CSV
        csv_df = pd.DataFrame(csv_rows)
        csv_filename = f"inference_features_{target_ts.strftime('%Y-%m-%d')}.csv"
        csv_df.to_csv(csv_filename, index=False)
        
        print("\n============================================================")
        print("PREDICTION")
        print("============================================================\n")
        for cls_name in ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]:
            p = result["probabilities"].get(cls_name, 0.0) * 100
            print(f"{cls_name:<15}: {p:.2f}%")
        print(f"\nPredicted class: {result['predicted_class_name']}")
        print("============================================================")

    return result
