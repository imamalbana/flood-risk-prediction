"""
Test murah (import validation, model load, scaler load, static validation
gate logic). TIDAK ADA online test (akuisisi Ogimet/Wyoming sungguhan) --
sesuai instruksi, online test dilakukan pada CHECKPOINT 03.

Jalankan: python -m pytest tests/test_inference.py -v
(dari dalam folder inference_pipeline/, atau tambahkan inference_pipeline/
ke PYTHONPATH)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CLASS_NAMES, FEATURE_COLUMNS, LOOKBACK, MODEL_PATH, SCALER_PATH


# ---------------------------------------------------------------------------
# 1. Import validation
# ---------------------------------------------------------------------------
def test_import_all_modules():
    from src import (
        atmospheric_indices,
        calendar_utils,
        integration,
        ogimet,
        pipeline,
        predictor,
        preprocessing,
        sequence,
        sounding,
        wyoming,
    )

    assert atmospheric_indices and calendar_utils and integration
    assert ogimet and pipeline and predictor and preprocessing
    assert sequence and sounding and wyoming


# ---------------------------------------------------------------------------
# 2. Model load (compile=False, shape check)
# ---------------------------------------------------------------------------
def test_model_load():
    from src.predictor import load_model

    model = load_model(MODEL_PATH)
    assert model.input_shape == (None, LOOKBACK, len(FEATURE_COLUMNS))
    assert model.output_shape == (None, len(CLASS_NAMES))


# ---------------------------------------------------------------------------
# 3. Scaler load
# ---------------------------------------------------------------------------
def test_scaler_load():
    from src.predictor import load_scaler

    scaler = load_scaler(SCALER_PATH)
    assert scaler.n_features_in_ == len(FEATURE_COLUMNS)
    assert list(scaler.feature_names_in_) == FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 4. Static validation: config/settings.py sinkron dengan kontrak
# ---------------------------------------------------------------------------
def test_feature_columns_order_matches_contract():
    expected = ["rr", "tavg", "rh", "cin", "kindex", "li", "tt", "sweat"]
    assert FEATURE_COLUMNS == expected


def test_lookback_is_7():
    assert LOOKBACK == 7


def test_class_names_order():
    assert CLASS_NAMES == ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]


def test_stage7_monthly_medians_loadable():
    from src.preprocessing import load_monthly_medians

    medians = load_monthly_medians()
    assert set(medians.keys()) == set(range(1, 13))
    for month, values in medians.items():
        for col in ["cin", "kindex", "li", "tt", "sweat"]:
            assert col in values


# ---------------------------------------------------------------------------
# 5. Unit validation murah: calendar_utils
# ---------------------------------------------------------------------------
def test_build_window_dates_excludes_target():
    from src.calendar_utils import build_window_dates

    target = pd.Timestamp("2024-06-15")
    window = build_window_dates(target, 7)
    assert len(window) == 7
    assert window[-1] == target - pd.Timedelta(days=1)
    assert window[0] == target - pd.Timedelta(days=7)
    assert target not in window


def test_validate_window_continuity_detects_gap():
    from src.calendar_utils import validate_window_continuity

    dates = pd.date_range("2024-01-01", "2024-01-05").tolist()
    dates_with_gap = dates[:2] + dates[3:]  # buang satu tanggal di tengah
    report = validate_window_continuity(dates_with_gap)
    assert not report["valid"]
    assert any("celah" in r for r in report["reasons"])


def test_validate_window_continuity_detects_duplicate():
    from src.calendar_utils import validate_window_continuity

    dates = pd.date_range("2024-01-01", "2024-01-03").tolist()
    dates_with_dup = dates + [dates[0]]
    report = validate_window_continuity(dates_with_dup)
    assert not report["valid"]
    assert any("duplikat" in r for r in report["reasons"])


def test_parse_target_date_accepts_string():
    from src.calendar_utils import parse_target_date

    ts = parse_target_date("2024-06-15")
    assert ts == pd.Timestamp("2024-06-15")


def test_parse_target_date_rejects_garbage():
    from src.calendar_utils import parse_target_date

    with pytest.raises(ValueError):
        parse_target_date("bukan-tanggal")


# ---------------------------------------------------------------------------
# 6. Unit validation murah: sequence.py (tanpa I/O, data sintetis)
# ---------------------------------------------------------------------------
def test_build_input_sequence_shape():
    from src.sequence import build_input_sequence

    dummy = np.zeros((7, 8))
    reshaped = build_input_sequence(dummy, lookback=7)
    assert reshaped.shape == (1, 7, 8)


def test_build_input_sequence_rejects_wrong_lookback():
    from src.sequence import build_input_sequence

    dummy = np.zeros((10, 8))
    with pytest.raises(ValueError):
        build_input_sequence(dummy, lookback=7)


def test_extract_feature_matrix_raises_on_missing_date():
    from src.sequence import extract_feature_matrix

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", "2024-01-05"),
        **{col: [0.0] * 5 for col in FEATURE_COLUMNS},
    })
    window_dates = pd.date_range("2024-01-01", "2024-01-10").tolist()  # melebihi df
    with pytest.raises(KeyError):
        extract_feature_matrix(df, window_dates)


# ---------------------------------------------------------------------------
# 7. Validation gate: pastikan REJECTED tanpa memanggil model.predict()
#    ketika NaN ada di dalam window (simulasi NO_SOUNDING), tanpa jaringan.
# ---------------------------------------------------------------------------
def test_validation_gate_rejects_nan_features():
    from src.pipeline import _run_validation_gate
    from src.predictor import load_scaler

    scaler = load_scaler(SCALER_PATH)
    window_dates = pd.date_range("2024-01-01", periods=7).tolist()
    matrix = pd.DataFrame(
        np.random.rand(7, len(FEATURE_COLUMNS)),
        index=window_dates,
        columns=FEATURE_COLUMNS,
    )
    matrix.iloc[5, 3] = np.nan  # simulasikan NO_SOUNDING pada satu tanggal

    reasons = _run_validation_gate(window_dates, matrix, scaler)
    assert reasons  # tidak boleh kosong -- harus ada alasan penolakan
    assert any("NaN" in r for r in reasons)


def test_validation_gate_passes_clean_matrix():
    from src.pipeline import _run_validation_gate
    from src.predictor import load_scaler

    scaler = load_scaler(SCALER_PATH)
    window_dates = pd.date_range("2024-01-01", periods=7).tolist()
    matrix = pd.DataFrame(
        np.random.rand(7, len(FEATURE_COLUMNS)),
        index=window_dates,
        columns=FEATURE_COLUMNS,
    )

    reasons = _run_validation_gate(window_dates, matrix, scaler)
    assert reasons == []


# ---------------------------------------------------------------------------
# 18. HARDENING -- Wyoming on-disk cache (Masalah 1)
# ---------------------------------------------------------------------------
# Seluruh test di bawah memakai CACHE_DIR sementara (tmp_path, via
# monkeypatch) dan memalsukan `download_single_sounding` (bukan HTTP
# sungguhan) agar bisa berjalan tanpa jaringan sekaligus menghitung
# persis berapa kali "HTTP" dipanggil.


def test_try_hour_cached_hit_skips_http(tmp_path, monkeypatch):
    """Cache hit: panggilan kedua untuk (date, hour) yang sama TIDAK
    memicu download_single_sounding lagi."""
    import src.wyoming as wyoming_mod

    monkeypatch.setattr(wyoming_mod, "CACHE_DIR", str(tmp_path))

    call_count = {"n": 0}

    def fake_download(date_str, hour_str, src, session):
        call_count["n"] += 1
        if src == "FM35":
            df = pd.DataFrame({"time": ["2024-06-01 12:00:00"], "pressure_hPa": [1000.0]})
            return "SUCCESS", df, "Success (src=FM35)"
        return "MISSING", None, "No sounding available in body (src=None)"

    monkeypatch.setattr(wyoming_mod, "download_single_sounding", fake_download)

    result1, _ = wyoming_mod._try_hour_cached(
        "2024-06-01", "12", 12, "SUCCESS_12Z", session=None, use_cache=True
    )
    assert result1 is not None
    assert result1.status == "SUCCESS_12Z"
    calls_after_first = call_count["n"]
    assert calls_after_first >= 1

    result2, _ = wyoming_mod._try_hour_cached(
        "2024-06-01", "12", 12, "SUCCESS_12Z", session=None, use_cache=True
    )
    assert result2 is not None
    assert result2.status == "SUCCESS_12Z"
    assert result2.message == "(dari cache)"
    # Panggilan kedua tidak boleh menambah jumlah "HTTP request"
    assert call_count["n"] == calls_after_first


def test_try_hour_cached_miss_downloads(tmp_path, monkeypatch):
    """Cache miss (belum pernah diminta): download_single_sounding tetap
    dipanggil seperti biasa."""
    import src.wyoming as wyoming_mod

    monkeypatch.setattr(wyoming_mod, "CACHE_DIR", str(tmp_path))

    called = {"flag": False}

    def fake_download(date_str, hour_str, src, session):
        called["flag"] = True
        return "MISSING", None, "No sounding available in body"

    monkeypatch.setattr(wyoming_mod, "download_single_sounding", fake_download)

    result, _ = wyoming_mod._try_hour_cached(
        "2024-06-02", "00", 0, "SUCCESS_00Z", session=None, use_cache=True
    )
    assert result is None
    assert called["flag"] is True


def test_try_hour_cached_no_hour_status_persisted(tmp_path, monkeypatch):
    """Hasil 'tidak tersedia' untuk satu (date, hour) juga di-cache (status
    NO_HOUR), sehingga percobaan berikutnya tidak mengulang HTTP request."""
    import src.wyoming as wyoming_mod

    monkeypatch.setattr(wyoming_mod, "CACHE_DIR", str(tmp_path))

    call_count = {"n": 0}

    def fake_download(date_str, hour_str, src, session):
        call_count["n"] += 1
        return "MISSING", None, "No sounding available in body"

    monkeypatch.setattr(wyoming_mod, "download_single_sounding", fake_download)

    result1, _ = wyoming_mod._try_hour_cached(
        "2024-06-03", "12", 12, "SUCCESS_12Z", session=None, use_cache=True
    )
    assert result1 is None
    n_after_first = call_count["n"]
    assert n_after_first >= 1

    result2, _ = wyoming_mod._try_hour_cached(
        "2024-06-03", "12", 12, "SUCCESS_12Z", session=None, use_cache=True
    )
    assert result2 is None
    # Cache NO_HOUR harus mencegah panggilan HTTP kedua
    assert call_count["n"] == n_after_first


def test_get_sounding_rows_for_dates_passes_use_cache(tmp_path, monkeypatch):
    """Regresi: use_cache=True pada get_sounding_rows_for_dates() WAJIB
    benar-benar diteruskan sampai ke wyoming._try_hour_cached (sebelum
    hardening, parameter ini diterima tapi tidak dipakai)."""
    import src.sounding as sounding_mod
    import src.wyoming as wyoming_mod

    monkeypatch.setattr(wyoming_mod, "CACHE_DIR", str(tmp_path))

    call_count = {"n": 0}

    def fake_download(date_str, hour_str, src, session):
        call_count["n"] += 1
        if hour_str == "12" and src == "FM35":
            df = pd.DataFrame({"time": [f"{date_str} 12:00:00"], "pressure_hPa": [1000.0]})
            return "SUCCESS", df, "Success (src=FM35)"
        return "MISSING", None, "No sounding available"

    monkeypatch.setattr(wyoming_mod, "download_single_sounding", fake_download)

    def fake_compute_indices(profile_df):
        return {"cin": 0.0, "kindex": 0.0, "li": 0.0, "tt": 0.0, "sweat": 0.0}

    monkeypatch.setattr(
        sounding_mod, "compute_indices_for_sounding", fake_compute_indices
    )

    dates = [pd.Timestamp("2024-06-10")]
    df1 = sounding_mod.get_sounding_rows_for_dates(dates, use_cache=True)
    calls_after_first_window = call_count["n"]

    # Panggil lagi untuk tanggal yang SAMA (mensimulasikan window LB7
    # berikutnya yang tumpang tindih) -- tidak boleh menambah panggilan HTTP.
    df2 = sounding_mod.get_sounding_rows_for_dates(dates, use_cache=True)
    assert call_count["n"] == calls_after_first_window
    assert df1.iloc[0]["selection_status"] == SELECTED_STATUS_FOR_TEST
    assert df2.iloc[0]["selection_status"] == SELECTED_STATUS_FOR_TEST


SELECTED_STATUS_FOR_TEST = "SELECTED"


# ---------------------------------------------------------------------------
# 19. HARDENING -- Sounding nominal_date fallback (Masalah 2), skenario A-D
# ---------------------------------------------------------------------------


def test_nominal_date_scenario_a_12z_available(monkeypatch):
    """A. 12Z(D) tersedia -> dipakai langsung, request_date == D."""
    import src.sounding as sounding_mod
    from src.wyoming import SoundingResult

    calls = []

    def fake_try_hour_cached(date_str, hour_str, hour_int, status_label, session, use_cache=True):
        calls.append((date_str, hour_str))
        if hour_str == "12":
            df = pd.DataFrame({"time": [f"{date_str} 12:00:00"]})
            return SoundingResult(
                date=date_str, selected_hour=12, status="SUCCESS_12Z",
                source="FM35", df=df, message="ok",
            ), ["ok"]
        raise AssertionError("00Z tidak boleh dicoba jika 12Z(D) sukses")

    monkeypatch.setattr(sounding_mod, "_try_hour_cached", fake_try_hour_cached)

    result = sounding_mod._attempt_nominal_date(pd.Timestamp("2024-06-15"), session=None)

    assert result["selected_hour"] == 12
    assert result["request_date"] == "2024-06-15"
    assert result["selection_status"] == "SELECTED"
    assert calls == [("2024-06-15", "12")]


def test_nominal_date_scenario_b_fallback_00z_same_day(monkeypatch):
    """B. 12Z(D) gagal, 00Z(D) tersedia -> dipakai, request_date == D.
    (BUG A fix: fallback 00Z menggunakan request_date D, bukan D-1.)"""
    import src.sounding as sounding_mod
    from src.wyoming import SoundingResult

    calls = []

    def fake_try_hour_cached(date_str, hour_str, hour_int, status_label, session, use_cache=True):
        calls.append((date_str, hour_str))
        if hour_str == "12":
            return None, ["gagal"]
        df = pd.DataFrame({"time": [f"{date_str} 00:00:00"]})
        return SoundingResult(
            date=date_str, selected_hour=0, status="SUCCESS_00Z",
            source="FM35", df=df, message="ok",
        ), ["ok"]

    monkeypatch.setattr(sounding_mod, "_try_hour_cached", fake_try_hour_cached)

    result = sounding_mod._attempt_nominal_date(pd.Timestamp("2024-06-15"), session=None)

    assert result["selected_hour"] == 0
    assert result["request_date"] == "2024-06-15"  # D, bukan D-1
    assert result["nominal_date"] == "2024-06-15"
    assert result["selection_status"] == "SELECTED"
    # Hanya dua kombinasi: 12Z(D) dan 00Z(D). D-1 tidak boleh muncul.
    assert calls == [("2024-06-15", "12"), ("2024-06-15", "00")]
    assert ("2024-06-14", "00") not in calls


def test_nominal_date_scenario_c_both_fail_no_sounding(monkeypatch):
    """C. 12Z(D) gagal DAN 00Z(D-1) gagal -> NO_SOUNDING, kelima fitur None."""
    import src.sounding as sounding_mod

    def fake_try_hour_cached(date_str, hour_str, hour_int, status_label, session, use_cache=True):
        return None, ["gagal"]

    monkeypatch.setattr(sounding_mod, "_try_hour_cached", fake_try_hour_cached)

    result = sounding_mod._attempt_nominal_date(pd.Timestamp("2024-06-15"), session=None)

    assert result["selection_status"] == "NO_SOUNDING"
    assert result["selected_hour"] is None
    assert result["request_date"] is None


def test_nominal_date_scenario_d_only_12z_and_00z_same_day_tried(monkeypatch):
    """D. Hanya 12Z(D) dan 00Z(D) yang boleh dicoba. Tidak ada tanggal lain."""
    import src.sounding as sounding_mod

    calls = []

    def fake_try_hour_cached(date_str, hour_str, hour_int, status_label, session, use_cache=True):
        calls.append((date_str, hour_str))
        return None, ["gagal"]

    monkeypatch.setattr(sounding_mod, "_try_hour_cached", fake_try_hour_cached)

    result = sounding_mod._attempt_nominal_date(pd.Timestamp("2024-06-15"), session=None)

    assert result["selection_status"] == "NO_SOUNDING"
    # Hanya dua kombinasi yang boleh dicoba: 12Z(D) dan 00Z(D).
    # D-1 sama sekali tidak boleh muncul.
    assert ("2024-06-14", "12") not in calls
    assert ("2024-06-14", "00") not in calls
    assert calls == [("2024-06-15", "12"), ("2024-06-15", "00")]


# ---------------------------------------------------------------------------
# 20. DATE ALIGNMENT FIX: TEST 1-5 (BUG A / B / C)
# ---------------------------------------------------------------------------

def test_date_alignment_test1_12z_success_stops(monkeypatch):
    """TEST 1: 12Z(D) SUCCESS -> selected=12Z, 00Z tidak dipanggil."""
    import src.sounding as sounding_mod
    from src.wyoming import SoundingResult

    calls = []

    def fake_try_hour_cached(date_str, hour_str, hour_int, status_label, session, use_cache=True):
        calls.append((date_str, hour_str))
        if hour_str == "12":
            df = pd.DataFrame({"time": [f"{date_str} 12:00:00"]})
            return SoundingResult(
                date=date_str, selected_hour=12, status="SUCCESS_12Z",
                source="FM35", df=df, message="ok",
            ), ["ok"]
        raise AssertionError("00Z tidak boleh dipanggil jika 12Z sukses")

    monkeypatch.setattr(sounding_mod, "_try_hour_cached", fake_try_hour_cached)

    result = sounding_mod._attempt_nominal_date(pd.Timestamp("2024-06-15"), session=None)

    assert result["selected_hour"] == 12
    assert result["selection_status"] == "SELECTED"
    # 00Z tidak boleh dipanggil sama sekali
    assert calls == [("2024-06-15", "12")]
    assert not any(h == "00" for _, h in calls)


def test_date_alignment_test2_12z_fail_00z_uses_same_day(monkeypatch):
    """TEST 2: 12Z(D) FAIL, 00Z(D) SUCCESS -> selected=00Z, request_date=D."""
    import src.sounding as sounding_mod
    from src.wyoming import SoundingResult

    calls = []

    def fake_try_hour_cached(date_str, hour_str, hour_int, status_label, session, use_cache=True):
        calls.append((date_str, hour_str))
        if hour_str == "12":
            return None, ["gagal"]
        df = pd.DataFrame({"time": [f"{date_str} 00:00:00"]})
        return SoundingResult(
            date=date_str, selected_hour=0, status="SUCCESS_00Z",
            source="FM35", df=df, message="ok",
        ), ["ok"]

    monkeypatch.setattr(sounding_mod, "_try_hour_cached", fake_try_hour_cached)

    result = sounding_mod._attempt_nominal_date(pd.Timestamp("2024-06-15"), session=None)

    assert result["selected_hour"] == 0
    assert result["selection_status"] == "SELECTED"
    # request_date harus D (2024-06-15), bukan D-1
    assert result["request_date"] == "2024-06-15"
    assert result["nominal_date"] == "2024-06-15"
    # D-1 tidak boleh muncul sama sekali
    assert ("2024-06-14", "00") not in calls
    assert ("2024-06-14", "12") not in calls


def test_date_alignment_test3_00z_obs_datetime_prev_day_nominal_is_D(monkeypatch):
    """TEST 3: 00Z(D) SUCCESS, observation_datetime=D-1 23:30 -> nominal_date=D.
    observation_datetime yang berbeda dari request_date tidak boleh menggeser
    nominal_date."""
    import src.sounding as sounding_mod
    from src.wyoming import SoundingResult

    D = pd.Timestamp("2024-06-15")
    D_minus_1 = "2024-06-14"

    def fake_try_hour_cached(date_str, hour_str, hour_int, status_label, session, use_cache=True):
        if hour_str == "12":
            return None, ["gagal"]
        # 00Z sounding direkam pada D-1 23:30 (observation_datetime = D-1)
        df = pd.DataFrame({
            "time": [f"{D_minus_1} 23:30:00"],
            "observation_datetime": [f"{D_minus_1} 23:30:00"],
            "observation_hour": [0],
        })
        return SoundingResult(
            date=date_str, selected_hour=0, status="SUCCESS_00Z",
            source="FM35", df=df, message="ok",
        ), ["ok"]

    monkeypatch.setattr(sounding_mod, "_try_hour_cached", fake_try_hour_cached)

    result = sounding_mod._attempt_nominal_date(D, session=None)

    # nominal_date harus D (2024-06-15), terlepas dari observation_datetime
    assert result["nominal_date"] == "2024-06-15"
    assert result["request_date"] == "2024-06-15"
    assert result["selected_hour"] == 0
    assert result["selection_status"] == "SELECTED"


def test_date_alignment_test4_one_nominal_date_max_one_selected(monkeypatch):
    """TEST 4: Satu nominal_date menghasilkan maksimal satu selected sounding."""
    import src.sounding as sounding_mod
    from src.wyoming import SoundingResult

    calls = []

    def fake_try_hour_cached(date_str, hour_str, hour_int, status_label, session, use_cache=True):
        calls.append((date_str, hour_str))
        # Keduanya sukses -- tapi 12Z harus dipilih dan 00Z tidak dicoba
        df = pd.DataFrame({"time": [f"{date_str} {hour_str}:00:00"]})
        return SoundingResult(
            date=date_str, selected_hour=hour_int, status=f"SUCCESS_{hour_str}Z",
            source="FM35", df=df, message="ok",
        ), ["ok"]

    monkeypatch.setattr(sounding_mod, "_try_hour_cached", fake_try_hour_cached)

    result = sounding_mod._attempt_nominal_date(pd.Timestamp("2024-06-15"), session=None)

    # Hanya 12Z yang terpilih; 00Z tidak pernah dicoba karena 12Z sukses
    assert result["selection_status"] == "SELECTED"
    assert result["selected_hour"] == 12
    # Hanya satu percobaan terjadi (12Z); tidak ada percobaan 00Z
    assert calls == [("2024-06-15", "12")]
    selected_rows = [r for r in [result] if r["selection_status"] == "SELECTED"]
    assert len(selected_rows) == 1


def test_date_alignment_test5_integration_one_feature_row_per_date(monkeypatch):
    """TEST 5: Sounding DataFrame tidak boleh mengandung duplicate date."""
    import src.sounding as sounding_mod
    from src.wyoming import SoundingResult

    def fake_try_hour_cached(date_str, hour_str, hour_int, status_label, session, use_cache=True):
        if hour_str == "12":
            df = pd.DataFrame({"time": [f"{date_str} 12:00:00"]})
            return SoundingResult(
                date=date_str, selected_hour=12, status="SUCCESS_12Z",
                source="FM35", df=df, message="ok",
            ), ["ok"]
        return None, ["gagal"]

    def fake_compute_indices(profile_df):
        return {"cin": 0.0, "kindex": 0.0, "li": 0.0, "tt": 0.0, "sweat": 0.0}

    monkeypatch.setattr(sounding_mod, "_try_hour_cached", fake_try_hour_cached)
    monkeypatch.setattr(sounding_mod, "compute_indices_for_sounding", fake_compute_indices)

    dates = [pd.Timestamp("2024-06-15"), pd.Timestamp("2024-06-16")]
    result_df = sounding_mod.get_sounding_rows_for_dates(dates, use_cache=False)

    # Satu tanggal = satu baris
    assert len(result_df) == 2
    assert result_df["date"].nunique() == 2
    # Tidak ada duplicate
    assert not result_df["date"].duplicated().any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
