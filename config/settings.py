"""
Konfigurasi statis inference pipeline.

Nilai-nilai di sini bersumber dari INFERENCE_CONTRACT.md dan
STAGE_DEPLOYMENT_MAP.md (CHECKPOINT 01). Dilarang mengubah FEATURE_COLUMNS,
LOOKBACK, atau CLASS_NAMES tanpa audit ulang terhadap artefak model/scaler.
"""

from __future__ import annotations

import os

# -----------------------------------------------------------------------
# PATH
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(MODELS_DIR, "model_final_4_class.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
STAGE7_MEDIANS_PATH = os.path.join(CONFIG_DIR, "stage7_monthly_medians.json")

# Direktori cache akuisisi data (Ogimet/Wyoming), dibuat saat runtime jika
# belum ada. Bukan bagian dari kontrak model, hanya kemudahan operasional.
CACHE_DIR = os.path.join(BASE_DIR, ".cache")

# -----------------------------------------------------------------------
# KONTRAK MODEL (INFERENCE_CONTRACT.md Bagian 1-2-3-5-6) — JANGAN UBAH
# -----------------------------------------------------------------------
STATION_ID = "96163"

# Urutan ini WAJIB identik dengan scaler.feature_names_in_ dan dengan
# FEATURE_COLUMNS pada kode aktual Stage 10. Dilarang mengurutkan ulang.
FEATURE_COLUMNS = [
    "rr",       # curah hujan harian (Ogimet)
    "tavg",     # suhu udara rata-rata harian (Ogimet)
    "rh",       # kelembapan relatif (Ogimet)
    "cin",      # SBCIN (SounderPy)
    "kindex",   # K-Index (SHARPpy direct)
    "li",       # LI_SB_500 (SHARPpy direct)
    "tt",       # Total Totals (SHARPpy direct)
    "sweat",    # SWEAT (SHARPpy direct)
]

# Kolom Ogimet vs kolom SounderPy/SHARPpy, dipakai untuk memisahkan
# strategi missing-value handling (Stage 7).
OGIMET_FEATURE_COLUMNS = ["rr", "tavg", "rh"]
SOUNDING_FEATURE_COLUMNS = ["cin", "kindex", "li", "tt", "sweat"]

LOOKBACK = 30  # D-30 ... D-1, target D tidak termasuk window

CLASS_NAMES = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]

# -----------------------------------------------------------------------
# STAGE 2 — placeholder standardisasi Ogimet (WAJIB direplikasi persis)
# -----------------------------------------------------------------------
OGIMET_PLACEHOLDER_MAP = {
    "Tr": 0.0,
    "----": None,
    "-----": None,
}

# -----------------------------------------------------------------------
# STAGE 4 / wyouming_downloader.py — prioritas sumber & jam observasi
# -----------------------------------------------------------------------
SRC_ORDER = ["FM35", None]
HOUR_PRIORITY = [12, 0]  # coba 12Z dulu, fallback 00Z

# -----------------------------------------------------------------------
# STAGE 7 — missing-value handling (WAJIB direplikasi persis)
# -----------------------------------------------------------------------
NO_SOUNDING_STATUS = "NO_SOUNDING"
SELECTED_STATUS = "SELECTED"

# OPEN DECISION 1 (INFERENCE_CONTRACT.md #10): ukuran buffer sebelum D-30
# untuk interpolate(method="time", limit_direction="both") BELUM diputuskan
# pada CHECKPOINT 01. Implementasi ini TIDAK mengarang nilai baru — buffer
# harus disuplai eksplisit oleh caller (lihat src/preprocessing.py) dan
# kegagalan menyuplai cukup data historis akan membuat validation gate
# menolak prediksi, bukan diam-diam memakai default yang tidak diverifikasi.
INTERPOLATION_BUFFER_DAYS = None  # sengaja None -- lihat OPEN DECISION 1

# Path ke integrated dataset historis
HISTORICAL_DATASET_PATH = os.path.join(BASE_DIR, "data", "integrated", "integrated_dataset.csv")


