# FINAL REPORT — Inference Pipeline (LSTM Flood Risk, Kota Padang)

**Lingkup pekerjaan sesi ini:** final review + packaging atas `inference_pipeline/`
dari CHECKPOINT_03. **Tidak ada** pipeline dibangun ulang, tidak ada training,
tidak ada Optuna, tidak ada Streamlit. Metodologi penelitian tidak diubah.

---

## 1. Status Final

**PACKAGE-READY.** Seluruh 17 test unit/logika lulus (dijalankan ulang di sesi
ini, hasil identik dengan `VALIDATION_REPORT.md` CHECKPOINT_03). Tidak
ditemukan bug baru pada re-check singkat terhadap `pipeline.py`,
`settings.py`, `requirements.txt`, dan `README.md`. Tidak ada perubahan kode
yang dilakukan pada sesi ini — checkpoint 03 sudah bersih dari sisi
package/import/model/scaler/feature order/Stage 7/median bulanan/date
alignment/sounding selection/integration/sequence LB30/validation
gate/error handling/cache/requirements/README.

## 2. File dalam Paket

```
inference_pipeline/
├── FINAL_REPORT.md              <- baru (laporan ini)
├── README.md                    <- tidak diubah
├── requirements.txt              <- tidak diubah, sudah tersedia & sudah diverifikasi terinstal bersih
├── config/
│   ├── settings.py               <- kontrak: FEATURE_COLUMNS, LOOKBACK=30, CLASS_NAMES
│   └── stage7_monthly_medians.json
├── models/
│   ├── model_final_4_class.keras <- tidak disentuh
│   └── scaler.pkl                <- tidak disentuh
├── src/
│   ├── ogimet.py, wyoming.py, atmospheric_indices.py
│   ├── calendar_utils.py, sounding.py, integration.py
│   ├── preprocessing.py, sequence.py, predictor.py, pipeline.py
└── tests/
    └── test_inference.py
```

## 3. Status Test

Dijalankan ulang di sesi ini: `pytest tests/test_inference.py -v`

```
17 passed in 5.44s
```

Mencakup: import semua modul, load model (`input_shape=(None,30,8)`,
`output_shape=(None,4)`), load scaler (`n_features_in_=8`, urutan fitur
cocok kontrak), urutan `FEATURE_COLUMNS` == `scaler.feature_names_in_`,
`LOOKBACK=30`, urutan `CLASS_NAMES`, median bulanan Stage 7 termuat 12
bulan lengkap, `build_window_dates` mengecualikan tanggal target,
deteksi celah/duplikat pada window, parsing tanggal target, shape
`(1,30,8)`, penolakan lookback salah, `KeyError` saat tanggal hilang,
dan validation gate menolak NaN / meloloskan matriks bersih.

Semua ini adalah pengujian **offline/lokal**. Tidak ada perubahan pada
hasil dibanding `VALIDATION_REPORT.md` CHECKPOINT_03 (14 item PASS, 2 item
live-HTTP NOT TESTED — lihat Bagian 6).

## 4. Known Limitation

1. **Akses jaringan live ke `weather.uwyo.edu` dan `ogimet.com` tidak
   dapat diuji** dari sandbox manapun yang egress-nya dibatasi allowlist
   (dikonfirmasi 403 di CHECKPOINT_03, dan environment sesi ini punya
   allowlist yang sama — hanya PyPI/GitHub/npm, tanpa kedua host
   tersebut). Ini bukan bug kode, melainkan keterbatasan environment
   testing. Pipeline harus dijalankan di environment dengan akses
   internet penuh ke kedua host tersebut untuk pengujian end-to-end
   sungguhan.
2. **`sounderpy` dan `metpy`** (dependency `atmospheric_indices.py`)
   berhasil diinstal & diimpor dari PyPI di sesi ini, tapi fungsi
   perhitungan indeksnya sendiri (CAPE/CIN/K-Index/dst dari data sounding
   nyata) tidak dieksekusi ulang di sesi ini — sudah divalidasi di
   riwayat CHECKPOINT sebelumnya (lihat `VALIDATION_REPORT.md`), tidak
   diulang di sini sesuai instruksi "jangan membaca ulang seluruh
   Stage 1–12".

## 5. Masalah Metodologis yang BELUM Diputuskan (dicatat, TIDAK diam-diam diperbaiki)

Kedua item ini diwariskan dari CHECKPOINT 01 dan sengaja dibiarkan terbuka
oleh implementasi (lihat `config/settings.py` dan `src/preprocessing.py`
docstring). Ini keputusan metodologis, bukan bug — tidak diubah pada sesi
ini:

- **OPEN DECISION 1** — ukuran buffer data sebelum D-30 untuk interpolasi
  Ogimet (`interpolate(method="time", limit_direction="both")`) pada window
  terbatas. Default saat ini `integration_start_buffer_days=0` (tanpa
  buffer). Perlu keputusan riset apakah buffer > 0 diperlukan agar
  interpolasi pada tanggal awal window tidak bias.
- **OPEN DECISION 2** — desain modul akuisisi data sounding. Implementasi
  saat ini memilih opsi paling konservatif: mereplikasi persis logika
  per-tanggal `wyouming_downloader.py::process_date` (12Z fallback ke
  00Z hari sebelumnya). Ini konsisten dengan training, tapi belum
  "disetujui secara resmi" sebagai desain final menurut catatan
  CHECKPOINT 01.

Tidak ada perbaikan diam-diam dilakukan terhadap dua item ini pada sesi
manapun — keduanya tetap eksplisit sebagai parameter/`None` di kode agar
tidak ada nilai yang dikarang.

## 6. Cara Menjalankan

```bash
cd inference_pipeline
pip install -r requirements.txt
```

```python
from src.pipeline import predict_for_date

result = predict_for_date("2024-06-15")
print(result)
# SUKSES: {"status": "SUCCESS", "target_date": "...",
#          "predicted_class_index": ..., "predicted_class_name": ...,
#          "probabilities": {...}}
# GAGAL VALIDASI: {"status": "REJECTED", "target_date": "...", "reasons": [...]}
```

Untuk pemanggilan berulang, muat model/scaler sekali di luar:

```python
from src.predictor import load_model, load_scaler
model, scaler = load_model(), load_scaler()
result = predict_for_date("2024-06-15", model=model, scaler=scaler)
```

**Catatan:** menjalankan `predict_for_date()` sungguhan memerlukan akses
internet ke `ogimet.com` dan `weather.uwyo.edu` (tidak tersedia di sandbox
review ini — lihat Bagian 4).

## 7. Apa yang Belum Dapat Diverifikasi (di sesi review ini maupun sesi CHECKPOINT_03)

- Hasil prediksi end-to-end pada tanggal nyata (mis. `2024-11-27`,
  `2026-08-06`) — **NOT TESTED**, bukan PASS/FAIL, karena tidak ada akses
  jaringan ke sumber data live dari sandbox manapun yang dipakai sejauh
  ini. Tidak ada data sintetis dipakai untuk berpura-pura sebagai hasil
  online.
- Akurasi numerik CAPE/CIN/K-Index/LI/TT/SWEAT dari data sounding nyata
  (bergantung pada `sounderpy`/`SHARPpy` yang delegasinya tidak diaudit
  ulang isi algoritmanya pada sesi ini — hanya diverifikasi dapat
  diimpor).
- Perilaku pipeline saat kegagalan jaringan parsial (mis. Ogimet berhasil
  tapi Wyoming timeout di tengah rentang tanggal) belum diuji dengan
  skenario network-flaky sungguhan; yang sudah diuji adalah exception
  handling generik (`try/except Exception` membungkus akuisisi dan
  mengembalikan status `REJECTED` dengan alasan, tidak pernah membuat
  pipeline crash tanpa penjelasan).

---

*Laporan ini melengkapi, bukan menggantikan, `VALIDATION_REPORT.md` dan
`CHANGELOG.md` dari CHECKPOINT_03 yang tetap disertakan sebagai referensi
riwayat lengkap.*
