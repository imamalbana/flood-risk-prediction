# VALIDATION REPORT — CHECKPOINT 03

**Lingkup:** Test & debug `inference_pipeline/` dari CHECKPOINT 02. Tidak ada pipeline dibangun ulang, tidak ada Streamlit, tidak ada training/Optuna.
**Environment:** egress network dibatasi allowlist (`api.anthropic.com`, `pypi.org`, dst). `ogimet.com` dan `weather.uwyo.edu` **TIDAK** ada di allowlist (dikonfirmasi HTTP 403 "Host not in allowlist").

Status yang dipakai: **PASS / FAIL / NOT TESTED / BLOCKED**

---

## 1. Package / import — **PASS**
`pytest tests/test_inference.py::test_import_all_modules` PASSED. Seluruh modul (`ogimet`, `wyoming`, `atmospheric_indices`, `calendar_utils`, `sounding`, `integration`, `preprocessing`, `sequence`, `predictor`, `pipeline`) berhasil diimpor. `sounderpy` (termasuk `sounderpy.SHARPPYMAIN.sharppy.sharptab.*`) berhasil diinstal dari PyPI dan diverifikasi dapat diimpor langsung.

## 2. Model loading — **PASS**
`load_model(compile=False)` berhasil. `model.input_shape == (None, 30, 8)`, `model.output_shape == (None, 4)` — cocok dengan `INFERENCE_CONTRACT.md`.

## 3. Scaler loading — **PASS**
`scaler.n_features_in_ == 8`, `scaler.feature_names_in_ == ['rr','tavg','rh','cin','kindex','li','tt','sweat']`.

## 4. Feature order — **PASS**
`FEATURE_COLUMNS` di `config/settings.py` identik urutan dengan `scaler.feature_names_in_` dan kontrak. Diverifikasi otomatis via `test_feature_columns_order_matches_contract`.

## 5. Stage 7 preprocessing — **PASS**
Diuji offline dengan data sintetis: interpolasi Ogimet (`method="time", limit_direction="both"`) mengisi NaN `rr` dengan benar; baris `SELECTED` dengan fitur atmosfer NaN diisi median bulanan; baris `NO_SOUNDING` **tetap NaN** (0 imputasi pada 2 baris NO_SOUNDING uji).

## 6. Monthly median — **PASS**
`config/stage7_monthly_medians.json` termuat, 12 bulan lengkap, kelima kolom (`cin,kindex,li,tt,sweat`) ada di tiap bulan. Nilai dipakai persis dari file (tidak dihitung ulang).

## 7. Sounding selection — **PASS (logika/unit), NOT TESTED (live HTTP)**
Logika `src/sounding.py` (12Z pada request_date=D, fallback 00Z pada request_date=D-1) diverifikasi terhadap alur `wyouming_downloader.py::process_date` secara statis — konsisten dengan aturan `nominal_date`. Panggilan HTTP sungguhan ke `weather.uwyo.edu` **tidak dapat diuji** (host tidak ada di allowlist jaringan sandbox ini).

## 8. nominal_date — **PASS**
Aturan `nominal_date` (00Z → tanggal request+1; 12Z → tanggal request sama) direplikasi identik dari `atmospheric_indices.py::export_results`. Konsekuensinya terhadap pemilihan request_date per nominal_date target didokumentasikan di `src/sounding.py` dan `BUILD_REPORT.md` CHECKPOINT 02.

## 9. Integration — **PASS**
Diuji offline: `backbone LEFT JOIN ogimet LEFT JOIN sounding` menghasilkan jumlah baris = jumlah baris backbone (32 == 32), tidak ada baris hilang/bertambah.

## 10. LB30 — **PASS**
`LOOKBACK = 30` diverifikasi via `test_lookback_is_30` dan dipakai konsisten di `build_window_dates`, `extract_feature_matrix`, `build_input_sequence`.

## 11. Sequence D-30...D-1 — **PASS**
`build_window_dates(target, 30)` menghasilkan 30 tanggal ascending tanpa celah, tanggal target **tidak termasuk** (diverifikasi: `target not in window` = True, `window[-1] == target - 1 hari`, `window[0] == target - 30 hari`).

## 12. Shape (1,30,8) — **PASS**
Diuji end-to-end offline (data sintetis, lihat Bagian "Test Wiring Offline" di bawah): `X_input.shape == (1, 30, 8)` tercapai melalui `extract_feature_matrix` → `scale_features` → `build_input_sequence`.

## 13. NaN validation gate — **PASS**
Dua skenario diuji:
- Window bersih (tanpa `NO_SOUNDING`) → gate lulus, `reasons == []`.
- Window mengandung satu tanggal `NO_SOUNDING` → gate **REJECTED** dengan alasan spesifik (tanggal + kolom yang NaN), `model.predict()` **tidak pernah dipanggil** (diverifikasi lewat pengujian `predict_for_date` dengan sounding sintetis yang sengaja mengandung NO_SOUNDING).

## 14. Class mapping — **PASS**
`CLASS_NAMES = ["Rendah","Sedang","Tinggi","Sangat Tinggi"]` sesuai `Dense(4)` & kontrak. Diverifikasi hasil `argmax` konsisten dengan key `probabilities` pada uji wiring offline.

---

## TEST HISTORIS — 2024-11-27 — **NOT TESTED**
Environment sandbox ini tidak memiliki akses jaringan ke `ogimet.com` maupun `weather.uwyo.edu` (keduanya di luar network egress allowlist, terverifikasi via curl → `403 Host not in allowlist`). Sesuai instruksi, **tidak ada data palsu dibuat** untuk mensimulasikan hasil online. Validasi yang dilakukan sebagai gantinya adalah validasi logika/unit/lokal (Bagian 1-14 di atas) plus uji wiring end-to-end dengan data sintetis yang eksplisit ditandai sintetis (lihat di bawah), bukan hasil tanggal 2024-11-27 sungguhan.

## TEST TANGGAL BARU — 2026-08-06 — **NOT TESTED**
Alasan sama seperti di atas: tidak ada akses ke sumber data Ogimet/Wyoming dari environment ini. Tidak dipaksakan prediksi.

## BLOCKED
Tidak ada item yang berstatus BLOCKED — seluruh keterbatasan di atas bersumber dari isolasi jaringan sandbox (izin akses eksternal), bukan dari cacat pada pipeline atau ketidaklengkapan kode.

---

## Uji Wiring Offline (data sintetis, BUKAN hasil online, hanya untuk memvalidasi jalur kode)

Menggunakan monkeypatch pada `get_ogimet_daily`/`get_sounding_rows_for_dates` dengan angka acak (`numpy.random`, seed tetap) agar bentuk data mendekati bentuk data asli tanpa berpura-pura sebagai observasi nyata:

- Skenario window bersih -> `predict_for_date("2024-07-01", ...)` -> `status: SUCCESS`, `predicted_class_name: "Sedang"`, shape `(1,30,8)` tanpa NaN.
- Skenario window mengandung `NO_SOUNDING` -> `status: REJECTED` dengan alasan eksplisit, `model.predict()` tidak dipanggil.

Hasil ini membuktikan jalur kode (acquisition-interface → integrasi → Stage 7 → validation gate → scaling → predict) tersambung benar; **bukan** bukti akurasi terhadap data cuaca sungguhan tanggal manapun.

---

## Ringkasan status

| No | Item | Status |
|---|---|---|
| 1 | Package/import | PASS |
| 2 | Model loading | PASS |
| 3 | Scaler loading | PASS |
| 4 | Feature order | PASS |
| 5 | Stage 7 preprocessing | PASS |
| 6 | Monthly median | PASS |
| 7 | Sounding selection (logika) | PASS |
| 7b | Sounding selection (live HTTP) | NOT TESTED |
| 8 | nominal_date | PASS |
| 9 | Integration | PASS |
| 10 | LB30 | PASS |
| 11 | Sequence D-30...D-1 | PASS |
| 12 | Shape (1,30,8) | PASS |
| 13 | NaN validation gate | PASS |
| 14 | Class mapping | PASS |
| — | Online test 2024-11-27 | NOT TESTED (no network access) |
| — | Online test 2026-08-06 | NOT TESTED (no network access) |

Lihat `CHANGELOG.md` untuk daftar bug yang ditemukan & diperbaiki.
