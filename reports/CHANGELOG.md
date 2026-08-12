# CHANGELOG — CHECKPOINT 02 → CHECKPOINT 03

## Fixed

- **`src/sequence.py::scale_features`** — sebelumnya mengirim `feature_matrix.to_numpy(dtype=float)`
  (ndarray polos, tanpa nama kolom) ke `scaler.transform()`. Karena `scaler` di-*fit* dengan
  `feature_names_in_`, ini memicu `UserWarning: X does not have valid feature names, but
  MinMaxScaler was fitted with feature names`. Diperbaiki dengan mengirim `feature_matrix`
  sebagai DataFrame (`feature_matrix.astype(float)`) langsung ke `scaler.transform()`.
  **Tidak mengubah** nilai hasil transform, urutan fitur, atau perilaku `fit` — murni
  representasi input. Diverifikasi: `scale_features()` tidak lagi memicu warning tersebut
  (diuji dengan `warnings.simplefilter("error")`).

## Verified (tidak ada perubahan kode, hanya konfirmasi lewat pengujian)

- Model & scaler load sesuai kontrak (`compile=False`, `n_features_in_=8`, urutan fitur).
- Stage 7 (interpolasi Ogimet + median bulanan SounderPy pada SELECTED, NO_SOUNDING tetap NaN)
  berperilaku sesuai kontrak pada data sintetis.
- Integrasi LEFT JOIN tidak mengurangi/menambah baris backbone.
- Window D-30..D-1 tidak menyertakan tanggal target, tidak ada celah, tidak duplikat.
- Validation gate menolak prediksi (status `REJECTED`, `model.predict()` tidak dipanggil)
  ketika ada `NO_SOUNDING` di dalam window; meloloskan window bersih ke shape `(1,30,8)`
  tanpa NaN.
- `sounderpy` (termasuk modul internal `sounderpy.SHARPPYMAIN.sharppy.sharptab.*`) berhasil
  diinstal dan diimpor di environment test ini — dependency runtime untuk
  `src/atmospheric_indices.py` terkonfirmasi tersedia di PyPI.

## Tidak diubah (sesuai batasan tugas)

- Model (`model_final_4_class.keras`) — tidak disentuh.
- Scaler (`scaler.pkl`) — tidak disentuh.
- Urutan `FEATURE_COLUMNS` — tidak diubah.
- `LOOKBACK = 30` — tidak diubah.
- `CLASS_NAMES` / class mapping — tidak diubah.
- Metodologi Stage 2/3/4/5/7/10 — tidak diubah, hanya diverifikasi ulang.

## Known limitation (bukan bug, keterbatasan environment)

- Online test terhadap `ogimet.com` dan `weather.uwyo.edu` **tidak dapat dijalankan** pada
  environment ini karena kedua host berada di luar network egress allowlist sandbox
  (`403 Host not in allowlist`). Tanggal 2024-11-27 dan 2026-08-06 berstatus **NOT TESTED**,
  bukan PASS/FAIL — lihat `VALIDATION_REPORT.md`. Tidak ada data palsu dibuat untuk
  mensimulasikan hasil kedua tanggal tersebut.
- OPEN DECISION 1 (buffer interpolasi Ogimet) dan OPEN DECISION 2 (desain modul akuisisi)
  yang diwariskan dari CHECKPOINT 01 **masih belum diputuskan secara resmi** — tidak
  termasuk cakupan test & debug checkpoint ini.
