## Instalasi

```bash
pip install -r requirements.txt
```

`sounderpy` menyertakan modul internal SHARPpy (`sounderpy.SHARPPYMAIN.*`)
yang dipakai langsung oleh `src/atmospheric_indices.py` untuk LI/TT/K-Index/
SWEAT (tidak diekspos API resmi SounderPy).

## Pemakaian

```python
from src.pipeline import predict_for_date

result = predict_for_date("2024-06-15")
print(result)
# {"status": "SUCCESS", "target_date": "2024-06-15",
#  "predicted_class_index": ..., "predicted_class_name": ...,
#  "probabilities": {...}}
# atau:
# {"status": "REJECTED", "target_date": "...", "reasons": [...]}
```

Model dan scaler dapat dimuat sekali di luar dan dioper ke pemanggilan
berulang agar lebih efisien:

```python
from src.predictor import load_model, load_scaler
model = load_model()
scaler = load_scaler()
result = predict_for_date("2024-06-15", model=model, scaler=scaler)
```

## Struktur

```
inference_pipeline/
├── models/
│   ├── model_final_4_class.keras   
│   └── scaler.pkl                  
├── config/
│   ├── settings.py                 # kontrak (feature order, lookback, class mapping)
│   └── stage7_monthly_medians.json 
├── src/
│   ├── ogimet.py           # get data + standardisasi Ogimet (Stage 2)
│   ├── wyoming.py          # get data sounding per-tanggal (downloader logic)
│   ├── atmospheric_indices.py  # indeks atmosfer SounderPy/SHARPpy (tidak diubah algoritmanya)
│   ├── calendar_utils.py   # master calendar, window D-30..D-1
│   ├── sounding.py         # seleksi 12Z->00Z per nominal_date + hitung indeks
│   ├── integration.py      # LEFT JOIN backbone->Ogimet->SounderPy
│   ├── preprocessing.py    # interpolasi Ogimet + median bulanan SounderPy
│   ├── sequence.py         # scaler.transform() + reshape (1,30,8)
│   ├── predictor.py        # load_model(compile=False), predict, argmax, class mapping
│   └── pipeline.py         # predict_for_date() + validation gate
├── tests/
│   └── test_inference.py   # testing (tanpa jaringan)
├── requirements.txt
└── README.md
```

Nama `calendar_utils.py` (bukan `calendar.py`) sengaja dipilih agar tidak
membayangi modul stdlib Python `calendar`.

## Aplikasi Streamlit (UI)

Untuk menjalankan antarmuka visual monokrom minimalis berbasis Streamlit:

```bash
streamlit run app.py
```

Aplikasi web dapat diakses melalui browser pada `http://localhost:8501`.

## Penting: Excluded Artifacts & Dataset

Demi alasan keamanan dan batas ukuran file (file size limits), repositori ini **tidak menyertakan**:
1. **Model & Scaler**: Berkas model deep learning (`models/model_final_4_class.keras`) dan normalisasi (`models/scaler.pkl`) harus ditempatkan di direktori `models/` secara manual secara lokal.
2. **Dataset Penelitian Asli**: Dataset terintegrasi penuh (`data/integrated/integrated_dataset.csv`) tidak disertakan. Tempatkan dataset asli Anda di jalur tersebut.
3. Sebagai acuan struktur/skema data, berkas contoh telah disediakan di `data/integrated/integrated_dataset_example.csv`.

Struktur Direktori Final:
```
inference_pipeline/
├── app.py                      # Aplikasi Streamlit Single-Page & Live Tracker
├── models/
│   ├── .gitkeep
│   ├── model_final_4_class.keras   # [Lokal saja] File model
│   └── scaler.pkl                  # [Lokal saja] File normalisasi scaler
├── config/
│   ├── settings.py
│   └── stage7_monthly_medians.json
├── data/
│   └── integrated/
│       ├── integrated_dataset_example.csv  # Contoh skema/template dataset
│       └── integrated_dataset.csv          # [Lokal saja] Dataset asli
├── src/
│   └── ... (Inference modules)
├── tests/
│   └── test_inference.py
├── requirements.txt
└── README.md
```
