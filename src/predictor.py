"""
Load model & scaler, jalankan model.predict(), argmax, class mapping.

WAJIB `compile=False` (INFERENCE_CONTRACT.md Bagian 5): model tersimpan
dengan loss custom (`loss_fn`) yang tidak terdaftar sebagai custom_object,
`load_model(..., compile=True)` akan gagal dengan TypeError. Inference
hanya butuh `model.predict()`, sehingga compile/optimizer/loss tidak
relevan.

Dilarang (INFERENCE_CONTRACT.md Bagian 9): model.fit(...).
"""

from __future__ import annotations

import joblib
import numpy as np

from config.settings import CLASS_NAMES, MODEL_PATH, SCALER_PATH, LOOKBACK, FEATURE_COLUMNS


def load_model(path: str = MODEL_PATH):
    """keras.models.load_model(path, compile=False) -- WAJIB compile=False."""
    import keras

    return keras.models.load_model(path, compile=False)


def load_scaler(path: str = SCALER_PATH):
    return joblib.load(path)


def predict(model, X_input: np.ndarray) -> dict:
    """Jalankan model.predict() pada X_input shape (1, 7, 8), argmax,
    dan class mapping. TIDAK memanggil model.fit() atau melatih ulang
    apa pun."""
    expected_shape = (1, LOOKBACK, len(FEATURE_COLUMNS))
    if X_input.shape != expected_shape:
        raise ValueError(f"Shape input harus {expected_shape}, diterima {X_input.shape}")

    proba = model.predict(X_input, verbose=0)
    pred_class = int(np.argmax(proba, axis=1)[0])

    return {
        "predicted_class_index": pred_class,
        "predicted_class_name": CLASS_NAMES[pred_class],
        "probabilities": {
            CLASS_NAMES[i]: float(proba[0][i]) for i in range(len(CLASS_NAMES))
        },
    }
