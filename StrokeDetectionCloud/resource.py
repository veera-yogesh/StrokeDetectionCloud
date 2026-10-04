import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "stroke_lstm.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi"
]

model = None
scaler = None


def load_resources():
    global model, scaler

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found.")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler file not found.")

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


def predict_stroke(values):
    global model, scaler

    if model is None or scaler is None:
        load_resources()

    data = np.asarray(values, dtype=np.float32)

    if data.shape != (5,):
        raise ValueError(
            "Enter exactly 5 values: age, hypertension, "
            "heart disease, average glucose, and BMI."
        )

    if not np.isfinite(data).all():
        raise ValueError("Input contains invalid values.")

    if data[1] not in (0, 1) or data[2] not in (0, 1):
        raise ValueError(
            "Hypertension and heart disease must be 0 or 1."
        )

    data_scaled = scaler.transform(data.reshape(1, 5))

    probability = float(
        model.predict(data_scaled, verbose=0)[0][0]
    )

    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "result": (
            "Positive model classification"
            if prediction == 1
            else "Negative model classification"
        )
    }