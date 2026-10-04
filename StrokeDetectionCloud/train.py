
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# File locations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "stroke_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Prepare features
df = df.drop(columns=["id", "gender", "ever_married",
                      "work_type", "Residence_type",
                      "smoking_status"], errors="ignore")

df["bmi"] = df["bmi"].fillna(df["bmi"].median())

X = df.drop(columns=["stroke"])
y = df["stroke"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a neural network
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

# Save model and scaler
model.save(os.path.join(MODEL_DIR, "stroke_lstm.keras"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("Model and scaler saved successfully!")
