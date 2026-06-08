"""
inference.py — Model Serving untuk Klasifikasi Sampah
=====================================================
Flask-based serving endpoint yang memuat model TensorFlow
dari MLflow dan menyediakan endpoint untuk prediksi.

Endpoints:
  GET  /ping          — Health check
  POST /invocations   — Prediksi dari JSON payload
  POST /predict_image — Prediksi dari file gambar (multipart/form-data)
"""

import os
import io
import base64

import mlflow.pyfunc
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

# ─── Configuration ───
import pathlib
mlruns_dir = (pathlib.Path(__file__).parent.parent / "2-Membangun_model" / "mlruns").resolve()
import mlflow
mlflow.set_tracking_uri(f"file:{mlruns_dir}")

MODEL_URI = os.getenv("MODEL_URI", "models:/SampahClassifier/5")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Kaca", "Kardus", "Kertas", "Logam", "Plastik", "Residu"]

# ─── Load Model ───
print(f"Loading model dari: {MODEL_URI}")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("Model berhasil dimuat!")


def preprocess_image(image_bytes):
    """Preprocess gambar untuk input model."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # batch dimension
    return arr


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_uri": MODEL_URI})


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Prediksi dari JSON payload.

    Format JSON yang diterima:
    1. {"instances": [[...]]}   — array of preprocessed images
    2. {"image_base64": "..."}  — base64 encoded image
    """
    payload = request.get_json()
    if payload is None:
        return jsonify({"error": "Missing JSON payload."}), 400

    try:
        if "image_base64" in payload:
            img_bytes = base64.b64decode(payload["image_base64"])
            features = preprocess_image(img_bytes)
        elif "instances" in payload:
            features = np.array(payload["instances"], dtype=np.float32)
        else:
            return jsonify({"error": "Payload harus berisi 'instances' atau 'image_base64'."}), 400

        predictions = model.predict(features)
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()

        results = []
        for pred in predictions:
            class_idx = int(np.argmax(pred))
            results.append({
                "class": CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx),
                "confidence": float(max(pred)),
                "probabilities": pred if isinstance(pred, list) else list(pred),
            })

        return jsonify({"predictions": results})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/predict_image", methods=["POST"])
def predict_image():
    """Prediksi dari file gambar (multipart/form-data)."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Gunakan key 'file'."}), 400

    try:
        file = request.files["file"]
        img_bytes = file.read()
        features = preprocess_image(img_bytes)
        predictions = model.predict(features)

        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()

        pred = predictions[0]
        class_idx = int(np.argmax(pred))

        return jsonify({
            "prediction": {
                "class": CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx),
                "confidence": float(max(pred)),
                "probabilities": pred if isinstance(pred, list) else list(pred),
            }
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005)
