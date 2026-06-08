from flask import Flask, request, jsonify, Response
import os
import requests
import time
import psutil  # Untuk monitoring sistem
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Metrik untuk API model
REQUEST_COUNT = Counter(
	"http_requests_total",
	"Total HTTP Requests",
	["status_code", "endpoint"],
)
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP Request Latency", ["endpoint"])
UPSTREAM_LATENCY = Histogram("model_upstream_latency_seconds", "Latency to model API", ["endpoint"])
PREDICT_ERRORS = Counter("model_predict_errors_total", "Total prediction errors", ["endpoint"])

# Metrik untuk sistem
CPU_USAGE = Gauge("system_cpu_usage", "CPU Usage Percentage")  # Penggunaan CPU
RAM_USAGE = Gauge("system_ram_usage", "RAM Usage Percentage")  # Penggunaan RAM

MODEL_API_URL = os.getenv("MODEL_API_URL", "http://127.0.0.1:5005")

# Endpoint untuk Prometheus
@app.route("/", methods=["GET"])
def index():
	return jsonify({"status": "ok", "metrics": "/metrics", "invocations": "/invocations", "predict_image": "/predict_image"})

@app.route("/metrics", methods=["GET"])
def metrics():
	# Update metrik sistem setiap kali /metrics diakses
	CPU_USAGE.set(psutil.cpu_percent(interval=1))  # Ambil data CPU usage (persentase)
	RAM_USAGE.set(psutil.virtual_memory().percent)  # Ambil data RAM usage (persentase)

	return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# Endpoint gateway untuk mengakses API model (JSON)
@app.route("/invocations", methods=["POST"])
def invocations():
	start_time = time.time()
	upstream_start = time.time()
	data = request.get_json()

	try:
		response = requests.post(f"{MODEL_API_URL}/invocations", json=data, timeout=10)
		response.raise_for_status()
		REQUEST_COUNT.labels(status_code=str(response.status_code), endpoint="/invocations").inc()
		return jsonify(response.json()), response.status_code
	except Exception as e:
		REQUEST_COUNT.labels(status_code="error", endpoint="/invocations").inc()
		PREDICT_ERRORS.labels(endpoint="/invocations").inc()
		return jsonify({"error": str(e)}), 500
	finally:
		REQUEST_LATENCY.labels(endpoint="/invocations").observe(time.time() - start_time)
		UPSTREAM_LATENCY.labels(endpoint="/invocations").observe(time.time() - upstream_start)

# Endpoint gateway untuk mengakses API model (Image File)
@app.route("/predict_image", methods=["POST"])
def predict_image():
	start_time = time.time()
	upstream_start = time.time()

	try:
		if "file" not in request.files:
			return jsonify({"error": "No file uploaded. Gunakan key 'file'."}), 400

		file = request.files["file"]
		files = {"file": (file.filename, file.read(), file.mimetype)}
		
		response = requests.post(f"{MODEL_API_URL}/predict_image", files=files, timeout=10)
		response.raise_for_status()
		
		REQUEST_COUNT.labels(status_code=str(response.status_code), endpoint="/predict_image").inc()
		return jsonify(response.json()), response.status_code
	except Exception as e:
		REQUEST_COUNT.labels(status_code="error", endpoint="/predict_image").inc()
		PREDICT_ERRORS.labels(endpoint="/predict_image").inc()
		return jsonify({"error": str(e)}), 500
	finally:
		REQUEST_LATENCY.labels(endpoint="/predict_image").observe(time.time() - start_time)
		UPSTREAM_LATENCY.labels(endpoint="/predict_image").observe(time.time() - upstream_start)

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8000)