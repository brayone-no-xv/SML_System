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
	["status_code"],
)
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP Request Latency")
UPSTREAM_LATENCY = Histogram("model_upstream_latency_seconds", "Latency to model API")
PREDICT_ERRORS = Counter("model_predict_errors_total", "Total prediction errors")

# Metrik untuk sistem
CPU_USAGE = Gauge("system_cpu_usage", "CPU Usage Percentage")  # Penggunaan CPU
RAM_USAGE = Gauge("system_ram_usage", "RAM Usage Percentage")  # Penggunaan RAM

MODEL_API_URL = os.getenv("MODEL_API_URL", "http://127.0.0.1:5005/invocations")


# Endpoint untuk Prometheus
@app.route("/", methods=["GET"])
def index():
	return jsonify({"status": "ok", "metrics": "/metrics", "predict": "/predict"})


@app.route("/metrics", methods=["GET"])
def metrics():
	# Update metrik sistem setiap kali /metrics diakses
	CPU_USAGE.set(psutil.cpu_percent(interval=1))  # Ambil data CPU usage (persentase)
	RAM_USAGE.set(psutil.virtual_memory().percent)  # Ambil data RAM usage (persentase)

	return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# Endpoint untuk mengakses API model dan mencatat metrik
@app.route("/predict", methods=["POST"])
def predict():
	start_time = time.time()

	data = request.get_json()
	upstream_start = time.time()

	try:
		response = requests.post(MODEL_API_URL, json=data, timeout=10)
		response.raise_for_status()
		REQUEST_COUNT.labels(status_code=str(response.status_code)).inc()
		return jsonify(response.json())
	except Exception as e:
		REQUEST_COUNT.labels(status_code="error").inc()
		PREDICT_ERRORS.inc()
		return jsonify({"error": str(e)}), 500
	finally:
		REQUEST_LATENCY.observe(time.time() - start_time)
		UPSTREAM_LATENCY.observe(time.time() - upstream_start)


if __name__ == "__main__":
	app.run(host="127.0.0.1", port=8000)