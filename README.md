# ♻️ SampahClassifier - End-to-End MLOps System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange.svg)](https://tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.19.0-blue.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

Proyek ini adalah implementasi sistem **Machine Learning Operations (MLOps)** secara penuh (*End-to-End*) untuk mengklasifikasikan jenis sampah daur ulang. Model Deep Learning (MobileNetV2) yang dibangun mampu membedakan 6 kelas sampah: **Kaca, Kardus, Kertas, Logam, Plastik, dan Residu**.

Proyek ini disusun sebagai **Tugas Akhir / Submission** untuk kelas **Belajar Penerapan Machine Learning dengan MLOps** di **Dicoding Academy**, dengan tujuan memenuhi kriteria kelulusan tingkat **Lanjut (Advance/Excellent)**.

---

## 🛠️ Teknologi yang Digunakan
Sistem ini memadukan berbagai *stack* teknologi standar industri:
*   **Modeling Framework:** TensorFlow & Keras (Functional API)
*   **Experiment Tracking & Registry:** MLflow
*   **CI/CD Pipeline:** GitHub Actions & MLflow Project (`MLproject`)
*   **Containerization:** Docker & Docker Hub
*   **Model Serving:** Flask REST API
*   **Monitoring & Observability:** Prometheus (TSDB) & Grafana (Alerting & Visualization)

---

## 📂 Struktur Proyek Terintegrasi

```text
SML_System_Muhammad_Rahman/
├── 1-Preprocessing/          # Script otomatisasi penyiapan dataset dan Exploratory Data Analysis (EDA)
├── 2-Membangun_model/        # Arsitektur model, tuning, dan MLflow tracking (berisi folder mlruns/)
├── 3-Workflow-CI/            # Konfigurasi MLProject dan pipeline CI/CD (GitHub Actions)
├── 4-Monitoring_dan_Logging/ # Sistem Gateway API, Prometheus Exporter, dan script testing otomatis
├── mlflow-dockerfile/        # Artefak Docker hasil ekspor dari MLflow Models
├── .github/workflows/        # Script otomatisasi GitHub Actions (CI/CD)
├── requirements.txt          # Daftar dependensi library Python
└── README.md                 # Dokumentasi utama proyek
```

---

## 🚀 Fitur & Pencapaian Utama

### 1. 📊 Preprocessing & Automatisasi Modeling
*   Dataset otomatis diunduh melalui *Kaggle API* (`fathurrahmanalfarizy/sampah-daur-ulang`).
*   Model menggunakan arsitektur **MobileNetV2 (Transfer Learning)** dengan teknik *Fine-Tuning* dan lapisan *Data Augmentation* dinamis.
*   Seluruh metrik pelatihan (Akurasi, Loss), parameter (Epochs, Batch Size), dan model biner (`.keras`) dicatat secara otomatis menggunakan **MLflow Tracking**.

### 2. ⚙️ Otomatisasi CI/CD (GitHub Actions)
Proyek ini mengadopsi standar eksekusi `MLproject`. Melalui GitHub Actions (`main.yml`), setiap perubahan kode yang masuk ke *branch* `main` akan memicu *pipeline* otomatis yang:
1. Mengunduh dataset dan men-*training* ulang model.
2. Mencatat model terbaru ke registri MLflow.
3. Men-*generate* wadah (*container*) menggunakan `mlflow models build-docker`.
4. Mengunggah (*push*) *image* secara otonom ke registri **Docker Hub**.

### 3. 🌐 Model Serving Instan & API Gateway
Model diekspos melalui **Flask REST API** (`inference.py` di port `5005`). Untuk keamanan dan analitik, terdapat sebuah **Eksportir Prometheus / API Gateway** (`prometheus_exporter.py` di port `8000`) yang bertugas mencegat (*intercept*) seluruh trafik HTTP sebelum masuk ke *Inference Server*.

### 4. 📈 Observability (Pemantauan & Peringatan Dini)
Gateway secara waktu nyata mengekspor berbagai metrik krusial ke **Prometheus**:
*   **Metrik Sistem:** Penggunaan CPU (`process_cpu_seconds_total`) dan penggunaan RAM (`process_resident_memory_bytes`).
*   **Metrik API:** Total Request, Waktu Latensi Respons, dan *Error Rate* (Kode 4xx/5xx).
Data ini divisualisasikan dalam bentuk dasbor interaktif di **Grafana**, yang dilengkapi dengan sistem **Alerting** jika terjadi anomali (misalnya: *CPU Usage* di atas batas wajar atau *Error Rate* tinggi).

---

## 🏃‍♂️ Panduan Evaluasi untuk Reviewer (Cara Menjalankan)

Proyek ini telah dikondisikan sedemikian rupa agar sangat ringan dan ramah pengujian (*Clean Workspace*). Anda **tidak perlu** melakukan proses *training* yang memakan waktu lama, karena seluruh *database* MLflow (`mlflow.db`) dan model `.keras` versi terakhir sudah disematkan utuh di dalam folder `mlruns/`.

**1. Install Dependensi Lingkungan**
```bash
pip install -r requirements.txt
```

**2. Jalankan Inference API (Port 5005)**
*Buka terminal 1, jalankan perintah ini:*
```bash
python 4-Monitoring_dan_Logging/inference.py
```

**3. Jalankan Prometheus Gateway / Exporter (Port 8000)**
*Buka terminal 2, jalankan perintah ini:*
```bash
python 4-Monitoring_dan_Logging/prometheus_exporter.py
```

**4. Buktikan Serving Secara Langsung!**
*Buka terminal 3, jalankan perintah script tester otomatis ini:*
```bash
python 4-Monitoring_dan_Logging/test_serving.py
```
*Script ini akan menciptakan sebuah gambar secara in-memory dan mengirimkannya ke API Gateway. Anda akan langsung melihat balasan `Status Code: 200` beserta tebakan probabilitas kelas sampah dari model!*

**5. Opsional: Jalankan UI MLflow**
Jika ingin melihat riwayat eksperimen dan parameter:
```bash
cd 2-Membangun_model
mlflow ui --port 5000
```

---

**🎓 Developed by:** Muhammad Rahman  
*Submission Kelas Belajar Penerapan Machine Learning dengan MLOps - Dicoding Indonesia.*
