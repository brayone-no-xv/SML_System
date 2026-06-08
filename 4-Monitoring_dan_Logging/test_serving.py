import requests

# Konfigurasi
URL = "http://127.0.0.1:8000/predict_image"
IMAGE_PATH = "sampah_test.jpg"  # Pastikan file ini ada di komputer Anda!

print(f"Mengirim gambar {IMAGE_PATH} ke API...")

try:
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (IMAGE_PATH, f, "image/jpeg")}
        response = requests.post(URL, files=files)

    print("\n--- STATUS API ---")
    print(f"Status Code: {response.status_code}")
    
    print("\n--- HASIL PREDIKSI ---")
    print(response.json())

except FileNotFoundError:
    print(f"EROR: File {IMAGE_PATH} tidak ditemukan! Silakan siapkan satu gambar sampah (misal: botol plastik) dan beri nama sampah_test.jpg di folder ini.")
except Exception as e:
    print(f"EROR KONEKSI: {e}")
