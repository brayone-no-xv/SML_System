import mlflow
import requests
from sklearn import load_model

run_id = "feb37af613904b6886705eaf0576879b"
# model_uri = f"runs://{run_id}/model"
data = f"runs:/{run_id}/model/"

model_registry = f"models:/{model_name}/{model_version}"

# Run prompt 
# If you want to serve a *registered* model, use `models:/` (not `runs:/`).
# Use a valid env manager: one of `local`, `conda`, `virtualenv`, or `uv`.
# Example (uses your currently active environment):
# mlflow models serve -m models:/sml_system/1 -p 5000 --env-manager local
# Example (serve a specific run artifact path):
# mlflow models serve -m runs:/feb37af613904b6886705eaf0576879b/best_model -p 5000 --env-manager local

# mlflow models generate-dockerfile -m models:/sml_system/1 -o test_docker



url = "http://127.0.0.1:5000/invocations"  # endpoint REST API model, bukan URL UI
headers = {"Content-Type": "application/json"}
data = '{"data": [[1, 2, 3, 4]]}'  # contoh data, sesuaikan dengan input model Anda

response = requests.post(url, data=data, headers=headers)
predictions = response.json().get("predictions")
print(predictions)