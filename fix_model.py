import os
import tensorflow as tf
import mlflow
from mlflow.models import infer_signature
from pathlib import Path
import numpy as np

# Monkey-patch RandomRotation to ignore value_range from config
old_init = tf.keras.layers.RandomRotation.__init__
def new_init(self, factor, fill_mode='reflect', interpolation='bilinear', seed=None, data_format=None, value_range=None, **kwargs):
    old_init(self, factor=factor, fill_mode=fill_mode, interpolation=interpolation, seed=seed, data_format=data_format, **kwargs)
tf.keras.layers.RandomRotation.__init__ = new_init

# Tracking URI
mlruns_dir = Path("2-Membangun_model/mlruns").resolve()
mlflow.set_tracking_uri(f"file:{mlruns_dir}")
run_id = "8c0652cce4f2421b8e65c23d57ec3ced"
model_uri = f"runs:/{run_id}/model"

print("Loading model...")
model = mlflow.tensorflow.load_model(model_uri)

print("Removing data augmentation layer and rebuilding graph...")
# Functional API rebuild to prevent shape mismatch bugs
inputs = tf.keras.Input(shape=(224, 224, 3))
x = inputs
for layer in model.layers[1:]:  # Skip data_aug
    x = layer(x)
new_model = tf.keras.Model(inputs=inputs, outputs=x)

print("Generating signature...")
input_example = np.random.rand(1, 224, 224, 3).astype(np.float32)
predictions = new_model.predict(input_example)
signature = infer_signature(input_example, predictions)

print("Saving fixed model...")
mlflow.set_experiment("sampah-daur-ulang-tuning")
with mlflow.start_run(run_name="fixed_deployment_model"):
    mlflow.tensorflow.log_model(
        new_model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name="SampahClassifier",
    )
print("Fix completed! New version registered.")
