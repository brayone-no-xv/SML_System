"""
modelling_tuning.py — Model Building & Hyperparameter Tuning
=============================================================
Deep learning image classification (MobileNetV2 Transfer Learning)
untuk dataset Sampah Daur Ulang.

Terintegrasi dengan pipeline preprocessing di 1-Preprocessing/automate_Muhammad_Rahman.py
dan logging manual MLflow.

Usage:
    python modelling_tuning.py
"""

import json
import sys
from pathlib import Path

import mlflow
from mlflow.models import infer_signature
import numpy as np
import tensorflow as tf

# ======================== PATH SETUP ========================
PROJECT_DIR = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = PROJECT_DIR / "1-Preprocessing"

# Add preprocessing directory to path for import
sys.path.insert(0, str(PREPROCESS_DIR))
from automate_Muhammad_Rahman import preprocess, make_data_augmentation  # noqa: E402

# ======================== CONFIGURATION ========================
SEED = 42
IMG_SIZE = (224, 224)
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ======================== MODEL ========================

def build_model(num_classes, learning_rate=1e-3, dropout=0.3, dense_units=256):
    """Build MobileNetV2 Sequential model with transfer learning."""
    data_aug = make_data_augmentation()

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.Input(shape=IMG_SIZE + (3,)),
        data_aug,
        tf.keras.layers.Rescaling(1./127.5, offset=-1, name="preprocess_input"),
        base_model,
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="post_conv"),
        tf.keras.layers.MaxPooling2D(name="post_pool"),
        tf.keras.layers.GlobalAveragePooling2D(name="post_gap"),
        tf.keras.layers.BatchNormalization(name="post_bn1"),
        tf.keras.layers.Dropout(dropout, name="post_dropout1"),
        tf.keras.layers.Dense(dense_units, activation="relu", name="post_dense1"),
        tf.keras.layers.BatchNormalization(name="post_bn2"),
        tf.keras.layers.Dropout(dropout, name="post_dropout2"),
        tf.keras.layers.Dense(dense_units // 2, activation="relu", name="post_dense2"),
        tf.keras.layers.Dropout(dropout * 0.67, name="post_dropout3"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="post_logits"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def evaluate_metrics(model, dataset, name="eval"):
    """Evaluate model on a dataset and return metrics dict."""
    loss, acc = model.evaluate(dataset, verbose=1)
    return {f"{name}_loss": loss, f"{name}_accuracy": acc}


# ======================== TRAINING ========================

def train_with_tuning():
    """Two-phase training with MLflow manual logging."""
    # Tracking URI: simpan mlruns di folder 2-Membangun_model/ (sejajar dengan script)
    MLRUNS_DIR = Path(__file__).resolve().parent / "mlruns"
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment("sampah-daur-ulang-tuning")

    # Load data from preprocessing pipeline
    train_ds, val_ds, test_ds, class_names, config = preprocess()
    num_classes = config["NUM_CLASSES"]

    # Hyperparameter grid (lightweight for local training)
    param_configs = [
        {"learning_rate": 1e-3, "dropout": 0.3, "dense_units": 256, "epochs_p1": 15, "epochs_p2": 20},
        {"learning_rate": 5e-4, "dropout": 0.4, "dense_units": 128, "epochs_p1": 15, "epochs_p2": 20},
    ]

    best_run = {"val_accuracy": 0.0, "params": None}

    with mlflow.start_run(run_name="tuning_parent") as parent_run:
        mlflow.log_param("architecture", "Sequential_MobileNetV2")
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("class_names", str(class_names))
        mlflow.log_param("img_size", str(IMG_SIZE))
        mlflow.log_param("total_configs", len(param_configs))

        for cfg_idx, params in enumerate(param_configs):
            print(f"\n{'='*60}")
            print(f"CONFIG {cfg_idx+1}/{len(param_configs)}: {params}")
            print(f"{'='*60}")

            with mlflow.start_run(nested=True, run_name=f"config_{cfg_idx+1}"):
                mlflow.log_params(params)

                model, base_model = build_model(
                    num_classes,
                    learning_rate=params["learning_rate"],
                    dropout=params["dropout"],
                    dense_units=params["dense_units"],
                )

                # Phase 1: Head training (base frozen)
                callbacks_p1 = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_accuracy", patience=5, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
                    ),
                ]
                h1 = model.fit(
                    train_ds, validation_data=val_ds,
                    epochs=params["epochs_p1"], callbacks=callbacks_p1,
                )

                # Phase 2: Fine-tuning top layers
                base_model.trainable = True
                fine_tune_at = max(0, len(base_model.layers) - 50)
                for layer in base_model.layers[:fine_tune_at]:
                    layer.trainable = False

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-5),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )

                callbacks_p2 = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_accuracy", patience=8, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=str(CHECKPOINT_DIR / f"best_config_{cfg_idx+1}.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=1,
                    ),
                ]
                h2 = model.fit(
                    train_ds, validation_data=val_ds,
                    epochs=params["epochs_p2"], callbacks=callbacks_p2,
                )

                # Log metrics per epoch
                combined = {
                    k: h1.history[k] + h2.history[k]
                    for k in h1.history
                }
                for i in range(len(combined["loss"])):
                    mlflow.log_metric("train_loss", combined["loss"][i], step=i)
                    mlflow.log_metric("train_accuracy", combined["accuracy"][i], step=i)
                    mlflow.log_metric("val_loss", combined["val_loss"][i], step=i)
                    mlflow.log_metric("val_accuracy", combined["val_accuracy"][i], step=i)

                # Test evaluation
                test_metrics = evaluate_metrics(model, test_ds, "test")
                mlflow.log_metrics(test_metrics)

                final_val_acc = max(combined["val_accuracy"])
                mlflow.log_metric("best_val_accuracy", final_val_acc)

                # Log model with signature and input example
                # Ambil satu batch dari test_ds sebagai contoh input
                for sample_batch, _ in test_ds.take(1):
                    input_example = sample_batch[:1].numpy()
                    break
                predictions = model.predict(input_example)
                signature = infer_signature(input_example, predictions)
                mlflow.tensorflow.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name="SampahClassifier",
                    extra_pip_requirements=["gunicorn", "keras==3.3.3"],
                )

                # Track best
                if final_val_acc > best_run["val_accuracy"]:
                    best_run = {
                        "val_accuracy": final_val_acc,
                        "params": params,
                        "test_metrics": test_metrics,
                        "config_idx": cfg_idx + 1,
                    }
                    model.save(str(CHECKPOINT_DIR / "best_model.keras"))

        # Log best run summary to parent
        mlflow.log_params({f"best_{k}": v for k, v in best_run["params"].items()})
        mlflow.log_metric("best_val_accuracy", best_run["val_accuracy"])
        for k, v in best_run["test_metrics"].items():
            mlflow.log_metric(f"best_{k}", v)
        mlflow.log_text(json.dumps(best_run, indent=2, default=str), "best_run_summary.json")

        print(f"\n{'='*60}")
        print(f"BEST CONFIG: #{best_run['config_idx']}")
        print(f"  Val Accuracy : {best_run['val_accuracy']:.4f}")
        print(f"  Test Metrics : {best_run['test_metrics']}")
        print(f"  Params       : {best_run['params']}")
        print(f"{'='*60}")


if __name__ == "__main__":
    train_with_tuning()