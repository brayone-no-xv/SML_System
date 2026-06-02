from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "1-Preprocessing" / "dataset" / "apbd_data_2026.csv"
PREPROCESSED_PATH = Path(__file__).resolve().parent / "apbd-dataset-2026" / "apbd_data_2026_preprocessed.csv"
TARGET_COLUMN = "Persentase"


def normalize_numeric_columns(dataframe: pd.DataFrame, columns):
    cleaned = dataframe.copy()
    for col in columns:
        cleaned[col] = (
            cleaned[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace("nan", pd.NA)
        )
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    return cleaned


def detect_numeric_like_columns(dataframe: pd.DataFrame, threshold: float = 0.9):
    numeric_like = []
    for col in dataframe.columns:
        if dataframe[col].dtype != "object":
            continue
        cleaned = (
            dataframe[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace("nan", pd.NA)
        )
        coerced = pd.to_numeric(cleaned, errors="coerce")
        non_null_ratio = coerced.notna().mean()
        if non_null_ratio >= threshold:
            numeric_like.append(col)
    return numeric_like


def load_dataset():
    if PREPROCESSED_PATH.exists():
        dataframe = pd.read_csv(PREPROCESSED_PATH)
        return dataframe, True
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Preprocessed dataset not found. Place it in apbd-dataset-2026/ or provide the raw dataset."
        )
    dataframe = pd.read_csv(RAW_DATA_PATH)
    return dataframe, False


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = features.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_with_tuning():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("apbd-2026-tuning")

    dataframe, is_preprocessed = load_dataset()
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    numeric_columns = dataframe.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_like_columns = detect_numeric_like_columns(dataframe)
    for col in numeric_like_columns:
        if col not in numeric_columns:
            numeric_columns.append(col)
    if TARGET_COLUMN not in numeric_columns:
        numeric_columns.append(TARGET_COLUMN)
    dataframe = normalize_numeric_columns(dataframe, numeric_columns)
    dataframe = dataframe.dropna(subset=[TARGET_COLUMN])

    if not is_preprocessed:
        PREPROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(PREPROCESSED_PATH, index=False)

    X = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    param_grid = ParameterGrid(
        {
            "n_estimators": [150, 300],
            "max_depth": [8, 12],
            "min_samples_split": [2, 5],
        }
    )

    best_run = {
        "rmse": float("inf"),
        "model": None,
        "params": None,
        "metrics": None,
        "signature": None,
        "input_example": None,
    }

    with mlflow.start_run(run_name="tuning") as parent_run:
        mlflow.log_param("dataset_path", str(PREPROCESSED_PATH if is_preprocessed else RAW_DATA_PATH))
        mlflow.log_param("is_preprocessed", is_preprocessed)

        for params in param_grid:
            with mlflow.start_run(nested=True):
                model = RandomForestRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    random_state=42,
                )

                has_categorical = X_train.select_dtypes(include=["object"]).shape[1] > 0
                if is_preprocessed and not has_categorical:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    trained_model = model
                else:
                    preprocessor = build_preprocessor(X_train)
                    pipeline = Pipeline(
                        steps=[
                            ("preprocess", preprocessor),
                            ("model", model),
                        ]
                    )
                    pipeline.fit(X_train, y_train)
                    preds = pipeline.predict(X_test)
                    trained_model = pipeline

                metrics = evaluate_metrics(y_test, preds)
                input_example = X_test.head(5)
                signature = infer_signature(input_example, preds[:5])

                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                    trained_model,
                    artifact_path="model",
                    input_example=input_example,
                    signature=signature,
                )

                if metrics["rmse"] < best_run["rmse"]:
                    best_run = {
                        "rmse": metrics["rmse"],
                        "model": trained_model,
                        "params": params,
                        "metrics": metrics,
                        "signature": signature,
                        "input_example": input_example,
                    }

        mlflow.log_params({f"best_{k}": v for k, v in best_run["params"].items()})
        mlflow.log_metrics({f"best_{k}": v for k, v in best_run["metrics"].items()})
        mlflow.sklearn.log_model(
            best_run["model"],
            artifact_path="best_model",
            input_example=best_run["input_example"],
            signature=best_run["signature"],
        )


if __name__ == "__main__":
    train_with_tuning()