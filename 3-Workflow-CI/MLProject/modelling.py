import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_TARGET = "Persentase"
DEFAULT_PREPROCESSED = "apbd-dataset-2026/apbd_data_2026_preprocessed.csv"
DEFAULT_RAW = "../1-Preprocessing/dataset/apbd_data_2026.csv"


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


def load_dataset(data_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(data_path)
    numeric_columns = dataframe.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_like = detect_numeric_like_columns(dataframe)
    for col in numeric_like:
        if col not in numeric_columns:
            numeric_columns.append(col)
    return normalize_numeric_columns(dataframe, numeric_columns)


def train(data_path: str, target_column: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("apbd-2026-ci")

    dataframe = load_dataset(data_path)
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    dataframe = dataframe.dropna(subset=[target_column])
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    with mlflow.start_run():
        mlflow.log_param("dataset_path", data_path)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        metrics = evaluate_metrics(y_test, preds)
        mlflow.log_metrics(metrics)

        input_example = X_test.head(5)
        signature = infer_signature(input_example, preds[:5])

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_PREPROCESSED)
    parser.add_argument("--target-column", default=DEFAULT_TARGET)
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    return parser.parse_args()


def resolve_data_path(data_path: str) -> str:
    candidate = Path(data_path)
    if candidate.exists():
        return str(candidate)
    fallback = Path(DEFAULT_RAW)
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("Dataset not found in MLProject or raw dataset path.")


def main():
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    train(data_path, args.target_column, args.tracking_uri)


if __name__ == "__main__":
    main()
