from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "1-Preprocessing" / "dataset" / "apbd_data_2026.csv"
PREPROCESSED_PATH = Path(__file__).resolve().parent / "apbd-dataset-2026" / "apbd_data_2026_preprocessed.csv"
TARGET_COLUMN = "Persentase"


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


def train():
	mlflow.set_tracking_uri("file:./mlruns")
	mlflow.set_experiment("apbd-2026-basic")
	mlflow.sklearn.autolog(log_models=True)

	dataframe, is_preprocessed = load_dataset()
	if TARGET_COLUMN not in dataframe.columns:
		raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

	X = dataframe.drop(columns=[TARGET_COLUMN])
	y = dataframe[TARGET_COLUMN]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.3, random_state=42
	)

	if is_preprocessed:
		model = RandomForestRegressor(
			n_estimators=200, max_depth=12, random_state=42
		)
		with mlflow.start_run():
			mlflow.log_param("dataset_path", str(PREPROCESSED_PATH))
			mlflow.log_param("is_preprocessed", True)
			model.fit(X_train, y_train)
			preds = model.predict(X_test)
	else:
		preprocessor = build_preprocessor(X_train)
		model = RandomForestRegressor(
			n_estimators=200, max_depth=12, random_state=42
		)
		pipeline = Pipeline(
			steps=[
				("preprocess", preprocessor),
				("model", model),
			]
		)
		with mlflow.start_run():
			mlflow.log_param("dataset_path", str(RAW_DATA_PATH))
			mlflow.log_param("is_preprocessed", False)
			pipeline.fit(X_train, y_train)
			preds = pipeline.predict(X_test)

	rmse = mean_squared_error(y_test, preds, squared=False)
	mae = mean_absolute_error(y_test, preds)
	r2 = r2_score(y_test, preds)

	mlflow.log_metrics({
		"rmse": rmse,
		"mae": mae,
		"r2": r2,
	})


if __name__ == "__main__":
	train()
