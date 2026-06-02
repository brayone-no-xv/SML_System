from pathlib import Path
import pandas as pd
from automate_Muhammad_Rahman import preprocess_data

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(
    dataframe: pd.DataFrame,
    target_column: str,
    test_size: float = 0.3,
    random_state: int = 42,
    save_preprocessed_path: str | None = None,
):


    data = pd.read_csv("1-Preprocessing/dataset/apbd_data_2026.csv")
    preprocessed_path = Path("2-Membangun_model/namadataset_preprocessing/apbd_data_2026_preprocessed.csv")
    preprocess_data(data, "Persentase", save_preprocessed_path=str(preprocessed_path))

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    numeric_candidates = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_column not in numeric_candidates:
        numeric_candidates.append(target_column)
    dataframe = dataframe.copy()
    for col in numeric_candidates:
        dataframe[col] = (
            dataframe[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .replace('nan', pd.NA)
        )
        dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    dataframe = dataframe.dropna(subset=[target_column])

    if save_preprocessed_path:
        pd.DataFrame(dataframe).to_csv(save_preprocessed_path, index=False)

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

