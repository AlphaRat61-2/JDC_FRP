from __future__ import annotations

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.paths import project_root
from src.ml.model_features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN


def _ml_dir():
    path = project_root() / "outputs" / "ml"
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_baseline_model(settings, logger, batch, train_df: pd.DataFrame):
    ml_dir = _ml_dir()

    if train_df.empty:
        logger.warning("Training dataset is empty.")
        return None

    numeric_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in train_df.columns]
    feature_cols = numeric_cols + categorical_cols

    model_df = train_df[feature_cols + [TARGET_COLUMN]].copy()
    model_df = model_df.dropna(subset=[TARGET_COLUMN])

    if model_df.empty:
        logger.warning("No labeled rows available for model training.")
        return None

    X = model_df[feature_cols]
    y = model_df[TARGET_COLUMN].astype(int)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )

    model.fit(X, y)

    model_path = ml_dir / "baseline_failure_model.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved baseline model | path=%s", model_path)
    return model