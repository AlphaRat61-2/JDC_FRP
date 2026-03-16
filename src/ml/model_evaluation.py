from __future__ import annotations

import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.paths import project_root
from src.io.writers import write_table
from src.ml.model_features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN


def _ml_dir():
    path = project_root() / "outputs" / "ml"
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_model(settings, logger, batch, model, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ml_dir = _ml_dir()

    if model is None or test_df.empty:
        logger.warning("Model evaluation skipped due to missing model or empty test set.")
        return pd.DataFrame(), pd.DataFrame()

    numeric_cols = [c for c in FEATURE_COLUMNS if c in test_df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in test_df.columns]
    feature_cols = numeric_cols + categorical_cols

    eval_df = test_df.dropna(subset=[TARGET_COLUMN]).copy()
    if eval_df.empty:
        logger.warning("No labeled rows available for evaluation.")
        return pd.DataFrame(), pd.DataFrame()

    X_test = eval_df[feature_cols]
    y_true = eval_df[TARGET_COLUMN].astype(int)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else None
    avg_prec = average_precision_score(y_true, y_prob) if y_true.nunique() > 1 else None

    perf = pd.DataFrame(
        [
            {"metric": "roc_auc", "value": roc_auc},
            {"metric": "average_precision", "value": avg_prec},
            {"metric": "precision_at_0_5", "value": precision_score(y_true, y_pred, zero_division=0)},
            {"metric": "recall_at_0_5", "value": recall_score(y_true, y_pred, zero_division=0)},
            {"metric": "f1_at_0_5", "value": f1_score(y_true, y_pred, zero_division=0)},
        ]
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    confusion = pd.DataFrame(
        [
            {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "threshold": 0.5,
            }
        ]
    )

    write_table(perf, ml_dir, "ml_model_performance", settings)
    write_table(confusion, ml_dir, "ml_model_confusion", settings)

    batch.set_row_count("ml_model_performance", len(perf))
    batch.set_row_count("ml_model_confusion", len(confusion))
    logger.info("Built ML evaluation outputs.")
    return perf, confusion
