from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.batch import close_batch_context, create_batch_context
from src.common.logging_utils import setup_logger
from src.common.paths import load_settings
from src.io.batch_history import append_batch_history

from src.ml.baseline_model import train_baseline_model
from src.ml.feature_importance import build_model_feature_importance
from src.ml.model_evaluation import evaluate_model
from src.ml.model_scoring import score_failure_risk
from src.ml.train_test_splitter import build_training_dataset, split_train_test_by_time
from src.reporting.build_rpt_failure_risk import build_rpt_failure_risk
from src.reporting.build_rpt_prediction_summary import build_rpt_prediction_summary
from src.reporting.build_rpt_run_status import build_rpt_run_status


def main() -> None:
    settings = load_settings()
    logger = setup_logger(settings)
    batch = create_batch_context(settings, run_type="ML_PREP", logger=logger)

    try:
        ds = build_training_dataset(settings, logger, batch)
        train_df, test_df = split_train_test_by_time(ds, test_fraction=0.2)

        model = train_baseline_model(settings, logger, batch, train_df)
        evaluate_model(settings, logger, batch, model, test_df)
        build_model_feature_importance(settings, logger, batch, model)
        score_failure_risk(settings, logger, batch)
        build_rpt_failure_risk(settings, logger, batch)
        build_rpt_prediction_summary(settings, logger, batch)

        close_batch_context(batch, logger, status="SUCCESS")
        append_batch_history(settings, batch, logger)

        if settings.get("runtime", {}).get("write_run_status", True):
            build_rpt_run_status(settings, logger, batch)

    except Exception as exc:
        logger.exception("ML pipeline failed: %s", exc)
        close_batch_context(batch, logger, status="FAILED")
        append_batch_history(settings, batch, logger)
        if settings.get("runtime", {}).get("write_run_status", True):
            build_rpt_run_status(settings, logger, batch)
        raise


if __name__ == "__main__":
    main()

