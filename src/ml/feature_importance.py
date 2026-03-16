from __future__ import annotations

import pandas as pd

from src.common.paths import project_root
from src.io.writers import write_table


def _ml_dir():
    path = project_root() / "outputs" / "ml"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_model_feature_importance(settings, logger, batch, model) -> pd.DataFrame:
    ml_dir = _ml_dir()

    if model is None:
        logger.warning("No model available for feature importance output.")
        return pd.DataFrame()

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
        coefs = classifier.coef_[0]
        imp = pd.DataFrame(
            {
                "feature_name": feature_names,
                "coefficient": coefs,
                "abs_coefficient": abs(coefs),
            }
        ).sort_values("abs_coefficient", ascending=False)
    except Exception as exc:
        logger.warning("Could not derive feature importance: %s", exc)
        return pd.DataFrame()

    write_table(imp, ml_dir, "ml_feature_importance", settings)
    batch.set_row_count("ml_feature_importance", len(imp))
    logger.info("Built ml_feature_importance | rows=%s", len(imp))
    return imp
