from __future__ import annotations

import pandas as pd

from src.common.paths import get_path, project_root
from src.io.writers import write_table


def build_rpt_ml_readiness(settings, logger, batch) -> pd.DataFrame:
    reports_dir = get_path(settings, "reports")
    ml_dir = project_root() / "outputs" / "ml"

    feat_path = ml_dir / "ml_feature_well_daily.csv"
    label_path = ml_dir / "ml_label_failure.csv"

    rows: list[dict] = []

    if feat_path.exists():
        feat = pd.read_csv(feat_path, dtype={"well_id": str})
        rows.append(
            {
                "dataset_name": "ml_feature_well_daily",
                "row_count": len(feat),
                "well_count": feat["well_id"].nunique() if "well_id" in feat.columns else 0,
                "date_min": feat["date"].min() if "date" in feat.columns else None,
                "date_max": feat["date"].max() if "date" in feat.columns else None,
            }
        )

    if label_path.exists():
        lab = pd.read_csv(label_path, dtype={"well_id": str})
        rows.append(
            {
                "dataset_name": "ml_label_failure",
                "row_count": len(lab),
                "well_count": lab["well_id"].nunique() if "well_id" in lab.columns else 0,
                "date_min": lab["date"].min() if "date" in lab.columns else None,
                "date_max": lab["date"].max() if "date" in lab.columns else None,
            }
        )

    rpt = pd.DataFrame(rows)
    write_table(rpt, reports_dir, "rpt_ml_readiness", settings)
    batch.set_row_count("rpt_ml_readiness", len(rpt))
    logger.info("Built rpt_ml_readiness | rows=%s", len(rpt))
    return rpt
