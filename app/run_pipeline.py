from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


from src.common.batch import close_batch_context, create_batch_context
from src.common.logging_utils import setup_logger
from src.common.paths import load_settings
from src.io.batch_history import append_batch_history
from src.io.file_registry import detect_incoming_files

from src.staging.stage_master_well import stage_master_well
from src.staging.stage_production import stage_production
from src.staging.stage_chemical_rates import stage_chemical_rates
from src.staging.stage_chemical_cost import stage_chemical_cost
from src.staging.stage_failures import stage_failures
from src.staging.stage_workovers import stage_workovers
from src.staging.stage_scada import stage_scada

from src.dimensions.build_dim_date import build_dim_date
from src.dimensions.build_dim_well import build_dim_well
from src.dimensions.build_dim_chemical import build_dim_chemical

from src.production.build_fact_production_daily import build_fact_production_daily
from src.failures.build_fact_failure_event import build_fact_failure_event
from src.workovers.build_fact_workover_event import build_fact_workover_event

from src.chemistry.build_expected_chem_daily import build_expected_chem_daily
from src.chemistry.target_resolution import build_fact_chem_target_daily
from src.chemistry.actual_builder import build_fact_chem_actual_daily
from src.reconciliation.build_fact_chem_recon_daily import build_fact_chem_recon_daily

from src.scada.build_fact_scada_daily import build_fact_scada_daily

from src.month_close.month_status import build_fact_month_status

from src.ml.feature_builder import build_ml_feature_well_daily
from src.ml.label_builder import build_ml_label_failure

from src.reporting.build_rpt_chemistry import build_rpt_chemistry
from src.reporting.build_rpt_scada import build_rpt_scada
from src.reporting.build_rpt_bad_actors import build_rpt_bad_actors
from src.reporting.build_rpt_month_status import build_rpt_month_status
from src.reporting.build_rpt_ml_readiness import build_rpt_ml_readiness
from src.reporting.build_rpt_run_status import build_rpt_run_status
from src.reporting.build_rpt_workover import build_rpt_workover
from src.reporting.build_rpt_top_risk_wells import build_rpt_top_risk_wells


def main() -> None:
    settings = load_settings()
    logger = setup_logger(settings)
    batch = create_batch_context(settings, run_type="FULL_PIPELINE", logger=logger)

    try:
        detect_incoming_files(settings, logger, batch)

        stage_master_well(settings, logger, batch)
        stage_production(settings, logger, batch)
        stage_chemical_rates(settings, logger, batch)
        stage_chemical_cost(settings, logger, batch)
        stage_failures(settings, logger, batch)
        stage_workovers(settings, logger, batch)
        stage_scada(settings, logger, batch)

        build_dim_date(settings, logger, batch)
        build_dim_well(settings, logger, batch)
        build_dim_chemical(settings, logger, batch)

        build_fact_production_daily(settings, logger, batch)
        build_fact_failure_event(settings, logger, batch)
        build_fact_workover_event(settings, logger, batch)

        build_fact_chem_target_daily(settings, logger, batch)
        build_expected_chem_daily(settings, logger, batch)
        build_fact_chem_actual_daily(settings, logger, batch)
        build_fact_chem_recon_daily(settings, logger, batch)

        build_fact_scada_daily(settings, logger, batch)

        build_fact_month_status(settings, logger, batch)

        build_ml_feature_well_daily(settings, logger, batch)
        build_ml_label_failure(settings, logger, batch)

        build_rpt_chemistry(settings, logger, batch)
        build_rpt_scada(settings, logger, batch)
        build_rpt_bad_actors(settings, logger, batch)
        build_rpt_workover(settings, logger, batch)
        build_rpt_month_status(settings, logger, batch)
        build_rpt_ml_readiness(settings, logger, batch)

        close_batch_context(batch, logger, status="SUCCESS")
        append_batch_history(settings, batch, logger)

        if settings.get("runtime", {}).get("write_run_status", True):
            build_rpt_run_status(settings, logger, batch)

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        close_batch_context(batch, logger, status="FAILED")
        append_batch_history(settings, batch, logger)
        if settings.get("runtime", {}).get("write_run_status", True):
            build_rpt_run_status(settings, logger, batch)
        raise


if __name__ == "__main__":
    main()


