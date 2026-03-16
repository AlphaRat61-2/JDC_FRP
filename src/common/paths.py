from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# CONFIG
# ============================================================

CONFIG_DIR = ROOT / "config"

SETTINGS_FILE = CONFIG_DIR / "settings.yaml"

CHEM_MAPPING_FILE = CONFIG_DIR / "chemical_mapping.csv"

ALERT_THRESHOLDS_FILE = CONFIG_DIR / "alert_thresholds.csv"

MAINT_RULES_FILE = CONFIG_DIR / "maintenance_rules.csv"

RL_ACTION_SPACE_FILE = CONFIG_DIR / "rl_action_space.csv"

FAILURE_CLASSES_FILE = CONFIG_DIR / "failure_classes.csv"


# ============================================================
# RAW DATA
# ============================================================

RAW_DATA_DIR = ROOT / "data" / "raw"

PRODUCTION_RAW = RAW_DATA_DIR / "production_data.csv"

MASTER_WELL_RAW = RAW_DATA_DIR / "master_well.csv"

CHEM_RATES_RAW = RAW_DATA_DIR / "chemical_rates.csv"

CHEM_COST_RAW = RAW_DATA_DIR / "chemical_cost.csv"

FAILURES_RAW = RAW_DATA_DIR / "failure_date.csv"

SCADA_RAW = RAW_DATA_DIR / "scada_data.csv"

WORKOVER_RAW = RAW_DATA_DIR / "workover_data.csv"


# ============================================================
# MODELED DATA
# ============================================================

MODELED_DIR = ROOT / "data" / "modeled"

FACT_PRODUCTION = MODELED_DIR / "fact_production_daily.csv"

FACT_SCADA = MODELED_DIR / "fact_scada_daily.csv"

FACT_FAILURE = MODELED_DIR / "fact_failure_event.csv"

FACT_CHEM_RECON = MODELED_DIR / "fact_chem_recon_daily.csv"

FACT_WORKOVER = MODELED_DIR / "fact_workover_event.csv"


# ============================================================
# REPORT OUTPUTS
# ============================================================

REPORT_DIR = ROOT / "data" / "reports"

RPT_PRODUCTION = REPORT_DIR / "rpt_production_summary.csv"

RPT_CHEMISTRY = REPORT_DIR / "rpt_chemistry_daily.csv"

RPT_FAILURE_RISK = REPORT_DIR / "rpt_failure_risk.csv"

RPT_PRESCRIPTIVE = REPORT_DIR / "rpt_prescriptive_actions.csv"

RPT_OPTIMIZATION = REPORT_DIR / "rpt_optimization_results.csv"

RPT_DIGITAL_TWIN = REPORT_DIR / "rpt_digital_twin.csv"

RPT_RL_RECOMMENDATIONS = REPORT_DIR / "rpt_rl_recommendations.csv"

RPT_RL_FEEDBACK = REPORT_DIR / "rpt_rl_feedback_loop.csv"


# ============================================================
# MACHINE LEARNING OUTPUTS
# ============================================================

ML_DIR = ROOT / "outputs" / "ml"

ML_FAILURE_MODEL = ML_DIR / "failure_model.pkl"

ML_FAILURE_CAUSE_MODEL = ML_DIR / "failure_cause_model.pkl"

ML_FAILURE_RISK = ML_DIR / "ml_failure_risk_scored.csv"

ML_FAILURE_CAUSE = ML_DIR / "ml_failure_cause_predictions.csv"

ML_PRESCRIPTIVE = ML_DIR / "ml_prescriptive_actions.csv"

ML_OPTIMIZATION = ML_DIR / "ml_optimization_results.csv"

ML_DIGITAL_TWIN = ML_DIR / "ml_digital_twin_simulations.csv"

ML_RL_STATE = ML_DIR / "rl_state_table.csv"

ML_RL_POLICY = ML_DIR / "rl_policy_table.csv"

ML_RL_RECOMMEND = ML_DIR / "ml_rl_recommendations.csv"

ML_RL_OUTCOME = ML_DIR / "rl_outcome_table.csv"

ML_RL_REWARD = ML_DIR / "rl_actual_rewards.csv"


# ============================================================
# LOGS
# ============================================================

LOG_DIR = ROOT / "logs"

PIPELINE_LOG = LOG_DIR / "pipeline.log"

ERROR_LOG = LOG_DIR / "errors.log"


# ============================================================
# UTILITY
# ============================================================

def ensure_directories():
    """
    Create required directories if they don't exist.
    """

    dirs = [
        CONFIG_DIR,
        RAW_DATA_DIR,
        MODELED_DIR,
        REPORT_DIR,
        ML_DIR,
        LOG_DIR,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)