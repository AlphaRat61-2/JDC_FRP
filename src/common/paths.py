from pathlib import Path
import yaml


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================
# CONFIG FILES
# ============================================================

CONFIG_DIR = ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"


def project_root():
    return ROOT


def load_settings(path: str | Path | None = None):
    # If no path is provided, use the single canonical settings.yaml
    settings_file = Path(path) if path is not None else SETTINGS_FILE

    with open(settings_file, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    return settings


# Load settings once for repo-wide path constants
SETTINGS = load_settings()
PATHS = SETTINGS["paths"]
DATA_ROOT = Path(PATHS["data_root"])


# ============================================================
# REPO-BASED CONFIG FILES
# ============================================================

CHEM_MAPPING_FILE = CONFIG_DIR / "chemical_mapping.csv"
ALERT_THRESHOLDS_FILE = CONFIG_DIR / "alert_thresholds.csv"
MAINT_RULES_FILE = CONFIG_DIR / "maintenance_rules.csv"
RL_ACTION_SPACE_FILE = CONFIG_DIR / "rl_action_space.csv"
FAILURE_CLASSES_FILE = CONFIG_DIR / "failure_classes.csv"


# ============================================================
# DATA / OUTPUT / LOG DIRECTORIES
# ============================================================

RAW_DATA_DIR = DATA_ROOT / PATHS["raw"]
INCOMING_DIR = DATA_ROOT / PATHS["incoming"]
STAGED_DIR = DATA_ROOT / PATHS["staged"]
MODELED_DIR = DATA_ROOT / PATHS["modeled"]
REPORT_DIR = DATA_ROOT / PATHS["reports"]
ML_DIR = DATA_ROOT / PATHS["outputs_ml"]
LOG_DIR = DATA_ROOT / PATHS["logs"]


# ============================================================
# RAW INPUT FILES
# ============================================================

PRODUCTION_RAW = RAW_DATA_DIR / "production_data.csv"
MASTER_WELL_RAW = RAW_DATA_DIR / "master_well.csv"
CHEM_RATES_RAW = RAW_DATA_DIR / "chemical_rates.csv"
CHEM_COST_RAW = RAW_DATA_DIR / "chemical_cost.csv"
FAILURES_RAW = RAW_DATA_DIR / "failure_data.csv"
SCADA_RAW = RAW_DATA_DIR / "scada_data.csv"
WORKOVER_RAW = RAW_DATA_DIR / "workover_data.csv"


# ============================================================
# MODELED DATA
# ============================================================

FACT_PRODUCTION = MODELED_DIR / "fact_production_daily.csv"
FACT_SCADA = MODELED_DIR / "fact_scada_daily.csv"
FACT_FAILURE = MODELED_DIR / "fact_failure_event.csv"
FACT_CHEM_RECON = MODELED_DIR / "fact_chem_recon_daily.csv"
FACT_WORKOVER = MODELED_DIR / "fact_workover_event.csv"


# ============================================================
# REPORT OUTPUTS
# ============================================================

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
# LOG FILES
# ============================================================

PIPELINE_LOG = LOG_DIR / "pipeline.log"
ERROR_LOG = LOG_DIR / "errors.log"


# ============================================================
# HELPERS
# ============================================================

def get_path(settings, key: str) -> Path:
    data_root = Path(settings["paths"]["data_root"])
    return data_root / settings["paths"][key]


def ensure_directories():
    dirs = [
        CONFIG_DIR,
        RAW_DATA_DIR,
        INCOMING_DIR,
        STAGED_DIR,
        MODELED_DIR,
        REPORT_DIR,
        ML_DIR,
        LOG_DIR,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
