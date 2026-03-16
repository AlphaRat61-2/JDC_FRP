from __future__ import annotations

FEATURE_COLUMNS = [
    "oil_bbl",
    "gas_mcf",
    "water_bbl",
    "boe",
    "total_fluid_bbl",
    "water_cut",
    "production_change_7d",
    "boe_7d_avg",
    "chem_exception_7d",
    "spend_30d",
    "runtime_7d_avg",
    "fillage_7d_avg",
    "fillage_decline_7d",
    "trip_count_7d",
    "shutdown_count_7d",
    "deterioration_score",
    "days_since_last_failure",
    "failures_last_90d",
    "workovers_last_90d",
    "workover_cost_90d",
]

CATEGORICAL_COLUMNS = [
    "asset",
    "route",
    "lift_type",
    "equipment_profile_id",
]

TARGET_COLUMN = "failure_within_30d"
