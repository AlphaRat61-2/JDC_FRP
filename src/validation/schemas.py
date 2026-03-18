MASTER_WELL_SCHEMA = {
    "required": ["well_id", "well_name", "asset", "route"],
    "optional": [],
}

PRODUCTION_SCHEMA = {
    "required": ["asset", "route", "well_id", "well_name", "date", "bopm", "mcfm", "bwpm"],
    "optional": [],
}

CHEMICAL_RATES_SCHEMA = {
    "required": [
        "date", "asset", "route", "well_id", "well_name",
        "chem_name", "chem_type", "target_gpd"
    ],
    "optional": [],
}

CHEMICAL_COST_SCHEMA = {
    "required": [
        "date", "asset", "route", "well_id", "well_name",
        "chem_name", "chem_type", "equipment", "qty",
        "unit_cost", "actual_cost"
    ],
    "optional": [],
}

FAILURE_SCHEMA = {
    "required": [
        "asset", "route", "well_id", "well_name", "install_date", "fail_date",
        "equipment_type", "failure_type", "failure_cause", "failure_location",
        "depth", "failure_cost", "vendor", "comment"
    ],
    "optional": [],
}