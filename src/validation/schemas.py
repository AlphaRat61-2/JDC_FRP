MASTER_WELL_SCHEMA = {
    "required": ["well_id", "well_name", "asset", "route"],
    "optional": [],
}

PRODUCTION_SCHEMA = {
    # Raw incoming production columns expected BEFORE rename_production_columns()
    "required": ["asset", "route", "well_id", "well_name", "date", "bopd", "mcfd", "bwpd"],
    # Keep gfflap available as an input variable without making it mandatory
    "optional": ["gfflap"],
}

CHEMICAL_RATES_SCHEMA = {
    "required": [
        "date",
        "asset",
        "route",
        "well_id",
        "well_name",
        "chem_name",
        "chem_type",
        "target_gpd",
    ],
    "optional": [],
}

CHEMICAL_COST_SCHEMA = {
    "required": [
        "date",
        "asset",
        "route",
        "well_id",
        "well_name",
        "vendor",
        "line_category",
        "qty",
        "unit_cost",
        "actual_cost",
    ],
    "optional": [
        "chem_name",
        "chem_type",
        "equipment",
        "truck_treat",
        "uom",
    ],
}

FAILURE_SCHEMA = {
    "required": [
        "asset",
        "route",
        "well_id",
        "well_name",
        "install_date",
        "fail_date",
        "equipment_type",
        "failure_type",
        "failure_cause",
        "failure_location",
        "depth",
        "failure_cost",
        "vendor",
        "comment",
    ],
    "optional": [],
}