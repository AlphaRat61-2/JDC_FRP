import importlib
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.paths import (
    RAW_DATA_ROOT,
    MODELED_DIR,
    REPORT_DIR,
    ML_DIR,
    LOG_DIR,
    CONFIG_DIR,
)

print("\n----- ENVIRONMENT TEST -----\n")

# ---------------------------------------------------
# Python version
# ---------------------------------------------------

print("Python version:", sys.version)

# ---------------------------------------------------
# Required packages
# ---------------------------------------------------

packages = ["pandas", "numpy", "sklearn", "yaml", "matplotlib"]

print("\nChecking packages...\n")
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"[OK] {pkg}")
    except ImportError:
        print(f"[MISSING] {pkg}")

# ---------------------------------------------------
# Directory structure
# ---------------------------------------------------

print("\nChecking directories...\n")

dirs_to_check = {
    "raw data": RAW_DATA_ROOT,
    "modeled": MODELED_DIR,
    "reports": REPORT_DIR,
    "outputs/ml": ML_DIR,
    "logs": LOG_DIR,
    "config": CONFIG_DIR,
}

for label, path in dirs_to_check.items():
    if path.exists():
        print(f"[OK] {label} -> {path}")
    else:
        print(f"[MISSING] {label} -> {path}")

# ---------------------------------------------------
# Required input data
# ---------------------------------------------------

print("\nChecking raw data files...\n")

required_raw_files = [
    "production_data.csv",
    "master_well.csv",
    "chemical_rates.csv",
    "chemical_cost.csv",
    "failure_data.csv",
]

for filename in required_raw_files:
    path = RAW_DATA_ROOT / filename
    if path.exists():
        print(f"[OK] {filename}")
    else:
        print(f"[MISSING] {filename}")

# ---------------------------------------------------
# Config files
# ---------------------------------------------------

print("\nChecking config files...\n")

config_files = [
    "settings.yaml",
    "alert_thresholds.csv",
    "chemical_mapping.csv",
    "maintenance_rules.csv",
    "rl_action_space.csv",
]

for filename in config_files:
    path = CONFIG_DIR / filename
    if path.exists():
        print(f"[OK] {filename}")
    else:
        print(f"[MISSING] {filename}")

print("\n----- TEST COMPLETE -----\n")
