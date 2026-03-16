import importlib
from pathlib import Path

print("\n----- ENVIRONMENT TEST -----\n")

# ---------------------------------------------------
# Python version
# ---------------------------------------------------

import sys

print("Python version:", sys.version)


# ---------------------------------------------------
# Required packages
# ---------------------------------------------------

packages = [
    "pandas",
    "numpy",
    "sklearn",
    "yaml",
    "matplotlib"
]

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

required_dirs = [
    "data/raw",
    "data/modeled",
    "data/reports",
    "outputs/ml",
    "config",
    "logs"
]

print("\nChecking directories...\n")

for d in required_dirs:

    path = Path(d)

    if path.exists():
        print(f"[OK] {d}")
    else:
        print(f"[MISSING] {d}")


# ---------------------------------------------------
# Required input data
# ---------------------------------------------------

required_files = [
    "data/raw/production_data.csv",
    "data/raw/master_well.csv",
    "data/raw/chemical_rates.csv",
    "data/raw/chemical_cost.csv",
    "data/raw/failure_date.csv"
]

print("\nChecking raw data files...\n")

for f in required_files:

    path = Path(f)

    if path.exists():
        print(f"[OK] {f}")
    else:
        print(f"[MISSING] {f}")


# ---------------------------------------------------
# Config files
# ---------------------------------------------------

config_files = [
    "config/settings.yaml",
    "config/alert_thresholds.csv",
    "config/chemical_mapping.csv",
    "config/maintenance_rules.csv",
    "config/rl_action_space.csv",
]

print("\nChecking config files...\n")

for f in config_files:

    path = Path(f)

    if path.exists():
        print(f"[OK] {f}")
    else:
        print(f"[MISSING] {f}")


print("\n----- TEST COMPLETE -----\n")