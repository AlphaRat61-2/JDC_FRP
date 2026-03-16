from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]


CONFIG_DIR = ROOT / "config"
LOG_DIR = ROOT / "logs"

RAW_DATA_DIR = ROOT / "data" / "raw"
MODELED_DIR = ROOT / "data" / "modeled"
REPORT_DIR = ROOT / "data" / "reports"
ML_DIR = ROOT / "outputs" / "ml"


def project_root():
    return ROOT


def get_path(settings, key):
    return ROOT / settings["paths"][key]


def ensure_directories(settings=None):
    dirs = [
        CONFIG_DIR,
        LOG_DIR,
        RAW_DATA_DIR,
        MODELED_DIR,
        REPORT_DIR,
        ML_DIR,
    ]

    if settings and "paths" in settings:
        for p in settings["paths"].values():
            dirs.append(ROOT / p)

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_settings(settings_file="config/settings.yaml"):
    path = ROOT / settings_file

    with open(path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    ensure_directories(settings)
    return settings