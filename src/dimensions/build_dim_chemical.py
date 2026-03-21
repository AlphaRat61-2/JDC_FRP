from __future__ import annotations

import re
import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def _normalize_chem_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    text = re.sub(r"[\s\-/]+", " ", text)
    text = re.sub(r"[^A-Z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _build_fallback_key(norm_name: str | None) -> str | None:
    if not norm_name:
        return None
    return norm_name.replace(" ", "_")


def build_dim_chemical(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")
    config_dir = get_path(settings, "incoming").parents[0] / "config"

    frames = []

    for fname in ["stg_chemical_rates.csv", "stg_chemical_cost.csv"]:
        path = staged_dir / fname
        if path.exists():
            df = pd.read_csv(path)

            if "chem_name" not in df.columns:
                logger.warning("%s missing chem_name column. Skipping.", fname)
                continue

            if "chem_type" not in df.columns:
                df["chem_type"] = pd.NA

            part = df[["chem_name", "chem_type"]].copy()
            part["chem_name_raw"] = part["chem_name"]
            part["chem_name_norm"] = part["chem_name"].apply(_normalize_chem_text)
            part["chem_type"] = part["chem_type"].astype("string").str.strip()
            frames.append(part)

    if not frames:
        logger.warning("No chemistry staging files found for dim_chemical.")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True).copy()
    raw = raw.loc[raw["chem_name_norm"].notna()].copy()

    raw = (
        raw.sort_values(["chem_name_norm", "chem_type", "chem_name_raw"], na_position="last")
        .drop_duplicates(subset=["chem_name_norm"], keep="first")
        .copy()
    )

    chem_map_path = config_dir / "chem_name_map.csv"

    if chem_map_path.exists():
        chem_map = pd.read_csv(chem_map_path)

        if "chem_name" not in chem_map.columns:
            raise ValueError("chem_name_map.csv must contain a chem_name column.")

        chem_map = chem_map.copy()
        chem_map["chem_name_norm"] = chem_map["chem_name"].apply(_normalize_chem_text)

        if "chemical_key" not in chem_map.columns:
            chem_map["chemical_key"] = chem_map["chem_name_norm"].apply(_build_fallback_key)

        if "normalized_chemical_name" not in chem_map.columns:
            chem_map["normalized_chemical_name"] = chem_map["chem_name_norm"]

        if "dose_basis" not in chem_map.columns:
            chem_map["dose_basis"] = pd.NA

        chem_map = (
            chem_map.sort_values(["chem_name_norm", "chemical_key"], na_position="last")
            .drop_duplicates(subset=["chem_name_norm"], keep="first")
            .copy()
        )

        dim = raw.merge(
            chem_map[
                [
                    "chem_name_norm",
                    "chemical_key",
                    "normalized_chemical_name",
                    "dose_basis",
                ]
            ],
            how="left",
            on="chem_name_norm",
            validate="many_to_one",
        )
    else:
        dim = raw.copy()
        dim["chemical_key"] = dim["chem_name_norm"].apply(_build_fallback_key)
        dim["normalized_chemical_name"] = dim["chem_name_norm"]
        dim["dose_basis"] = pd.NA

    # Fill fallback mappings when map file has no match
    dim["chemical_key"] = dim["chemical_key"].fillna(dim["chem_name_norm"].apply(_build_fallback_key))
    dim["normalized_chemical_name"] = dim["normalized_chemical_name"].fillna(dim["chem_name_norm"])
    dim["dose_basis"] = dim["dose_basis"] if "dose_basis" in dim.columns else pd.NA

    unmapped = dim.loc[dim["chemical_key"].isna(), ["chem_name_raw", "chem_name_norm", "chem_type"]].copy()
    if len(unmapped) > 0:
        logger.warning(
            "dim_chemical has rows with missing chemical_key after normalization: %s | sample:\n%s",
            len(unmapped),
            unmapped.head(20).to_string(index=False),
        )

    keep_cols = [
        "chemical_key",
        "normalized_chemical_name",
        "chem_type",
        "dose_basis",
    ]
    dim = dim[keep_cols].drop_duplicates().copy()

    dim = dim.loc[dim["chemical_key"].notna()].copy()

    write_table(dim, modeled_dir, "dim_chemical", settings)
    batch.set_row_count("dim_chemical", len(dim))
    logger.info("Built dim_chemical | rows=%s", len(dim))
    return dim