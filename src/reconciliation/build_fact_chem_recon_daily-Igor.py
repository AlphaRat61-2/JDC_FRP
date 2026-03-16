from __future__ import annotations

import pandas as pd

from src.common.exceptions import make_exception
from src.common.paths import get_path
from src.chemistry.ppm_calculations import add_target_ppm
from src.io.exception_store import append_exceptions
from src.io.writers import write_table
from src.reconciliation.compliance_status import (
    classify_operational_status,
    load_chem_tolerance,
)
from src.reconciliation.exception_rules import assign_exception_code
from src.reconciliation.financial_status import classify_financial_status


def build_fact_chem_recon_daily(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")

    expected_path = modeled_dir / "fact_expected_chem_daily.csv"
    target_path = modeled_dir / "fact_chem_target_daily.csv"
    prod_path = modeled_dir / "fact_production_daily.csv"
    actual_period_path = modeled_dir / "fact_chem_actual_period.csv"
    chem_dim_path = modeled_dir / "dim_chemical.csv"

    expected = (
        pd.read_csv(expected_path, dtype={"well_id": str, "chemical_key": str})
        if expected_path.exists()
        else pd.DataFrame()
    )
    target = (
        pd.read_csv(target_path, dtype={"well_id": str, "chemical_key": str})
        if target_path.exists()
        else pd.DataFrame()
    )
    prod = (
        pd.read_csv(prod_path, dtype={"well_id": str})
        if prod_path.exists()
        else pd.DataFrame()
    )
    actual_period = (
        pd.read_csv(actual_period_path, dtype={"well_id": str, "chemical_key": str})
        if actual_period_path.exists()
        else pd.DataFrame()
    )
    chem_dim = (
        pd.read_csv(chem_dim_path, dtype={"chemical_key": str})
        if chem_dim_path.exists()
        else pd.DataFrame()
    )

    if expected.empty and target.empty:
        logger.warning("No expected or target chemistry data found.")
        return pd.DataFrame()

    if not expected.empty and "date" in expected.columns:
        expected["date"] = pd.to_datetime(expected["date"], errors="coerce").dt.date
    if not target.empty and "date" in target.columns:
        target["date"] = pd.to_datetime(target["date"], errors="coerce").dt.date
    if not prod.empty and "date" in prod.columns:
        prod["date"] = pd.to_datetime(prod["date"], errors="coerce").dt.date

    if expected.empty:
        recon = target.copy()
        recon["expected_flag"] = False
    else:
        recon = expected.merge(
            target,
            how="outer",
            on=["well_id", "date", "chemical_key"],
            suffixes=("_exp", ""),
        )
        recon["expected_flag"] = recon["required_flag"].fillna(False)

    recon["target_exists_flag"] = recon["target_rate"].notna() if "target_rate" in recon.columns else False
    recon["actual_exists_flag"] = False
    recon["actual_rate"] = None
    recon["actual_ppm"] = None
    recon["variance_rate_pct"] = None
    recon["variance_ppm_pct"] = None
    recon["confidence_status"] = "TARGET_ONLY"

    if not chem_dim.empty:
        recon = recon.merge(
            chem_dim[["chemical_key", "chem_type", "dose_basis"]].drop_duplicates(),
            how="left",
            on="chemical_key",
            suffixes=("", "_dim"),
        )

        if "target_basis" not in recon.columns:
            recon["target_basis"] = None

        if "default_target_basis" in recon.columns:
            recon["target_basis"] = recon["target_basis"].fillna(recon["default_target_basis"])

        recon["target_basis"] = recon["target_basis"].fillna(recon["dose_basis"])

    if not prod.empty:
        recon = recon.merge(
            prod[["well_id", "date", "oil_bbl", "water_bbl", "gas_mcf"]],
            how="left",
            on=["well_id", "date"],
        )
    else:
        recon["oil_bbl"] = None
        recon["water_bbl"] = None
        recon["gas_mcf"] = None

    recon = add_target_ppm(recon)

    if not actual_period.empty:
        actual_period["period_start"] = pd.to_datetime(actual_period["period_start"], errors="coerce").dt.date
        spend_daily = (
            actual_period.groupby(["well_id", "chemical_key", "period_start"], as_index=False)
            .agg(spend=("actual_total_cost", "sum"))
            .rename(columns={"period_start": "date"})
        )
        recon = recon.merge(spend_daily, how="left", on=["well_id", "chemical_key", "date"])
        recon["spend_exists_flag"] = recon["spend"].notna()
    else:
        recon["spend"] = None
        recon["spend_exists_flag"] = False

    tol_df = load_chem_tolerance()

    recon["operational_status"] = recon.apply(
        lambda r: classify_operational_status(
            target_rate=r.get("target_rate"),
            actual_rate=r.get("actual_rate"),
            chem_type=r.get("chem_type"),
            tol_df=tol_df,
        ),
        axis=1,
    )

    recon["financial_status"] = recon.apply(
        lambda r: classify_financial_status(
            target_exists=bool(r.get("target_exists_flag")),
            spend_exists=bool(r.get("spend_exists_flag")),
        ),
        axis=1,
    )

    recon["exception_code"] = recon.apply(assign_exception_code, axis=1)

    exceptions: list[dict] = []
    for _, row in recon.iterrows():
        code = row.get("exception_code", "UNKNOWN")
        if code == "FULL_MATCH":
            continue

        severity = "WARNING"
        if code in {"MISSING_BASIS", "EXPECTED_NO_TARGET"}:
            severity = "CRITICAL"

        record_key = f"{row.get('well_id', '')}|{row.get('date', '')}|{row.get('chemical_key', '')}"
        exceptions.append(
            make_exception(
                module="fact_chem_recon_daily",
                message="Chemical reconciliation error: " + str(row.get("whatever_field")),
                well_id=row.get("well_id"),
                severity="WARNING",
            )
        )

    if exceptions:
        append_exceptions(exceptions)

    keep_cols = [
        "well_id",
        "date",
        "chemical_key",
        "chem_type",
        "expected_flag",
        "target_exists_flag",
        "actual_exists_flag",
        "spend_exists_flag",
        "target_rate",
        "actual_rate",
        "target_basis",
        "target_ppm",
        "actual_ppm",
        "spend",
        "variance_rate_pct",
        "variance_ppm_pct",
        "operational_status",
        "financial_status",
        "exception_code",
        "confidence_status",
    ]
    keep_cols = [c for c in keep_cols if c in recon.columns]
    recon = recon[keep_cols].copy()

    write_table(recon, modeled_dir, "fact_chem_recon_daily", settings)
    batch.set_row_count("fact_chem_recon_daily", len(recon))
    logger.info("Built fact_chem_recon_daily | rows=%s", len(recon))
    return recon
