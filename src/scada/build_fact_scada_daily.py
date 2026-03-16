from __future__ import annotations

import pandas as pd

from src.common.exceptions import make_exception
from src.common.paths import get_path
from src.io.exception_store import append_exceptions
from src.io.writers import write_table
from src.scada.summarize_esp_daily import summarize_esp_daily
from src.scada.summarize_rod_pump_daily import summarize_rod_pump_daily
from src.scada.surveillance_features import add_surveillance_features
from src.scada.tag_mapping import map_scada_tags


def build_fact_scada_daily(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")

    scada_path = staged_dir / "stg_scada.csv"
    well_path = modeled_dir / "dim_well.csv"

    if not scada_path.exists():
        logger.warning("stg_scada.csv not found.")
        return pd.DataFrame()

    scada = pd.read_csv(scada_path, dtype={"well_id": str})
    scada["timestamp"] = pd.to_datetime(scada["timestamp"], errors="coerce")

    if well_path.exists():
        wells = pd.read_csv(well_path, dtype={"well_id": str})
        scada = scada.merge(
            wells[["well_id", "lift_type"]],
            how="left",
            on="well_id",
        )
    else:
        scada["lift_type"] = None

    exceptions: list[dict] = []

    missing_lift = scada[scada["lift_type"].isna()]
    for idx, row in missing_lift.iterrows():
        exceptions.append(
            make_exception(
                table_name="stg_scada",
                record_key=f"{row.get('well_id', '')}|{idx}",
                exception_category="SCADA",
                exception_code="MISSING_LIFT_TYPE",
                severity="WARNING",
                message="SCADA row could not be assigned a lift type from dim_well.",
                batch_id=batch.batch_id,
            )
        )

    scada = map_scada_tags(scada)

    unmapped_tags = scada[scada["metric_name"].isna()]
    for idx, row in unmapped_tags.iterrows():
        exceptions.append(
            make_exception(
                table_name="stg_scada",
                record_key=f"{row.get('well_id', '')}|{row.get('tag_name', '')}|{idx}",
                exception_category="SCADA",
                exception_code="UNMAPPED_SCADA_TAG",
                severity="WARNING",
                message=f"SCADA tag not mapped: {row.get('tag_name', '')}",
                batch_id=batch.batch_id,
            )
        )

    rod = scada[scada["lift_type"] == "ROD_PUMP"].copy()
    esp = scada[scada["lift_type"] == "ESP"].copy()

    rod_daily = summarize_rod_pump_daily(rod)
    if not rod_daily.empty:
        rod_daily["lift_type"] = "ROD_PUMP"
        rod_daily = add_surveillance_features(rod_daily, lift_type="ROD_PUMP")

    esp_daily = summarize_esp_daily(esp)
    if not esp_daily.empty:
        esp_daily["lift_type"] = "ESP"
        esp_daily = add_surveillance_features(esp_daily, lift_type="ESP")

    fact = pd.concat([rod_daily, esp_daily], ignore_index=True, sort=False)

    if not fact.empty:
        fact["scada_source"] = "scada"
        fact["comm_loss_flag"] = False

    if exceptions:
        append_exceptions(settings, exceptions, logger)

    write_table(fact, modeled_dir, "fact_scada_daily", settings)
    batch.set_row_count("fact_scada_daily", len(fact))
    logger.info("Built fact_scada_daily | rows=%s", len(fact))
    return fact
