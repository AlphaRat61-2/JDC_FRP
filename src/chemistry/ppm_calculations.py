from __future__ import annotations

import pandas as pd

from src.common.constants import BBL_TO_GAL, PPM_MULTIPLIER


def calc_water_basis_ppm(chem_gal_day: float, water_bbl_day: float) -> float | None:
    if pd.isna(chem_gal_day) or pd.isna(water_bbl_day) or water_bbl_day <= 0:
        return None
    return (chem_gal_day / (water_bbl_day * BBL_TO_GAL)) * PPM_MULTIPLIER


def calc_total_fluid_basis_ppm(
    chem_gal_day: float,
    oil_bbl_day: float,
    water_bbl_day: float,
) -> float | None:
    if pd.isna(chem_gal_day):
        return None
    oil = 0 if pd.isna(oil_bbl_day) else oil_bbl_day
    water = 0 if pd.isna(water_bbl_day) else water_bbl_day
    total = oil + water
    if total <= 0:
        return None
    return (chem_gal_day / (total * BBL_TO_GAL)) * PPM_MULTIPLIER


def calc_oil_basis_ppm(chem_gal_day: float, oil_bbl_day: float) -> float | None:
    if pd.isna(chem_gal_day) or pd.isna(oil_bbl_day) or oil_bbl_day <= 0:
        return None
    return (chem_gal_day / (oil_bbl_day * BBL_TO_GAL)) * PPM_MULTIPLIER


def calc_gal_per_mmscf(chem_gal_day: float, gas_mcf_day: float) -> float | None:
    if pd.isna(chem_gal_day) or pd.isna(gas_mcf_day) or gas_mcf_day <= 0:
        return None
    return chem_gal_day / (gas_mcf_day / 1000.0)


def add_target_ppm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _calc(row):
        basis = row.get("target_basis")
        rate = row.get("target_rate")
        oil = row.get("oil_bbl")
        water = row.get("water_bbl")
        gas = row.get("gas_mcf")

        if basis == "WATER":
            return calc_water_basis_ppm(rate, water)
        if basis == "TOTAL_FLUID":
            return calc_total_fluid_basis_ppm(rate, oil, water)
        if basis == "OIL":
            return calc_oil_basis_ppm(rate, oil)
        if basis == "GAS":
            return calc_gal_per_mmscf(rate, gas)
        return None

    out["target_ppm"] = out.apply(_calc, axis=1)
    return out
