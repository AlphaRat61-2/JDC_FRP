#import pandas as pd
#df = pd.read_csv("outputs/ml/ml_failure_risk_scored.csv", nrows=5)
#print(df.columns.tolist())

import pandas as pd

target = pd.read_csv("data/modeled/fact_chem_target_daily.csv", dtype={"well_id": str, "chemical_key": str})
actual = pd.read_csv("data/modeled/fact_chem_actual_daily.csv", dtype={"well_id": str, "chemical_key": str})

print("TARGET COLUMNS:", target.columns.tolist())
print("ACTUAL COLUMNS:", actual.columns.tolist())

print("\nTARGET DUPES:")
print(
    target.groupby(["well_id", "date", "chemical_key"])
    .size()
    .reset_index(name="n")
    .query("n > 1")
    .head(20)
)

print("\nACTUAL SAMPLE:")
print(actual.head())