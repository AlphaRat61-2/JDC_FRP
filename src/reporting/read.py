import pandas as pd

df = pd.read_csv("outputs/ml/ml_failure_risk_scored.csv")

print(df.groupby("risk_bucket")["pre_failure_flag"].mean())
print(df.groupby("risk_bucket")["failure_risk_30d"].mean())