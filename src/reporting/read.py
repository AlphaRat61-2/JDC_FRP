import pandas as pd

scored = pd.read_csv("outputs/ml/ml_failure_risk_scored.csv", dtype={"well_id": str})
labels = pd.read_csv("outputs/ml/ml_label_failure.csv", dtype={"well_id": str})

scored["date"] = pd.to_datetime(scored["date"], errors="coerce").dt.date
labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.date

df = scored.merge(
    labels[["well_id", "date", "failure_within_30d"]],
    how="left",
    on=["well_id", "date"],
)

print(df.groupby("risk_bucket")["failure_within_30d"].mean())
print(df.groupby("risk_bucket")["failure_risk_30d"].mean())

p95 = df["failure_risk_30d"].quantile(0.95)
top = df[df["failure_risk_30d"] >= p95]

print("95th percentile cutoff:", p95)
print("Top 5% failure rate:", top["failure_within_30d"].mean())
print("Overall failure rate:", df["failure_within_30d"].mean())