import pandas as pd

prod = pd.read_csv("data/modeled/fact_production_daily.csv", dtype={"well_id": str})
fail = pd.read_csv("data/modeled/fact_failure_event.csv", dtype={"well_id": str})

prod_ids = set(prod["well_id"].dropna().astype(str).str.strip())
fail_ids = set(fail["well_id"].dropna().astype(str).str.strip())

print("prod wells:", len(prod_ids))
print("fail wells:", len(fail_ids))
print("matching wells:", len(prod_ids & fail_ids))
print("fail-only sample:", list(sorted(fail_ids - prod_ids))[:20])
print("prod-only sample:", list(sorted(prod_ids - fail_ids))[:20])