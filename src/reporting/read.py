import pandas as pd

# 👇 point this to your actual staged production file
df = pd.read_csv(r"C:\Users\jcaron\OneDrive - Ring Energy\JDC_Data\JDC_FRP\data\raw\production_data.csv")


# add row numbers (so you can find them in Excel)
df["row_number"] = df.index + 2

# find exact duplicate rows
dupes = df[df.duplicated(keep=False)]

# export them
dupes.to_csv("duplicates.csv", index=False)