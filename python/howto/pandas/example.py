import pandas as pd


df = pd.DataFrame({
    "a": [2, 0, 1],
    "b": [3, 4, 5],
    "c": [6, 7, 8],
})

df = df.sort_values("a")

df = df.reset_index().rename(columns={"index": "sample"})
df["sample"] = df["sample"].astype(int)

for e in df["sample"]:
    print(e, type(e))
