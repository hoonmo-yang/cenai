import pandas as pd
from cenai_core.pandas_helper import example_df


example_df = example_df.apply(
    lambda x: x if(x.a // 2) else pd.Series(),
    axis=1
).dropna(how="all")

print(example_df)


example_df["x"] = example_df["a"].apply(lambda x:[x, x*2, x*3])
example_df["y"] = example_df["a"].apply(lambda x:[x, x*20, x*30])

example_df = example_df.explode(["x", "y"]).reset_index(drop=True)

print(example_df)
