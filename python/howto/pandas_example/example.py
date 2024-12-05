import pandas as pd

df = pd.DataFrame(
    {
        "a": [1, 1, 1, 2, 2, 2,],
        "b": [1, 2, 1, 2, 2, 1,],
        "c": [1, 2, 3, 4, 5, 6,],
    }
)

def f(group) -> str:
    return group + 2


k = df.groupby("a").c.apply(f)

print(k.to_frame().reset_index())