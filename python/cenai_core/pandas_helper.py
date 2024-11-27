from io import StringIO
import json
import pandas as pd
from pathlib import Path


example_df = pd.DataFrame({
    "a": [1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2,],
    "b": [4, 5, 6, 4, 5, 6, 6, 4, 5, 4, 4, 4,],
    "c": [7, 7, 7, 8, 7, 9, 9, 8, 8, 8, 9, 7,],
    "d": [1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 2, 3,],
})


def to_json(target: Path, *args, **kwargs) -> None:
    data = [
        dataframe.to_json(orient="split", **kwargs)
        for dataframe in args
    ]

    with target.open("wt") as fout:
        json.dump(data, fout)


def from_json(source: Path, **kwargs) -> list[pd.DataFrame]:
    with source.open("rt") as fin:
        data = json.load(fin)

    dataframes = [
        pd.read_json(StringIO(serial), orient="split", **kwargs)
        for serial in data
    ]

    return dataframes
