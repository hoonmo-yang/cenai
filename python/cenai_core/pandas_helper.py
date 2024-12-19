from typing import Any, Callable, Optional

from io import StringIO
import json
from os import PathLike
import pandas as pd
from pathlib import Path

from cenai_core.typing_helper import Columns, DateTime, TimeDelta
from cenai_core.dataman import concat_ranges


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


class DataFrameSchema:
    def __init__(
            self,
            serializer: str = "",
            header: int = 0,
            skiprows: list[range | int] = [],
            usecols: list[range | int] = [],
            columns: Columns = [],
            converters: dict[str, Callable[[Any], Any]] = {}
    ):
        self._serializer = serializer
        self._header = header
        self._skiprows = concat_ranges(*skiprows)
        self._usecols = concat_ranges(*usecols)
        self._columns = columns
        self._converters = converters

    @property
    def serializer(self) -> str:
        return self._serializer

    @property
    def header(self) -> int:
        return self._header

    @property
    def skiprows(self) -> list[int]:
        return self._skiprows

    @property
    def usecols(self) -> list[int]:
        return self._usecols

    @property
    def columns(self) -> Columns:
        return self._columns

    @property
    def converters(self) -> dict[str, Callable[[Any], Any]]:
        return self._converters


class schema_datetime:
    def __init__(self, auto_int: bool):
        self._auto_int = auto_int

    def __call__(self, value: Any) -> DateTime:
        if self._auto_int and isinstance(value, (int, float)):
            value = str(int(value))
        return pd.to_datetime(value, yearfirst=True, errors="coerce")


def schema_birthdate(value: Any) -> DateTime:
    return  pd.to_datetime(f"{value // 100}-{value % 100:02d}-01")


def schema_timestamp(value: Any) -> DateTime:
    return pd.to_datetime(value, yearfirst=True, utc=True, errors="coerce")


def schema_timedelta(value: Any) -> TimeDelta:
    return pd.to_timedelta(value, errors="coerce")


def schema_timedelta_xlsx(value: Any) -> TimeDelta:
    return pd.to_timedelta(value, unit="D", errors="coerce")


def schema_string(value: Any) -> Optional[str]:
    return (
        None if pd.isna(value) or value == "" else
        str(int(value)) if isinstance(value, float) and value.is_integer() else
        str(value)
    )


def schema_integer(value: Any) -> Optional[int]:
    return None if pd.isna(value) or value == "" else int(value)


def schema_float(value: Any) -> Optional[float]:
    return None if pd.isna(value) or value == "" else float(value)


def schema_boolean(value: Any) -> Optional[bool]:
    if isinstance(value, str):
        value = value.lower()
        return (
            True if value in ["o", "ok", "t", "true"] else
            False if value in ["x", "no", "f", "false"] else
            None
        )
    return True if value == 1 else False if value == 0 else None


def to_structured_dataframe(
        data_df: pd.DataFrame,
        schema: DataFrameSchema
    ) -> pd.DataFrame:
    for key, cast in schema.converters.items():
        data_df[key] = data_df[key].apply(lambda e: cast(e))
    return data_df


def excel_to_structured_dataframe(
    excel: str | PathLike[str],
    sheet_name: str,
    schema: DataFrameSchema,
    force: bool = False,
) -> pd.DataFrame:

    if schema.serializer != "xlsx":
        raise NameError(
            f"serializer isn't xlsx: {schema.serializer}"
        )

    try:
        data_df = pd.read_excel(
            excel, sheet_name,
            skiprows=schema.skiprows,
            usecols=schema.usecols,
            names=schema.columns,
            converters=schema.converters
        )

    except Exception as error:
        if not force:
            raise error
        data_df = pd.DataFrame(columns=schema.columns)

    return data_df


def json_to_structured_dataframe(
    json: str | PathLike[str],
    schema: DataFrameSchema,
    force: bool = False
) -> pd.DataFrame:

    if schema.serializer != "json":
        raise NameError(
            f"serializer isn't json: {schema.serializer}"
        )

    try:
        data_df = pd.read_json(
            json, dtype=False
        ).pipe(
            to_structured_dataframe,
            schema=schema,
        )

    except Exception as error:
        if not force:
            raise error
        data_df = pd.DataFrame(columns=schema.columns)

    return data_df


def structured_dataframe_to_json(
        data_df: pd.DataFrame,
        json: str | PathLike[str]
    ) -> None:
    data_df.to_json(json, date_unit="ns", force_ascii=False)


def to_pydatetime(x: pd.Series) -> pd.Series:
    xsel = x[~x.isna()]
    if xsel.empty:
        return x

    out = pd.Series(
        xsel.dt.to_pydatetime(),
        dtype=object,
        index=xsel.index
    )

    out = pd.concat([out, x[x.isna()]])
    return out


def to_pytimedelta(x: pd.Series) -> pd.Series:
    xsel = x[~x.isna()]
    if xsel.empty:
        return x

    out = pd.Series(
        xsel.dt.to_pytimedelta(),
        dtype=object,
        index=xsel.index
    )
    out = pd.concat([out, x[x.isna()]])
    return out


def subtract(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([x, y, y]).drop_duplicates(keep=False)


def concat_columns(x: Columns, y: Columns) -> Columns:
    return pd.Index(x).append(pd.Index(y)).unique().tolist()


def exclude_columns(x: Columns, y: Columns) -> Columns:
    return pd.Index(x).difference(pd.Index(y)).tolist()
