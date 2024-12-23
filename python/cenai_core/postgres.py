from typing import Any, Optional

import numpy as np
import os
import pandas as pd
from pathlib import Path
import psycopg2
from psycopg2.extras import Json

from cenai_core import Logger, load_dotenv
from cenai_core.dataman import optional, Q, QQ

from cenai_core.pandas_helper import (
    concat_columns, exclude_columns, to_pydatetime, to_pytimedelta
)

from cenai_core.typing_helper import Columns, PgConv


class Postgres(Logger):
    logger_name = "cenai.postgres"

    RESERVED_WORDS: set[str] = {
        "all", "analyse", "analyze", "and", "any", "array", "as", "asc",
        "asymmetric", "authorization", "binary", "both", "case", "cast",
        "check", "collate", "column", "concurrently", "constraint", "create",
        "cross", "current_catalog", "current_date", "current_role",
        "current_schema", "current_time", "current_timestamp", "current_user",
        "default", "deferrable", "desc", "distinct", "do", "else", "end",
        "except", "false", "fetch", "for", "foreign", "freeze", "from", "full",
        "grant", "group", "having", "ilike", "in", "initially", "inner",
        "intersect", "into", "is", "isnull", "join", "lateral", "leading",
        "left", "like", "limit", "localtime", "localtimestamp", "not",
        "notnull", "null", "offset", "on", "only", "or", "order", "outer",
        "overlaps", "placing", "primary", "references", "returning", "right",
        "select", "session_user", "similar", "some", "symmetric", "table",
        "tablesample", "then", "to", "trailing", "true", "union", "unique",
        "user", "using", "variadic", "verbose", "when", "where", "window",
        "with"
    }

    def __init__(self, database: str):
        super().__init__()

        load_dotenv()

        prefix = f"POSTGRES_{database.upper()}"

        for name, casting in zip(
            ["host", "port", "user", "password", "db"],
            [str, int, str, str, str],
        ):
            env_name = f"{prefix}_{name.upper()}"
            value = os.environ.get(env_name)

            if value is None:
                raise ValueError(
                    f"envvar {Q(env_name)} undefined for {Q(name)}"
                )

            self.__dict__[f"_{name}"] = casting(value)

        self._conn = None

    def connect(self) -> psycopg2.extensions.connection:
        self.INFO(
            f"host: {self._host} "
            f"port: {self._port} "
            f"user: {self._user} "
            f"password: {self._password[0:3]}* "
            f"db: {self._db} "
        )

        self._conn = psycopg2.connect(
            host=self._host,
            user=self._user,
            password=self._password,
            database=self._db,
            port=self._port,
            keepalives=1,
            keepalives_idle=130,
            keepalives_interval=10,
            keepalives_count=15
        )
        return self._conn

    @property
    def url(self) -> str:
        return str(
            f"postgresql+psycopg2://{self._user}:{self._password}"
            f"@{self._host}:{self._port}/{self._db}"
        )

    @property
    def name(self) -> str:
        return self._db

    @property
    def conn(self) -> psycopg2.extensions.connection:
        return self._conn

    def close(self) -> None:
        self.conn.close()


def R(column: str) -> str:
    return (
        f"{QQ(column)}"
    ) if column.lower() in Postgres.RESERVED_WORDS else (
        column
    )


def to_columns(
    columns: Columns, common: str = "", prefix: dict[str, str] = {}
) -> str:
    common = f"{R(common)}." if common else ""
    entries: list[str] = [
        f"{R(prefix[column])}.{R(column)}" if column in prefix else
        f"{common}{R(column)}"
        for column in columns
    ]
    return ", ".join(entries)


def to_percents(columns: Columns) -> str:
    return ", ".join(["%s"] * len(columns))


def to_updates(
    columns: Columns, include: list[str] = [], exclude: list[str] = [],
    prefix: str = "EXCLUDED"
) -> str:
    total = exclude_columns(
        concat_columns(columns, include),
        exclude
    )

    updates = [
        f"{R(column)} = {prefix}.{R(column)}" for column in total
    ]
    return ", ".join(updates)


def from_rows(rows: Optional[list[Any]]) -> list[Any]:
    return optional(rows, [])


def from_row(row: Optional[tuple[Any, ...]], k: int = 0) -> Optional[Any]:
    return None if row is None else row[k]


def from_records(
    src: list[Any], columns: Optional[pd.Index | list[str]] = None,
    pgconv: PgConv = {}
) -> pd.DataFrame:
    data = pd.DataFrame.from_records(
        src, columns=columns
    )

    for column, type_ in pgconv.items():
        if type_ == "json":
            continue

        if type_ not in ["datetime", "timestamp", "timedelta"]:
            raise TypeError(
                f"type {Q(type_)} in {Q('pgconv')} is unknown"
            )

        data[column] = (
            pd.to_datetime(data[column]) if type_ == "datetime" else
            pd.to_datetime(data[column], utc=True) if type_ == "timestamp" else
            pd.to_timedelta(data[column])
        )

    return data


def to_vars(data: pd.DataFrame, pgconv: PgConv = {}) -> list:
    for column, type_ in pgconv.items():
        if type_ not in ["datetime", "timestamp", "timedelta", "json"]:
            raise TypeError(
                f"type {Q(type_)} in {Q('pgconv')} is unknown"
            )

        data[column] = (
            to_pydatetime(data[column])
            if type_ in ["datetime", "timestamp"] else
            to_pytimedelta(data[column]) if type_ == "timedelta" else
            data[column].apply(
                lambda e: None if pd.isna(e) else Json(e)
            )
        )

    data = data.replace(
        {pd.NA: None, pd.NaT: None, np.nan: None}
    )
    return data.to_records(index=False).tolist()


def to_var(data: pd.DataFrame,
           pgconv: PgConv = {}
           ) -> tuple[Any, ...]:
    return tuple(to_vars(data, pgconv))


def execute_script(cursor: psycopg2.extensions.cursor,
                   text: Path | str) -> None:
    if isinstance(text, Path):
        text = text.read_text()

    commands = text.split(";")
    for command in commands:
        command = command.strip()
        if command:
            cursor.execute(command)
