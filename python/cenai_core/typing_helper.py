from typing import Union

import pandas as pd
from pandas._libs.tslibs.nattype import NaTType

Column = str
Columns = Union[list[str], pd.Index]
DateTime = Union[pd.Timestamp, NaTType]
PgConv = dict[str, str]
TimeDelta = Union[pd.Timedelta, NaTType]
