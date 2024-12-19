import pandas as pd
from pandas._libs.tslibs.nattype import NaTType

Column = str
Columns = list[str] | pd.Index
DateTime = pd.Timestamp | NaTType
PgConv = dict[str, str]
TimeDelta = pd.Timedelta | NaTType
