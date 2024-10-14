import pandas as pd
from sklearn.model_selection import train_test_split

from cenai_core import cenai_path


json_file = cenai_path("data/ct_report.json")

df = pd.read_json(json_file)

train_df = pd.DataFrame()
test_df = pd.DataFrame()

for group, data in df.groupby("유형"):
    print(data.columns)
    train, test = train_test_split(
        data, test_size=0.2, 
        random_state=42,
    )
    train_df = pd.concat([train_df, train], axis=0)
    test_df = pd.concat([test_df, test], axis=0)
