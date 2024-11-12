from datetime import datetime
import pandas as pd

from cenai_core import cenai_path
from cenai_core.pandas_helper import from_json

class PDACRecapper:
    def __init__(self):
        self._output_dir = cenai_path(
            "output/amc"
        )

        (self._evaluation_df,
         self._run_df,
         self._hparams) = self._merge_data()

        self._statistic_df = self._format_statistic()
        self._result_df = self._format_result()
        self._deviate_df = self._format_deviate()

    def _merge_data(self) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        datastore_dir = self._output_dir / "datastore"

        evaluation_df = pd.DataFrame()
        run_df = pd.DataFrame()

        for json_file in datastore_dir.rglob("*.json"):
            a_evaluation_df, a_run_df, *_ = from_json(json_file)

            evaluation_df = pd.concat(
                [evaluation_df, a_evaluation_df], axis=0,
            )

            run_df = pd.concat(
                [run_df, a_run_df], axis=0,
            )
            
        evaluation_df = evaluation_df.reset_index(drop=True)
        run_df = run_df.reset_index(drop=True)

        hparams = [
            column
            for column in evaluation_df.columns
            if column not in [
                "run_id",
                "dataset",
                "model",
                "algorithm",
                "sections",
                "hit_ratio",
                "hit",
                "total"
            ]
        ]

        return evaluation_df, run_df, hparams

    def _format_statistic(self) -> pd.DataFrame:
        time_df = self._run_df.groupby(
            "run_id", as_index=False
        )["소요시간"].mean()

        evaluation_df = pd.merge(
            self._evaluation_df,
            time_df,
            on="run_id",
            how="outer",
        )

        evaluation_df["dataset"] = evaluation_df.apply(
            lambda field: "_".join(field["run_id"].split("_")[:2]),
            axis=1
        )

        value_order = ["hit", "total", "hit_ratio", "소요시간"]

        statistic_df = pd.pivot_table(
            evaluation_df,
            values=["hit", "total", "hit_ratio", "소요시간"],
            index=["model", "algorithm"] + self._hparams,
            columns=["dataset", "sections"],
            aggfunc=["mean"],
            fill_value=pd.NA,
            dropna=False,
        ).dropna(
            how="all"
        ).reindex(
            value_order,
            level=1,
            axis=1,
        )

        return statistic_df

    def _format_result(self) -> pd.DataFrame:
        result_df = pd.merge(
            self._run_df,
            self._evaluation_df.drop(["hit"], axis=1),
            on="run_id",
            how="outer",
        )

        columns = [
            "model", "dataset", "sections", "algorithm",
        ] + self._hparams + [
            "hit", "정답", "예측", "근거", "본문", "결론", "생성질문",
        ]
        
        result_df = result_df[columns]
        return result_df

    def _format_deviate(self) -> pd.DataFrame:
        deviate_df = pd.merge(
            self._run_df,
            self._evaluation_df.drop(["hit"], axis=1),
            on="run_id",
            how="outer",
        )

        deviate_df = deviate_df[~deviate_df["hit"]]

        deviate_df["dataset"] = deviate_df.apply(
            lambda field: field["dataset"].split("_")[0],
            axis=1
        )

        columns = [
            "model", "dataset", "sections", "algorithm",
        ] + self._hparams + [
            "정답", "예측", "근거", "본문", "결론", "생성질문",
        ]

        deviate_df = deviate_df[columns]

        return deviate_df

    def export_excel(self):
        time = datetime.now().strftime("%Y-%m-%dT%H:%M")

        excel_dir = self._output_dir / "excel"
        excel_dir.mkdir(parents=True, exist_ok=True)
        
        excel_file = excel_dir / f"recap_{time}.xlsx"

        with pd.ExcelWriter(excel_file) as writer:
            self._statistic_df.to_excel(writer, sheet_name="Score Board")
            self._result_df.to_excel(writer, sheet_name="Total RUN List")
            self._deviate_df.to_excel(writer, sheet_name="Miss RUN List")

    @property
    def evaluation_df(self) -> pd.DataFrame:
        return self._evaluation_df

    @property
    def run_df(self) -> pd.DataFrame:
        return self._run_df

    @property
    def hparams(self) -> pd.DataFrame:
        return self._hparams

    @property
    def statistic_df(self) -> pd.DataFrame:
        return self._statistic_df

    @property
    def result_df(self) -> pd.DataFrame:
        return self._result_df

    @property
    def deviate_df(self) -> pd.DataFrame:
        return self._deviate_df
