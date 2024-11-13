from datetime import datetime
import pandas as pd

from cenai_core import cenai_path
from cenai_core.pandas_helper import from_json

class PDACRecapper:
    def __init__(self, grid: str):
        self._output_dir = cenai_path("gridout") / grid

        self._run_df, self._parameters = self._merge_data()

        self._statistic_df = self._format_statistic()
        self._result_df = self._format_result()
        self._deviate_df = self._format_deviate()

    def _merge_data(self) -> tuple[pd.DataFrame, list[str]]:
        prefix = self._output_dir.name

        metadata_df = pd.DataFrame()
        result_df = pd.DataFrame()

        for grid_dir in self._output_dir.glob(f"{prefix}*"):
            if not grid_dir.is_dir():
                continue

            datastore_dir = grid_dir / "datastore"
            for datastore_json in datastore_dir.glob(f"{prefix}*.json"):
                a_metadata_df, a_result_df, *_ = from_json(datastore_json)

                metadata_df = pd.concat(
                    [metadata_df, a_metadata_df], axis=0,
                )

                result_df = pd.concat(
                    [result_df, a_result_df], axis=0,
                )

        metadata_df = metadata_df.reset_index(drop=True)
        result_df = result_df.reset_index(drop=True)

        metadata_df["tags"] = metadata_df.apply(
             lambda field: ",".join(field.tags),
             axis=1
        )

        metadata_df["sections"] = metadata_df.apply(
             lambda field: ",".join(field.sections),
             axis=1
        )

        parameters = [
            column
            for column in metadata_df.columns
            if column not in [
                "grid_id",
                "run_id",
                "grid_name",
                "grid_index",
                "module",
                "institution",
                "task",
                "dataset",
                "prompt",
                "model",
                "tags",
                "grid_yaml",
                "sections",
            ]
        ]

        result_df["hit"] = result_df["정답"] == result_df["예측"]

        result_df["hit_ratio"] = (
            result_df.groupby(["grid_id", "run_id"])["hit"].transform("sum") /
            result_df.groupby(["grid_id", "run_id"]).transform("count")
        )

        run_df = pd.merge(
             result_df,
             metadata_df,
             on=["grid_id", "run_id"],
             how="outer",
        )

        return run_df, parameters

    def _format_statistic(self) -> pd.DataFrame:
        value_order = ["hit", "total", "hit_ratio", "소요시간"]

        run_df = self.run_df

        run_df["hit"] = run_df.groupby(["grid_id", "run_id"])["hit"].transform("sum")
        run_df["total"] = run_df.groupby(["grid_id", "run_id"]).transform("count")

        statistic_df = pd.pivot_table(
            run_df,
            values=["hit", "total", "hit_ratio", "소요시간"],
            index=["model", "module", "prompt"] + self.parameter,
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

    @property
    def run_df(self) -> pd.DataFrame:
        return self._run_df

    @property
    def parameters(self) -> list[str]:
        return self._parameters

    def _format_result(self) -> pd.DataFrame:

        columns = [
            "model", "dataset", "sections", "prompt", "module",
        ] + self.parameters + [
            "hit", "정답", "예측", "근거", "본문", "결론", "생성질문",
        ]

        result_df = self.result_df[columns]

        result_df["dataset"] = result_df.apply(
            lambda field: field["dataset"].split("_")[0],
            axis=1,
        )

        return result_df

    def _format_deviate(self) -> pd.DataFrame:

        columns = [
            "model", "dataset", "sections", "prompt", "module",
        ] + self.parameters + [
            "정답", "예측", "근거", "본문", "결론", "생성질문",
        ]

        deviate_df = self.result_df[~self.result_df["hit"]][columns]

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
