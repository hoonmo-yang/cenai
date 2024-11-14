import pandas as pd
from pathlib import Path
import re

from cenai_core.pandas_helper import from_json

class PDACRecapper:
    def __init__(self, grid_dir: Path):
        self._output_dir = grid_dir
        self._run_df, self._parameters = self._merge_data()

    def _merge_data(self) -> tuple[pd.DataFrame, list[str]]:
        prefix = self._output_dir.name
        base_dir = self._output_dir / "datastore"

        metadata_df = pd.DataFrame()
        result_df = pd.DataFrame()

        for datastore_dir in base_dir.glob(f"{prefix}*"):
            if not datastore_dir.is_dir():
                continue

            for datastore_json in datastore_dir.glob("*.json"):
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

        metadata_df["dataset"] = metadata_df.apply(
            lambda field: "".join(field["dataset"].split("_")[0]),
            axis=1,
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

        result_df["run_group"] = result_df.apply(
            lambda field:
                re.sub(r"_(\d{2})_", "_", field["run_id"]),
                axis=1,
        )

        result_df["hits"] = result_df.groupby(["grid_id", "run_group"])["hit"].transform("sum")
        result_df["total"] = result_df.groupby(["grid_id", "run_group"])["hit"].transform("count")
        result_df["hit_ratio"] = 100 * result_df["hits"] / result_df["total"]

        run_df = pd.merge(
             result_df,
             metadata_df,
             on=["grid_id", "run_id"],
             how="outer",
        )

        return run_df, parameters

    def export_excel(self):
        for grid_id, run_df in self.run_df.groupby(["grid_id"]):
            excel_dir = self._output_dir / "excel" 
            excel_dir.mkdir(parents=True, exist_ok=True)
            excel_file = excel_dir / f"{grid_id[0]}.xlsx"

            statistics_df = self._format_statistics(run_df)
            result_df = self._format_result(run_df)
            deviate_df = self._format_deviate(run_df)

            with pd.ExcelWriter(excel_file) as writer:
                statistics_df.to_excel(writer, sheet_name="Statistics")
                result_df.to_excel(writer, sheet_name="Results")
                deviate_df.to_excel(writer, sheet_name="Deviates")

    def _format_statistics(self, run_df: pd.DataFrame) -> pd.DataFrame:
        value_order = ["hits", "total", "hit_ratio", "소요시간"]

        statistic_df = pd.pivot_table(
            run_df,
            values=["hits", "total", "hit_ratio", "소요시간"],
            index=["model", "module", "prompt"] + self.parameters,
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

    def _format_result(self, run_df: pd.DataFrame) -> pd.DataFrame:
        candidates = [
            "model", "dataset", "sections", "prompt", "module",
        ] + self.parameters + [
            "hit", "정답", "예측", "근거", "본문", "결론", "생성질문",
        ]

        columns = [
            column
            for column in candidates
            if column in run_df.columns
        ]

        return run_df[columns]

    def _format_deviate(self, run_df: pd.DataFrame) -> pd.DataFrame:
        candidates = [
            "model", "dataset", "sections", "prompt", "module",
        ] + self.parameters + [
            "정답", "예측", "근거", "본문", "결론", "생성질문",
        ]

        columns = [
            column
            for column in candidates
            if column in run_df.columns
        ]

        deviate_df = run_df[~run_df["hit"]][columns]

        deviate_df["dataset"] = deviate_df.apply(
            lambda field: "".join(field["dataset"].split("_")[:2]),
            axis=1,
        )

        return deviate_df

    @property
    def run_df(self) -> pd.DataFrame:
        return self._run_df

    @property
    def parameters(self) -> list[str]:
        return self._parameters
