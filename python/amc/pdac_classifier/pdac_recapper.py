import pandas as pd
from pathlib import Path
import re

from cenai_core.grid import BaseRunner
from cenai_core.pandas_helper import from_json


class PDACRecapper(BaseRunner):
    def __init__(self, artifact_dir: Path):
        self._output_dir = artifact_dir
        self._case_df, self._parameters = self._merge_data()

    def _merge_data(self) -> tuple[pd.DataFrame, list[str]]:
        prefix = self._output_dir.name
        base_dir = self._output_dir / "datastore"

        metadata_df = pd.DataFrame()
        result_df = pd.DataFrame()

        for datastore_json in base_dir.glob(f"{prefix}*.json"):
            if not datastore_json.is_file():
                continue

            some_metadata_df, some_result_df, *_ = from_json(datastore_json)

            metadata_df = pd.concat(
                [metadata_df, some_metadata_df], axis=0,
            )

            result_df = pd.concat(
                [result_df, some_result_df], axis=0,
            )

        metadata_df = metadata_df.reset_index(drop=True)
        result_df = result_df.reset_index(drop=True)

        metadata_df["tags"] = metadata_df.apply(
             lambda field: ",".join(field.tags),
             axis=1
        )

        parameters = [
            column
            for column in metadata_df.columns
            if column not in [
                "suite_id",
                "case_id",
                "suite_prefix",
                "suite_label",
                "suite_index",
                "institution",
                "task",
                "tags",
                "model",
                "module",
                "corpus_mode",
                "corpus_prefix",
                "corpus_stem",
                "corpus_ext",
                "profile_file",
                "sections",
            ]
        ]

        result_df["hit"] = (result_df["정답"] == result_df["예측"])

        result_df["case_group"] = result_df.apply(
            lambda field:
                re.sub(r"_(\d+)_", "_", field["case_id"]),
                axis=1,
        )

        result_df["hits"] = result_df.groupby(["suite_id", "case_group"])["hit"].transform("sum")
        result_df["total"] = result_df.groupby(["suite_id", "case_group"])["hit"].transform("count")
        result_df["hit_ratio"] = 100 * result_df["hits"] / result_df["total"]

        case_df = pd.merge(
             result_df,
             metadata_df,
             on=["suite_id", "case_id"],
             how="outer",
        )

        return case_df, parameters

    def export_excel(self):
        for suite_id, case_df in self.case_df.groupby(["suite_id"]):
            excel_dir = self._output_dir / "export"
            excel_dir.mkdir(parents=True, exist_ok=True)
            excel_file = excel_dir / f"{suite_id[0]}.xlsx"

            statistics_df = self._format_statistics(case_df)
            result_df = self._format_result(case_df)
            deviate_df = self._format_deviate(case_df)

            with pd.ExcelWriter(excel_file) as writer:
                statistics_df.to_excel(writer, sheet_name="Statistics")
                result_df.to_excel(writer, sheet_name="Results")
                deviate_df.to_excel(writer, sheet_name="Deviates")

    def _format_statistics(self, case_df: pd.DataFrame) -> pd.DataFrame:
        value_order = ["hits", "total", "hit_ratio", "소요시간"]

        statistic_df = pd.pivot_table(
            case_df,
            values=["hits", "total", "hit_ratio", "소요시간"],
            index=["model", "module"] + self.parameters,
            columns=["corpus_prefix", "sections"],
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

    def _format_result(self, case_df: pd.DataFrame) -> pd.DataFrame:
        case_df["corpus"] = case_df.apply(
            lambda field: f"{field['corpus_stem']}{field['corpus_ext']}",
            axis=1
        )

        candidates = [
            "model", "corpus", "sections", "module",
        ] + self.parameters + [
            "hit", "정답", "예측", "근거", "본문", "결론", "생성질문",
        ]

        columns = [
            column
            for column in candidates
            if column in case_df.columns
        ]

        return case_df[columns]

    def _format_deviate(self, case_df: pd.DataFrame) -> pd.DataFrame:
        case_df["corpus"] = case_df.apply(
            lambda field: f"{field['corpus_stem']}{field['corpus_ext']}",
            axis=1
        )

        candidates = [
            "model", "corpus", "sections", "module",
        ] + self.parameters + [
            "정답", "예측", "근거", "본문", "결론", "생성질문",
        ]

        columns = [
            column
            for column in candidates
            if column in case_df.columns
        ]

        deviate_df = case_df[~case_df["hit"]][columns]

        return deviate_df

    def __call__(self) -> None:
        self.export_excel()

    @property
    def case_df(self) -> pd.DataFrame:
        return self._case_df

    @property
    def parameters(self) -> list[str]:
        return self._parameters
