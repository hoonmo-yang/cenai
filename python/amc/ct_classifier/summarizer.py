from datetime import datetime
from pathlib import Path
import pandas as pd
import re

from cenai_core import cenai_path

class Summarizer:
    def __init__(self):
        self._output_dir = cenai_path(
            "output/amc"
        )

        self._columns = (
            "model",
            "algorithm",
            "dataset",
            "version",
            "test_size",
            "seed",
            "topk",
            "hit_ratio",
            "hit",
            "total",
        )

        self._result_df = pd.DataFrame()

        self.P_RESULT = re.compile(
            r"HIT RATIO:(\d+\.\d+)% HIT:(\d+) TOTAL:(\d+)"
        )

        self.P_FILE = re.compile(
            r"(.+)_(.+)_(.+)_(\d+)-(\d+)_(\d+)_k(\d+)\.txt"
        )

    def create_results(self) -> None:
        self._result_df = self._extract_results()

    def report_results(self) -> None:
        pivot_df = pd.pivot_table(
            self._result_df,
            values=["hit_ratio"],
            index=["model", "algorithm", "topk"],
            columns=["dataset", "test_size", "version"],
            aggfunc=["mean", "std"],
            fill_value=0,
        )

        result_df = self._result_df.sort_values(
            by=[
                "model",
                "algorithm",
                "dataset",
                "version",
                "test_size",
                "topk",
                "seed",
            ]
        ).reset_index()

        time = datetime.now().strftime("%Y-%m-%dT%H:%M")
        excel_dir = cenai_path("output/amc/excel")
        excel_file = (
            excel_dir /
            f"summary_{time}.xlsx"
        )

        with pd.ExcelWriter(excel_file) as writer:
            pivot_df.to_excel(writer, sheet_name="Summary")
            result_df.to_excel(writer, sheet_name="Test List")

    def _extract_results(self) -> pd.DataFrame:
        result_df = pd.DataFrame()

        for text_file in self._output_dir.rglob("*.txt"):
            new_df = self._extract_result(text_file)

            result_df = pd.concat(
                [result_df, new_df], axis=0
            )

        return result_df.reset_index()

    def _extract_result(self, source: Path) -> pd.DataFrame:
        line = [
            line for line in source.read_text(encoding="utf-8").split("\n")
            if "HIT RATIO" in line
        ][0].strip()

        match = self.P_RESULT.search(line)
        hit_ratio = float(match[1]) if match else None
        hit = int(match[2]) if match else None
        total = int(match[3]) if match else None

        algorithm = source.parent.parent.name

        match = self.P_FILE.search(source.name)
        model = match[1] if match else None
        dataset = match[2] if match else None
        version = match[3] if match else None
        train = int(match[4]) if match else None
        test = int(match[5]) if match else None
        seed = int(match[6]) if match else None
        topk = int(match[7]) if match else None

        test_size = (
            None if test is None or train is None else test / (train + test)
        )

        return pd.DataFrame({
            "algorithm": [algorithm],
            "model": [model],
            "dataset": [dataset],
            "version": [version],
            "test_size": [test_size],
            "seed": [seed],
            "topk": [topk],
            "hit_ratio": [hit_ratio],
            "hit": [hit],
            "total": [total],
        })


def main() -> None:
    summarizer = Summarizer()
    summarizer.create_results()
    summarizer.report_results()


if __name__ == "__main__":
    main()