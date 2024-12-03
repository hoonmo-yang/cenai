from typing import Any

import pandas as pd
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from cenai_core import cenai_path
from cenai_core import Logger
from cenai_core.dataman import get_empty_html, load_json_yaml
from cenai_core.grid import GridRunner


class PaperSummarizationStreamlit(Logger):
    logger_name = "cenai.system"
    profile_dir = cenai_path("python/nrf/paper_summarizer/profile")
    profile_file = profile_dir / "nrf-poc-otf.yaml"

    def __init__(self):
        st.set_page_config(
            layout="wide",
        )

        self._profile = load_json_yaml(self.profile_file)

        if "result" not in st.session_state:
            st.session_state.result = {
                "select_file": [],
                "html_gt": [],
                "html_pv": [],
                "html_eval": [],
            }

        self._result = st.session_state.result
        self._choice = None
        self._run_button = None

    def change_parameter_values(self):
        with st.sidebar:
            st.subheader("Parameter settings")

            model = st.selectbox(
                "Select LLM model",
                ["gpt-4o", "hcx-003",]
            )

            module = st.selectbox(
                "Select module",
                ["stuff_summarizer", "map_reduce_summarizer",]
            )

            prefix = st.text_input(
                "Enter sub-folder:", "sample"
            )

            extension = st.selectbox(
                "Select file extension",
                [".hwp", ".hwpx", ".pdf", ".doc", ".*"]
            )

            self._run_button = st.button("Run", use_container_width=True)

        self._profile["models"] = [[model, "gpt-4o"]]
        self._profile["corpora"][0]["prefix"] = [prefix]
        self._profile["corpora"][0]["extension"] = [extension]
        self._profile["cases"][0]["module"] = [module]

        if module == "map_reduce_summarizer":
            self._profile["cases"][0]["parameter"].append("num_map_reduce_tokens")

    @staticmethod
    @st.cache_data
    def get_result(profile: dict[str, Any]) -> pd.DataFrame:
        runner = GridRunner(profile)
        result_df = runner.yield_result()

        return result_df

    def invoke(self):
        self.change_parameter_values()

        if self._run_button:
            result_df = self.get_result(self._profile)

            result_df["select_file"] = result_df.file.apply(
                lambda field: Path(field).name
            )

            result = {
                key: result_df[key].tolist()
                for key in [
                    "select_file", "html_gt", "html_pv", "html_eval",
                ]
            }
            st.session_state.result = result

        else:
            result = st.session_state.result

        with st.sidebar:
            st.subheader("파일 선택")

            choice = st.selectbox(
                "Choose a file:",
                range(len(result["select_file"])),
                format_func=lambda i: result["select_file"][i]
            )

        if result["select_file"]:
            html_pv = result["html_pv"][choice]
            html_gt = result["html_gt"][choice]
            html_eval = result["html_eval"][choice]

        else:
            html_pv = get_empty_html()
            html_gt = get_empty_html()
            html_eval = get_empty_html()

        st.subheader("요약 비교")
        components.html(html_eval, height=350)

        left, right = st.columns(2)

        with left:
            st.subheader("생성 요약")
            components.html(html_pv, height=4800)

        with right:
            st.subheader("정답 요약")
            components.html(html_gt, height=4800)

def main():
    paper_summarizer = PaperSummarizationStreamlit()
    paper_summarizer.invoke()


if __name__ == "__main__":
    main()
