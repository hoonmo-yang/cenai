from typing import Any

import pandas as pd
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from cenai_core import cenai_path
from cenai_core import Logger
from cenai_core.dataman import load_json_yaml
from cenai_core.grid import GridRunner


class PDACSummarizationStreamlit(Logger):
    logger_name = "cenai.system"
    profile_dir = cenai_path("python/amc/pdac_summarizer/profile")
    profile_file = profile_dir / "amc-poc-otf.yaml"

    def __init__(self):
        self._profile = load_json_yaml(self.profile_file)

        if "result" not in st.session_state:
            st.session_state.result = {
                "select_sample": [],
                "content": [],
                "html": [],
            }

        self._result = st.session_state.result
        self._choice = None
        self._run_button = None

    def change_parameter_values(self):
        with st.sidebar:
            st.subheader("Parameter settings")

            model = st.selectbox(
                "Select LLM model", 
                ["gpt-4o", "gpt-3.5-turbo",]
            )

            seeds = st.number_input(
                "Enter a random number",
                min_value=0,
                max_value=1000,
                value=0,
                step=1,
            )

            num_selects = st.number_input(
                "Select number of samples per type",
                min_value = 1,
                max_value = 20,
                value = 1,
                step = 1,
            )

            pdac_report = st.selectbox(
                "Select PDAC report", 
                ["pdac-report4.json",]
            )

            self._run_button = st.button("Run", use_container_width=True)

        self._profile["directive"]["num_selects"] = num_selects
        self._profile["model"] = [model]
        self._profile["corpora"][0]["stem"] = [Path(pdac_report).stem]
        self._profile["corpora"][0]["extension"] = [Path(pdac_report).suffix]
        self._profile["corpora"][0]["seeds"] = [seeds]

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

            result_df["select_sample"] = result_df.apply(
                lambda field:
                    f"Sample {field['sample']:02d} [{field['gt_type']}]",
                    axis=1
            )

            result = {
                key: result_df[key].tolist()
                for key in ["select_sample", "content", "html"]
            }
            st.session_state.result = result

        else:
            result = st.session_state.result

        with st.sidebar:
            st.subheader("선택")

            self._choice = st.selectbox(
                "Choose a report:",
                range(len(result["select_sample"])),
                format_func=lambda i: result["select_sample"][i]
            )

            if result["content"]:
                st.subheader("본문")

                st.markdown(
                """
                <style>
                .custom-text-box {
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    padding: 15px;
                    background-color: #f9f9f9;
                    font-size: 16px;
                    line-height: 1.6;
                    color: #333;
                    overflow-wrap: break-word;
                    text-align: justify;
                }
                </style>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                    f"""<div class="custom-text-box">
                        {result['content'][self._choice]}</div>""",
                    unsafe_allow_html=True
                )

        st.subheader("요약")
        if result["html"]:
            components.html(result["html"][self._choice], height=4800)


def main():
    pdac_summarizer = PDACSummarizationStreamlit()
    pdac_summarizer.invoke()


if __name__ == "__main__":
    main()
