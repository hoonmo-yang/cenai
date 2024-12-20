from typing import Any

from io import BytesIO
from datetime import datetime
import pandas as pd
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from cenai_core import cenai_path
from cenai_core import Logger
from cenai_core.dataman import generate_zip_buffer, get_empty_html, load_json_yaml
from cenai_core.grid import GridRunner


class PDACSummarizationStreamlit(Logger):
    logger_name = "cenai.system"
    profile_dir = cenai_path("python/amc/pdac_summarizer/profile")
    profile_file = profile_dir / "amc-poc-otf.yaml"

    runner = GridRunner()

    def __init__(self):
        st.set_page_config(
            layout="wide",
        )

        self._profile = load_json_yaml(self.profile_file)

        if "result" not in st.session_state:
            st.session_state.result = {
                "select_sample": [],
                "content": [],
                "html": [],
            }

        self._run_button = None

    def _change_parameter_values(self):
        with st.sidebar:
            st.subheader("파라미터 세팅")

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

            self._document_button = st.button(
                "Generate documents", use_container_width=True
            )

        label = f"{datetime.now().strftime("%Y-%m-%d")}"

        self._profile["metadata"]["label"] = label
        self._profile["directive"]["num_selects"] = num_selects
        self._profile["models"] = [[model]]
        self._profile["corpora"][0]["stem"] = [Path(pdac_report).stem]
        self._profile["corpora"][0]["extension"] = [Path(pdac_report).suffix]
        self._profile["corpora"][0]["seeds"] = [seeds]

    @staticmethod
    @st.cache_data
    def _get_result(profile: dict[str, Any]) -> dict[str, Any]:
        runner = PDACSummarizationStreamlit.runner
        runner.update(profile)
        result_df = runner.yield_result()

        result_df["select_sample"] = result_df.apply(
            lambda field:
                f"Sample {field.sample_id:03d} [{field.gt_type}]",
                axis=1
        )

        result = {
            key: result_df[key].tolist()
            for key in ["select_sample", "content", "html"]
        }

        return result

    @staticmethod
    @st.cache_data
    def _generate_documents(profile: dict[str, Any]) -> tuple[BytesIO, str]:
        runner = PDACSummarizationStreamlit.runner
        runner.update(profile)

        export_dir, extensions = runner.export_documents()

        files = [
            file_ for extension in extensions
            for file_ in export_dir.glob(f"*{extension}")
            if file_.is_file()
        ]

        zip_buffer = generate_zip_buffer(files)

        for file_ in files:
            file_.unlink()

        return [zip_buffer, f"{runner.suite_id}.zip"]

    def invoke(self):
        self._change_parameter_values()

        if self._run_button:
            st.session_state.result = self._get_result(self._profile)
            self._profile["directive"]["force"] = False
            self._profile["directive"]["truncate"] = False

        result = st.session_state.result

        with st.sidebar:
            if self._document_button and result["select_sample"]:
                data, file_name = self._generate_documents(self._profile)
                st.download_button(
                    label="Download ZIP file",
                    data=data,
                    file_name=file_name,
                    mime="application/zip",
                    use_container_width=True,
                )

            if st.button("Clear Cache", use_container_width=True):
                st.cache_data.clear()
                self._profile["directive"]["force"] = True
                self._profile["directive"]["truncate"] = True

                st.success("Cache ias been cleared")

            st.subheader("샘플 선택")

            choice = st.selectbox(
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
                        {result['content'][choice]}</div>""",
                    unsafe_allow_html=True
                )

        if result["html"]:
            html = result["html"][choice]

        else:
            html = get_empty_html()

        st.subheader("영상 판독문 요약")
        components.html(html, height=4800)


def main():
    pdac_summarizer = PDACSummarizationStreamlit()
    pdac_summarizer.invoke()


if __name__ == "__main__":
    main()
