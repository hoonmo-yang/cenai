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

    def __init__(self):
        if "result" not in st.session_state:
            st.session_state.result = {
                "select_sample": [],
                "content": [],
                "html": [],
            }

        self._profile = self._change_parameter_values()
        st.session_state.runner = self._get_runner(self.profile)

    @classmethod
    def _change_parameter_values(cls) -> dict[str, Any]:
        profile = load_json_yaml(cls.profile_file)

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

        label = f"{datetime.now().strftime("%Y-%m-%d")}"

        profile["metadata"]["label"] = label
        profile["directive"]["num_selects"] = num_selects
        profile["models"] = [[model]]
        profile["corpora"][0]["stem"] = [Path(pdac_report).stem]
        profile["corpora"][0]["extension"] = [Path(pdac_report).suffix]
        profile["corpora"][0]["seeds"] = [seeds]

        return profile

    @staticmethod
    @st.cache_resource
    def _get_runner(profile: dict[str, Any]) -> GridRunner:
        runner = GridRunner(profile)
        runner.activate()
        return runner

    def invoke(self):
        with st.sidebar:
            if st.button("Run", use_container_width=True):
                st.session_state.result = self._get_result(self.profile)

            result = st.session_state.result

            if st.button(
                "Generate Documents", use_container_width=True
            ) and result["select_sample"]:

                data, file_name = self._generate_documents(self.profile)

                st.download_button(
                    label="Download ZIP file",
                    data=data,
                    file_name=file_name,
                    mime="application/zip",
                    use_container_width=True,
                )

            if st.button("Clear Cache", use_container_width=True):
                st.cache_data.clear()
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

        html = (
            result["html"][choice] if result["html"] else
            get_empty_html()
        )

        st.subheader("영상 판독문 요약")
        components.html(html, height=4800)

    @staticmethod
    @st.cache_data
    def _get_result(profile: dict[str, Any]) -> dict[str, Any]:
        runner = st.session_state.runner
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
        runner = st.session_state.runner
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

    @property
    def profile(self) -> dict[str, Any]:
        return self._profile

def main():
    pdac_summarizer = PDACSummarizationStreamlit()
    pdac_summarizer.invoke()


if __name__ == "__main__":
    main()
