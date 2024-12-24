from typing import Any

from io import BytesIO
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components

from cenai_core import cenai_path
from cenai_core import Logger
from cenai_core.dataman import generate_zip_buffer, get_empty_html, load_json_yaml
from cenai_core.grid import GridRunner


class PJSummarizationStreamlit(Logger):
    logger_name = "cenai.system"
    profile_dir = cenai_path("python/amc/pj_summarizer/profile")
    profile_file = profile_dir / "amc-poc-otf.yaml"

    def __init__(self):
        if "result" not in st.session_state:
            st.session_state.result = {
                "resch_pat_id": [],
                "html": [],
            }

        if "runner" not in st.session_state:
            st.session_state.runner = GridRunner()

        self._profile = self._change_parameter_values()
        self._activate_runner(self.profile)

    @classmethod
    def _change_parameter_values(cls) -> dict[str, Any]:
        profile = load_json_yaml(cls.profile_file)

        with st.sidebar:
            st.subheader("파라미터 세팅")

            model = st.selectbox(
                "Select LLM model",
                ["gpt-4o", "gpt-3.5-turbo",]
            )

        label = f"{datetime.now().strftime("%Y-%m-%d")}"

        profile["metadata"]["label"] = label
        profile["models"] = [[model]]

        return profile

    @staticmethod
    @st.cache_resource
    def _activate_runner(profile: dict[str, Any]) -> None:
        st.session_state.runner.update(profile)
        st.session_state.runner.activate()

    def invoke(self):
        with st.sidebar:
            if st.button("Run", use_container_width=True):
                st.session_state.result = self._get_result(self.profile)

            result = st.session_state.result

            if st.button(
                "Generate Documents", use_container_width=True
            ) and result["resch_pat_id"]:

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
                st.success("Cache Cleared")

            st.subheader("파일 선택")

            choice = st.selectbox(
                "Choose a Patient ID:",
                range(len(result["resch_pat_id"])),
                format_func=lambda i: result["resch_pat_id"][i]
            )

        html = (
            result["html"][choice] if result["resch_pat_id"] else
            get_empty_html()
        )

        st.subheader("환자 여정 요약")
        components.html(html, height=4800)

    @staticmethod
    @st.cache_data
    def _get_result(profile: dict[str, Any]) -> dict[str, Any]:
        runner = st.session_state.runner
        runner.update(profile)

        result_df = runner.yield_result()

        result = {
            key: result_df[key].tolist()
            for key in [
                "resch_pat_id",
                "html",
            ]
        }

        result["resch_pat_id"] = [
            f"{resch_pat_id:010d}" for resch_pat_id in
            result["resch_pat_id"]
        ]

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
    pj_summarizer = PJSummarizationStreamlit()
    pj_summarizer.invoke()


if __name__ == "__main__":
    main()
