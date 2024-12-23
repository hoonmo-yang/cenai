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

    runner = GridRunner()

    def __init__(self):
        st.set_page_config(
            layout="wide",
        )

        if "result" not in st.session_state:
            st.session_state.result = {
                "resch_pat_id": [],
                "html": [],
            }

        self._change_parameter_values()

    def _change_parameter_values(self):
        self._profile = load_json_yaml(self.profile_file)

        with st.sidebar:
            st.subheader("파라미터 세팅")

            model = st.selectbox(
                "Select LLM model",
                ["gpt-4o", "gpt-3.5-turbo",]
            )

            self._run_button = st.button("Run", use_container_width=True)

            self._document_button = st.button(
                "Generate documents", use_container_width=True
            )

        label = f"{datetime.now().strftime("%Y-%m-%d")}"

        self._profile["metadata"]["label"] = label
        self._profile["models"] = [[model]]

    @staticmethod
    @st.cache_data
    def _get_result(profile: dict[str, Any]) -> dict[str, Any]:
        runner = PJSummarizationStreamlit.runner
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
        runner = PJSummarizationStreamlit.runner
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
        if self._run_button:
            st.session_state.result = self._get_result(self._profile)
            self._profile["directive"]["force"] = False
            self._profile["directive"]["truncate"] = False

        result = st.session_state.result

        with st.sidebar:
            if self._document_button and result["resch_pat_id"]:
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

            st.subheader("파일 선택")

            choice = st.selectbox(
                "Choose a Patient ID:",
                range(len(result["resch_pat_id"])),
                format_func=lambda i: result["resch_pat_id"][i]
            )

        if result["resch_pat_id"]:
            html = result["html"][choice]

        else:
            html = get_empty_html()

        st.subheader("환자 여정 요약")
        components.html(html, height=4800)


def main():
    pj_summarizer = PJSummarizationStreamlit()
    pj_summarizer.invoke()


if __name__ == "__main__":
    main()
