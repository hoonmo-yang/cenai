from typing import Any

from io import BytesIO
from datetime import datetime
from pathlib import Path
from shutil import rmtree
import streamlit as st
import streamlit.components.v1 as components

from cenai_core import cenai_path
from cenai_core import Logger
from cenai_core.dataman import generate_zip_buffer, get_empty_html, load_json_yaml, Q
from cenai_core.grid import GridRunner


class ResearchResportSummarizationStreamlit(Logger):
    logger_name = "cenai.system"
    profile_dir = cenai_path("python/nrf/research_report_summarizer/profile")
    profile_file = profile_dir / "nrf-poc-otf.yaml"
    corpus_dir = cenai_path("data/nrf/research-report-summarizer/corpus")

    def __init__(self):
        if "runner" not in st.session_state:
            st.session_state.runner = GridRunner()

        if "result" not in st.session_state:
            st.session_state.result = {
                "select_file": [],
                "html": [],
            }

        self._upload_files()

        self._profile = self._change_parameter_values()
        self._activate_runner(self.profile)

    @classmethod
    def _upload_files(cls):
        with st.sidebar:
            st.subheader("파일 관리")

            prefix = st.text_input(
                "Enter folder for file upload:", ""
            )

            placeholder = st.empty()

            uploaded_files = st.file_uploader(
                "File upload",
                ["hwpx", "hwp", "docx", "pdf"],
                label_visibility="collapsed",
                accept_multiple_files=True,
            )

            if uploaded_files:
                placeholder.empty()

                if not prefix:
                    st.error("Set folder to upload before file upload")

                else:
                    upload_dir = cls.corpus_dir / prefix
                    if upload_dir.is_dir():
                        rmtree(upload_dir)

                    upload_dir.mkdir(parents=True, exist_ok=True)

                    for uploaded_file in uploaded_files:
                        target = upload_dir / uploaded_file.name

                        with target.open("wb") as fout:
                            fout.write(uploaded_file.getbuffer())

            dirs = st.multiselect(
                " Select folders to delete:",
                cls._get_dirs(True),
            )

            if st.button("Delete folders", use_container_width=True):
                for dir_ in dirs:
                    rmtree(cls.corpus_dir / dir_)
                    st.success(f"{Q(dir_)} deleted")

    @classmethod
    def _change_parameter_values(cls) -> dict[str, Any]:
        profile = load_json_yaml(cls.profile_file)

        with st.sidebar:
            st.subheader("파라미터 세팅")

            model = st.selectbox(
                "Select LLM model",
                ["gpt-4o", "hcx-003",]
            )

            module = st.selectbox(
                "Select module",
                ["stuff_summarizer", "map_reduce_summarizer",]
            )

            prefix = st.selectbox(
                "Select input folder:", cls._get_dirs(True)
            )

        label = f"{datetime.now().strftime("%Y-%m-%d")}_{prefix}"

        profile["metadata"]["label"] = label
        profile["models"] = [[model, "gpt-4o"]]
        profile["corpora"][0]["prefix"] = [prefix]
        profile["cases"][0]["module"] = [module]

        if module == "map_reduce_summarizer":
            profile["cases"][0]["parameter"].append("num_map_reduce_tokens")

        return profile

    @classmethod
    def _get_dirs(cls, name_only: bool) -> list[Path | str]:
        dirs = [
            dir_ for dir_ in cls.corpus_dir.glob("*")
            if dir_.is_dir()
        ]

        if name_only:
            dirs = [dir_.name for dir_ in dirs]

        return dirs

    @staticmethod
    @st.cache_resource
    def _activate_runner(profile: dict[str, Any]) -> None:
        st.session_state.runner.update(profile)
        st.session_state.runner.activate()

    def invoke(self) -> None:
        with st.sidebar:
            if st.button("Run", use_container_width=True):
                st.session_state.result = self._get_result(self.profile)

            result = st.session_state.result

            if st.button(
                "Generate Documents", use_container_width=True
                ) and result["select_file"]:
            
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

            st.subheader("파일 선택")

            choice = st.selectbox(
                "Choose a file:",
                range(len(result["select_file"])),
                format_func=lambda i: result["select_file"][i]
            )

        html = (
            result["html"][choice] if result["select_file"] else
            get_empty_html()
        )

        st.subheader("요약 비교")
        components.html(html, height=4800)

    @staticmethod
    @st.cache_data
    def _get_result(profile: dict[str, Any]) -> dict[str, Any]:
        runner = st.session_state.runner
        runner.update(profile)

        result_df = runner.yield_result()

        result_df["select_file"] = result_df.file.apply(
            lambda field: Path(field).name
        )

        result = {
            key: result_df[key].tolist()
            for key in [
                "select_file",
                "html",
            ]
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
    research_report_summarizer = ResearchResportSummarizationStreamlit()
    research_report_summarizer.invoke()


if __name__ == "__main__":
    main()
