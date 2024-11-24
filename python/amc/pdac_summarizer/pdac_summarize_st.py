import pandas as pd
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from cenai_core import cenai_path
from cenai_core.dataman import load_json_yaml
from cenai_core.grid import GridRunner

if "result" not in st.session_state:
    st.session_state.result = {
        "gt_type": [],
        "content": [],
        "html": [],
    }

profile_dir = cenai_path("python/amc/pdac_summarizer/profile")
profile_file = profile_dir / "amc-poc-otf.yaml"

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


    profile = load_json_yaml(profile_file)

    profile["directive"]["num_selects"] = num_selects
    profile["model"] = [model]
    profile["corpora"][0]["stem"] = [Path(pdac_report).stem]
    profile["corpora"][0]["extension"] = [Path(pdac_report).suffix]
    profile["corpora"][0]["seeds"] = [seeds]

    run_button = st.button("Run", use_container_width=True)

@st.cache_data
def get_result(profile) -> pd.DataFrame:
    runner = GridRunner(profile)
    result_df = runner.yield_result()

    return result_df

if run_button:
    result_df = get_result(profile)

    result = {
        key: result_df[key].tolist()
        for key in ["gt_type", "content", "html"]
    }
    st.session_state.result = result

else:
    result = st.session_state.result

with st.sidebar:
    st.subheader("선택")
    choice = st.selectbox(
        "Choose a report:",
        range(len(result["gt_type"])),
        format_func=lambda i: result["gt_type"][i]
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

st.subheader("요약")
if result["html"]:
    components.html(result["html"][choice], height=4800)
