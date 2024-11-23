import pandas as pd
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from cenai_core import cenai_path
from cenai_core.dataman import load_json_yaml
from cenai_core.grid import GridRunner


profile_dir = cenai_path("python/amc/pdac_summarizer/profile")
profile_file = profile_dir / "amc-poc-otf.yaml"

with st.sidebar:
    st.header("Parameter settings")

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

    run_button = st.button("Run")

states = {}


@st.cache_data
def get_result(profile) -> pd.DataFrame:
    st.cache_data.clear()
    st.cache_resource.clear()

    runner = GridRunner(profile)
    result_df = runner.yield_result()

    return result_df


if run_button:
    result_df = get_result(profile)

    states["type"] = result_df["정답"].tolist()
    states["body"] = result_df["본문"].tolist()
    states["html"] = result_df["html"].tolist()
else:
    for key in ["type", "body", "html"]:
        if key not in states:
            states[key] = []

choice = 0

with st.sidebar:
    st.header("선택")
    choice = st.selectbox(
        "Choose a report:",
        range(len(states["type"])), format_func=lambda i: states["type"][i]
    )

    st.header("본문")
    if states["body"]:
        st.write(states["body"][choice])

st.header("요약")
if states["html"]:
    components.html(states["html"][choice], height=4800)
