
import streamlit as st

from cenai_core import cenai_path
from cenai_core import Logger
from cenai_core.dataman import compact_list, load_json_yaml
from cenai_core.grid import GridRunner


class PJChatbotStreamlit(Logger):
    logger_name = "cenai.system"
    profile_dir = cenai_path("python/amc/pj_chatbot/profile")
    profile_file = profile_dir / "amc-poc-otf.yaml"

    runner = GridRunner()

    def __init__(self):
        st.set_page_config(
            layout="wide",
        )

        self._change_parameter_values()

        if "messages" not in st.session_state:
            st.session_state.messages = []

    def _change_parameter_values(self):
        self._profile = load_json_yaml(self.profile_file)

        with st.sidebar:
            st.subheader("파라미터 세팅")

            model = st.selectbox(
                "Select LLM model",
                ["gpt-4o", "gpt-3.5-turbo",]
            )

        self._profile["models"] = [[model]]

    def invoke(self):
        st.title("Patient Q/A Chatbot")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("질문"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                streams = self.runner.stream(
                    messages=st.session_state.messages,
                )

                response = st.write_stream(streams[0])

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.messages = compact_list(st.session_state.messages, 5, 5)
