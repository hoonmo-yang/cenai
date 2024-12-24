from typing import Any

import streamlit as st

from cenai_core import cenai_path
from cenai_core import Logger
from cenai_core.dataman import compact_list, load_json_yaml
from cenai_core.grid import GridRunner


class PJChatbotStreamlit(Logger):
    logger_name = "cenai.system"
    profile_dir = cenai_path("python/amc/pj_chatbot/profile")
    profile_file = profile_dir / "amc-poc-otf.yaml"

    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

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

        profile["models"] = [[model]]

        return profile

    @staticmethod
    @st.cache_resource
    def _get_runner(profile: dict[str, Any]) -> GridRunner:
        runner = GridRunner(profile)
        runner.activate()
        return runner

    def invoke(self):
        with st.sidebar:
            if st.button("Clear Messages", use_container_width=True):
                st.session_state.messages = []
                st.success("Messages cleared")

            if st.button("Clear Cache", use_container_width=True):
                st.session_state.messages = []
                st.cache_data.clear()
                st.success("Cache & Messages Cleared")

        st.title("PJ Chatbot")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("질문"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                stream = st.session_state.runner.stream(
                    messages=st.session_state.messages,
                )

                response = st.write_stream(stream)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.messages = compact_list(st.session_state.messages, 5, 5)

    @property
    def profile(self) -> dict[str, Any]:
        return self._profile


def main():
    pj_chatbot = PJChatbotStreamlit()
    pj_chatbot.invoke()


if __name__ == "__main__":
    main()
