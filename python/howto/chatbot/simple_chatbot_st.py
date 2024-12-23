import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import load_dotenv
from cenai_core.dataman import compact_list
from cenai_core.langchain_helper import LangchainHelper


st.title("Simple Chatbot")

with st.sidebar:
    load_dotenv()

    model_name = st.selectbox(
        "Select LLM model",
        ["gpt-4o", "gpt-3.5-turbo", "hcx-003",]
    )

    LangchainHelper.bind_model(model_name)
    model = LangchainHelper.load_model()
    chat_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """당신은 유능하고 친절한 AI 도우미입니다.
            사용자의 질문에 항상 충실하게 답변해야 합니다.
            반드시 한국어를 사용해야 합니다.
            """
        ),
        (
            "placeholder", "{messages}"
        ),
    ])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        chain = chat_prompt | model | StrOutputParser()

        response = st.write_stream(
            chain.stream({
            "messages": st.session_state.messages,
            }
        ))

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.messages = compact_list(st.session_state.messages, 5, 5)
