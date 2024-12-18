import streamlit as st
import random
import numpy as np
import time

st.title("Echo Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []


def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )

    for word in response.split():
        yield word + " "
        time.sleep(0.05)

if prompt := st.chat_input("What is up?"):
    with st.chat_message("human"):
        st.markdown(prompt)

        st.session_state.messages.append(
            {"role": "human", "content": prompt}
        )

    response = f"Echo: {prompt}"

    with st.chat_message("assistant"):
        st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
