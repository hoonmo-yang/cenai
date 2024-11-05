import streamlit as st
import time

with st.sidebar:
    add_selectbox = st.selectbox(
        "How would you like to be contacted?",
        ("Email", "Home Phone", "Mobile Phone")
    )
    with st.echo():
        st.write("This code will be printed to the sidebar.")

    with st.spinner("Loading..."):
        time.sleep(1)
    st.success("Done")

tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

with tab1:
    st.write("hmyang71@gmail.com")

with tab2:
    st.write("02-2021-0327")

with tab3:
    st.write("010-6234-9174")

left, middle, right = st.columns(3)
if left.button("Plain button", use_container_width=True):
    left.markdown("You cliked the plain button.")
if middle.button("Emoji button", icon="😃", use_container_width=True):
    middle.markdown("You cliked the emoji button.")
if right.button("Material button", icon=":material/mood:", use_container_width=True):
    right.markdown("You cliked the Material button.")