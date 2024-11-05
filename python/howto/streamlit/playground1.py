import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import random

st.json(
    {
        "a": [1, 2, 3],
        "b": {
            "x": "fjd",
            "y": ["a", "b", "c"],
            "z": ["hi"]
        },
        "c": None
    },
    expanded=1,
)

df = pd.DataFrame(
    {
        "name": ["Roadmap", "Extras", "Issues"],
        "url": [
            "https://roadmap.streamlit.app",
            "https://extras.streamlit.app",
            "https://issues.streamlit.app"
        ],
        "stars":
            [random.randint(0, 1000) for _ in range(3)],
        "views_history":
            [[random.randint(0, 5000) for _ in range(30)] for _ in range(3)],
    }
)

st.dataframe(
    df,
    column_config={
        "name": "App name",
        "stars": st.column_config.NumberColumn(
            "Github starts",
            help="Number of stars on GitHub",
            format="%d ⭐",
        ),
        "url": st.column_config.LinkColumn("App URL"),
        "view_history": st.column_config.LineChartColumn(
            "Views (past 30 days)", y_min=0, y_max=5000
        ),
    },
    hide_index=True,
)

df = pd.DataFrame(np.random.randn(10, 20), columns=(f"col {i}" for i in range(20)))

st.dataframe(df.style.highlight_max(axis=0))

st.html(
    "<p><span style='text-decoration: line-through double red;'>Oops</span>!</p>"
)

st.text("Hello World", help="Hello World")

def get_user_name():
    return "훈모"

with st.echo():
    def get_punctuation():
        return "!!!"

    greeting = "안녕, "
    value = get_user_name()
    punctuation = get_punctuation()

    st.write(greeting, value, punctuation)


foo = "bar"
st.write("Done!")

code = """
def hello():
    print("Hello, Streamlit!")  # hello world 안녕하에쇼. ㅓㅏ러ㅏ러ㅏㅇㄹ 러ㅏ어라어라어라ㅣ어ㅏ리ㅓ아ㅣㄹ 러아러ㅏㅇ
"""

st.code(
    code,
    language="python",
    line_numbers=True,
    wrap_lines=True,
)

st.caption("This is a string that explains something above.")
st.caption("A caption with _italics_ :blue[colors] and emojis :sunglasses:")

md = st.text_area(
    "Type in your markdown string (without outer quotes)",
    "Happy Streamlit-ing! :balloon:"
)

st.code(f"""
import streamlit as st

st.markdown('''{md}''')
"""
)

st.markdown(md)



st.markdown("""
# Hello World

## Hi! Everyone

### Hey Everyone
:red[Streamlit] :orange[can] :green[write]
:rainbow[colors] :blue-background[:red[highlight]] text.!!!

#### Hey Everyone

&mdash; :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:
&mdash; :pig::dog::tiger::cat::mouse:

Hello world?


Hello world
hello world
            
hello world
"""
)
st.markdown("*Streamlit* is **really** ***cool***.")

st.title("This is a title 1")
st.title("This is a title 2")
st.title("This is a title 3")

st.title("_This is_ a :blue[title]!! :cry:")

st.subheader("_Streamlit_ is :blue[cool] :sunglasses:")
st.subheader("This is a subheader with a divider", divider="gray")
st.subheader("These subheaders have rotating dividers", divider=True)
st.subheader("One", divider=True)
st.subheader("Two", divider=True)
st.subheader("Three", divider=True)
st.subheader("Four", divider=True)

st.write(1234)
st.write(
    pd.DataFrame(
        {
            "first column": [1, 2, 3, 4],
            "second column": [10, 20, 30, 40],
        }
    )
)

df = pd.DataFrame({
    "a": [1, 1, 2, 2, 3, 3, 4, 5, 6],
    "b": [2, 2, 4, 5, 1, 1, 2, 2, 2],
    "value": [1, 2, 3, 4, 5, 6, 7, 8, 9],
})

pivot_table = pd.pivot_table(
    df,
    values=["value"],
    index=["a"],
    columns=["b"],
    aggfunc=["mean", "std"],
)

st.write("Below is a Pivot table", pivot_table.style.highlight_max(axis=0), "Above is a Pivot Table")

df = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])
c = (
    alt.Chart(df)
    .mark_circle()
    .encode(x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"])
)

st.write(c)
