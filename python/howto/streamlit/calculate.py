import streamlit as st

st.title("간단한 계산기")

num1 = st.number_input("첫번째 숫자를 입력하세요:", value=0)
num2 = st.number_input("두번째 숫자를 입력하세요:", value=0)

operation = st.selectbox(
    "연산을 선택하세요",
    ("덧셈", "뺄셈", "곱셈", "나눗셈"),
)

if st.button("계산하기"):
    if operation == "덧셈":
        result = num1 + num2
        st.write(f"결과: {num1} + {num2} = {result}")
    elif operation == "뺄셈":
        result = num1 - num2
        st.write(f"결과: {num1} + {num2} = {result}")
    elif operation == "곱셈":
        result = num1 * num2
        st.write(f"결과: {num1} + {num2} = {result}")
    elif operation == "나눗셈":
        result = num1 / num2
        st.write(f"결과: {num1} + {num2} = {result}")

age = st.slider("나이를 선택하세요:", min_value=0, max_value=100, value=25)
st.write("선택한 나이:", age)