from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import (LangchainHelper, load_dotenv)


load_dotenv()

model_name = "gpt-3.5-turbo"
model_name = "llama3.1:latest"
model_name = "llama3.1:70b"

LangchainHelper.bind_model(model_name)

model = LangchainHelper.load_model()

def get_today():
    return datetime.now().strftime("%B %d")


prompt = PromptTemplate(
    template='''
    오늘의 날짜는 {today}입니다.
    오늘이 생일인 유명인 {n}명을 나열해 주세요.
    생년월일을 표기해 주세요.
    ''',
    input_variables=["n"],
    partial_variables={
        "today": get_today
    },
)

chain = prompt | model | StrOutputParser()

print(chain.invoke(1))

print(chain.invoke({"n": 2}))

print(chain.invoke({"today": "Jan 02", "n": 1}))
