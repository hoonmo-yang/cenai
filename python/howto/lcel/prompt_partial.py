from datetime import datetime

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import load_dotenv


load_dotenv()


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

llm = ChatOllama(
    model="llama3.1:latest",
)

chain = prompt | llm | StrOutputParser()

print(chain.invoke(1))

print(chain.invoke({"n": 2}))

print(chain.invoke({"today": "Jan 02", "n": 1}))
