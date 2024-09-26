from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import (cenai_path, load_dotenv, Timer)


load_dotenv()

template = """
질문에 대한 답을 말하고 그 외는 어떤 것도 말하지 않습니다.

질문: {country}의 수도는 어디입니까?
"""

prompt = ChatPromptTemplate.from_template(template)

model_path = cenai_path("model/meta-llama-3.1-8B-instruct-Q8_0.gguf")

llm = ChatOllama(
    model="llama3.1:latest",
)

chain = prompt | llm | StrOutputParser()

timer = Timer()

print(chain.invoke({
    "country": "대한민국",
}))

timer.lap()

print(f"elapsed: {timer.seconds}S")
