from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import (LangchainHelper, load_dotenv)


load_dotenv()

model_name = "gpt-3.5-turbo"
model_name = "llama3.1:latest"
model_name = "llama3.1:70b"

LangchainHelper.bind_model(model_name)
model = LangchainHelper.load_model()

template = "{country}의 수도는 어디야?"

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     당신은 한국어로 간결하고 명료하게 답변해야 합니다.
     답변은 아래 예와 같습니다.
     
     {country}에 대한 질문 외에는 아무런 말도 하지 말아야 합니다.

     답변 예: 한국의 수도는 서울입니다.
     """
     ),
    ("human", template),
])

chain = prompt | model | StrOutputParser()

answers = chain.batch([
    {"country": "대한민국"},
    {"country": "일본"},
    {"country": "미국"},
    {"country": "중국"},
])

for answer in answers:
    print(answer)
