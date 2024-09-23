from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import load_dotenv


load_dotenv()

llm = ChatOllama(
    model="llama3.1:latest",
)

template = "{country}의 수도는 어디야?"

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """반드시 한국어로 짧고 명료하게 답변해야 해.
     답변의 형식은 아래와 같아야 해. xxx는 수도의 이름이야.
     아래 답변 이외에는 아무런 대답도 하지 마.

     답변: {country}의 수도는 xxx입니다.
     """
     ),
    ("human", template),
])

chain = prompt | llm | StrOutputParser()

answers = chain.batch([
    {"country": "대한민국"},
    {"country": "일본"},
    {"country": "미국"},
    {"country": "중국"},
])

for answer in answers:
    print(answer)
