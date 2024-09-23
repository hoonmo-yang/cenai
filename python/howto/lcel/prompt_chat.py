from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import load_dotenv


load_dotenv()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 {name}입니다."),
    ("human", "반가와요"),
    ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
    ("human", "{user_input}"),
])

messages = chat_prompt.format_messages(
    name="테디",
    user_input="당신의 이름은 무엇입니까?",
)

llm = ChatOllama(
    model="llama3.1:latest",
)

chain = llm | StrOutputParser()

print(chain.invoke(messages))

chain = chat_prompt | llm | StrOutputParser()

print(chain.invoke({
    "name": "Teddy",
    "user_input": "당신의 이름은 무엇입니까?",
}))
