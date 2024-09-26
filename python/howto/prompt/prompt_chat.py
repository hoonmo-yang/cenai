from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import (LangchainHelper, load_dotenv)


load_dotenv()

model_name = "gpt-3.5-turbo"
model_name = "llama3.1:latest"
model_name = "llama3.1:70b"

LangchainHelper.bind_model(model_name)
model = LangchainHelper.load_model()

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

chain = model | StrOutputParser()

print(chain.invoke(messages))

chain = chat_prompt | model | StrOutputParser()

print(chain.invoke({
    "name": "Teddy",
    "user_input": "당신의 이름은 무엇입니까?",
}))
