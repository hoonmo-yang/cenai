from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler

from cenai_core import (LangchainHelper, load_dotenv)


class MyHandler(BaseCallbackHandler):
    def __init__(self):
        self._result = ""

    def on_llm_end(self,
                   response,
                   **kwargs
                   ) -> None:
        self._result = response

    @property
    def result(self) -> str:
        return self._result


handler = MyHandler()

load_dotenv(False)

model_name = "gpt-3.5-turbo"

LangchainHelper.bind_model(model_name)
model = LangchainHelper.load_model()

chat_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
])

chain = (
    chat_prompt |
    model |
    StrOutputParser().with_config(
        callbacks=[handler],
    ) |
    model | StrOutputParser()
)

print(chain.invoke(
    {"question": "꽃이름 5개를 대고 그 꽃의 공통점이 뭐냐는 질문을 만들어 주세요.."},
))

print(handler.result)