from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import (LangchainHelper, load_dotenv)


load_dotenv()

model_name = "gpt-4o"

LangchainHelper.bind_model(model_name)
model = LangchainHelper.load_model()
embeddings = LangchainHelper.load_embeddings()

prompt = ChatPromptTemplate.from_template(
    """
    입력된 문장에 대해 아래의 예에서 제시한 유형을 대답하세요.

    문장: 사과
    유형: 둥글다

    문장: 콤파스
    유형: 세모나다

    문장: 바나나
    유형: 길다

    문장: 모니터
    유형: 네모나다

    문장: 국수
    유형: 길다

    문장: 공책
    유형: 네모나다

    문장: 축구공
    유형: 둥글다

    문장: 막대기
    유형: 길다

    문장: 기린
    유형: 길다

    문장: 삼각자
    유형: 세모나다

    문장: {input}
    유형:
    """,
)

chain = prompt | model | StrOutputParser()

answer = chain.invoke(
    {"input": "지팡이"}
)
print(answer)

answer = chain.invoke(
    {"input": "트라이앵글"}
)
print(answer)
