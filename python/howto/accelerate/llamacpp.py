from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import (cenai_path, load_dotenv, Timer)


load_dotenv()

template = """
아래 질문에 답변만 해.
답변은 아래와 나온 예처럼 하고 그외 다른 말은 절대 하지마.

#질문: {country}의 수도는 어디야?
#답변의 예: 한국의 수도는 서울입니다.
"""

prompt = ChatPromptTemplate.from_template(template)

model_path = cenai_path("model/meta-llama-3.1-8B-instruct-Q8_0.gguf")

llm = LlamaCpp(
    model_path=str(model_path),
    # max_tokens=128,
    # n_batch=512,
    # n_ctx=2048,
    n_gpu_layers=-1,
    # temperature=0.7,
    verbose=False,
)


chain = prompt | llm | StrOutputParser()

timer = Timer()

print(chain.invoke({
    "country": "대한민국",
}))

timer.lap()

print(f"elapsed: {timer.seconds}S")

timer.start()

print(chain.invoke({
    "country": "일본",
}))

timer.lap()

print(f"elapsed: {timer.seconds}S")
