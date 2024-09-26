from langchain_community.llms import LlamaCpp
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
