from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from cenai_core import (LangchainHelper, load_dotenv)
from cenai_core import Timer


load_dotenv()

model_name = "gpt-3.5-turbo"
model_name = "llama3.1:latest"
model_name = "llama3.1:70b"

LangchainHelper.bind_model(model_name)
model = LangchainHelper.load_model()

prompt = PromptTemplate.from_template(
    "{country}에 대해서 200자 내외로 요약해줘."
)

chain = prompt | model | StrOutputParser()

timer = Timer()

countries = ["한국", "한국", "한국"]

print("without cache")
for i in range(3):
    print(f"{i + 1} trial")

    timer.start()

    answer = chain.invoke({"country": countries[i]})
    print(answer)

    timer.lap()

    print(f"elapsed time: {timer.seconds}S")

set_llm_cache(InMemoryCache())

print("with cache")
for i in range(3):
    print(f"{i + 1} trial")

    timer.start()

    answer = chain.invoke({"country": countries[i]})
    print(answer)

    timer.lap()

    print(f"elapsed time: {timer.seconds}S")
