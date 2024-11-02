from operator import attrgetter, itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import (LangchainHelper, load_dotenv)

load_dotenv(False)

model_name = "gpt-3.5-turbo"

LangchainHelper.bind_model(model_name)
model = LangchainHelper.load_model()

prompt = PromptTemplate.from_template(
"""
{question}

{tag}
"""
)

chain = (
    {"question": itemgetter("question") | model | attrgetter("content"),
     "tag": itemgetter("tag")
    } | prompt |
    attrgetter("text")
)

response = chain.invoke({
    "question": "꽃 이름 3개만 말해줘",
    "tag": "hello world"
})

print(response)