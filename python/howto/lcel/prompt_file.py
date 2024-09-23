from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser

from cenai_core import (cenai_path, load_dotenv)


load_dotenv()

prompt_path = cenai_path("prompt") / "capital.yaml"
prompt = load_prompt(str(prompt_path))

llm = ChatOllama(
    model="llama3.1:latest",
)

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"country": "대한민국"}))
