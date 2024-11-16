from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser

from cenai_core import (cenai_path, LangchainHelper, load_dotenv)


load_dotenv()

model_name = "gpt-3.5-turbo"
model_name = "llama3.1:latest"
model_name = "llama3.1:70b"

LangchainHelper.bind_model(model_name)
model = LangchainHelper.load_model()

content_dir = cenai_path("data/aux/example/content")
prompt_path = content_dir / "pt-capital.yaml"
prompt = load_prompt(str(prompt_path))

chain = prompt | model | StrOutputParser()

print(chain.invoke({"country": "대한민국"}))
