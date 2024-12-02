from langchain_core.prompts import PromptTemplate

from cenai_core import load_dotenv
from cenai_core.langchain_helper import LangchainHelper

content = "이 연구 논문의 목적은 리튬 이온 전지의 간극 조정에 따른 효율을 개선하는 것이다."

load_dotenv()

LangchainHelper.bind_model("gpt-4o")
model = LangchainHelper.load_model()

num_tokens = model.get_num_tokens(content)
chars = len(content)

print(chars, num_tokens, num_tokens/chars)

LangchainHelper.bind_model("hcx-003")
model = LangchainHelper.load_model()

num_tokens = model.get_num_tokens(content)
chars = len(content)

print(chars, num_tokens, num_tokens/chars)

print("hi")