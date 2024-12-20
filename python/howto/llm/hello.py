from langchain_core.output_parsers import StrOutputParser

from cenai_core.langchain_helper import LangchainHelper

model_names = [
    "llama3.2",
    "gemma2",
]

model = {}

for model_name in model_names:
    LangchainHelper.bind_model(model_name)
    model[model_name] = LangchainHelper.load_model()

for model_name in model_names:
    chain = (
        model[model_name] | StrOutputParser()
    )

    response = chain.invoke("안녕하세요?")

    print(f"{model_name}: {response}")

