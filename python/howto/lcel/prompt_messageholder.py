from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.output_parsers import StrOutputParser

from cenai_core import load_dotenv


load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 요약 전문 AI 어시스턴스입니다. 당신의 임부는 주요 키워드로 대화를 요약하는 것입니다."
    ),
    MessagesPlaceholder(variable_name="conversation"),
    (
        "human",
        "지금까지의 대화를 {word_count} 단어로 요약합니다."
    ),
])

llm = ChatOllama(
    model="llama3.1:latest",
)

chain = prompt | llm | StrOutputParser()

print(chain.invoke({
    "word_count": 8,
    "conversation": [
        (
            "human",
            "너 어제 왜 회사에 안 나왔어."
        ),
        (
            "ai",
            "감기에 걸려 휴가를 냈습니다."
        ),
        (
            "human",
            "이제 괜찮아?"
        ),
        (
            "ai",
            "예 다시 건강해져서 오늘 출근했습니다."
        ),
    ]
}))
