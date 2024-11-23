import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import LangchainHelper, load_dotenv

model_name = "HCX-003"

load_dotenv()

LangchainHelper.bind_model(model_name)
model = LangchainHelper.load_model()

prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        """
        아래 문장을 요약하여 json 형태로 출력해 주세요.
        * json 형태:
        {{
            "발생일자": 사건이 발생한 날짜
            "요약": 사건이 발생한 날짜
        }}

        * 요약할 문장:
        {input}

        """
    ),
])

chain = prompt | model 

answer = chain.invoke({
    "input":
"""
중남미 페루에서 납치 신고가 접수된 우리 국민 1명이 납치 하루 만에 현지 경찰에 의해 무사히 구출됐다. 
25일(현지시간) 페루 현지 매체 등에 따르면 페루 경찰은 이날 페루 현지인에 의해 납치된 60대 한국인 사업가 A씨를 총격전 끝에 구조했다. A씨는 현재 병원으로 이송된 상태로 안전이 확인됐다고 전해진다. 
페루 매체 안디나통신은 현지 경찰이 추격 끝에 범죄조직에 납치된 한국인을 구출했고 납치 용의자 3명을 체포했다고 보도했다. 
외교부에 따르면 24일 새벽 페루의 수도 리마에서 A씨가 납치됐다는 신고가 주페루대사관에 접수됐고, 외교부는 보고를 받은 후 재외국민보호대책반을 가동, 재외국민보호대책본부로 격상해 김홍균 외교부 1차관 주재로 회의를 통해 현지 상황과 우리 국민의 안전 확보 대책을 논의했다.
현지 공관은 현장 지휘 본부를 설치하고 페루 경찰청 및 가족들과 긴밀히 소통하면서 필요한 영사 조력을 제공한 것으로 알려졌다.
"""
})

deserial = json.loads(answer.content)

print(deserial)