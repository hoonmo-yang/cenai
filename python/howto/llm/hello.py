from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from cenai_core import load_dotenv
from cenai_core.langchain_helper import LangchainHelper


load_dotenv()

model_names = [
    "gpt-4o",
    "gpt-3.5-turbo",
#   "hcx-003",
#   "llama3.2",
#   "gemma2",
]

prompt = ChatPromptTemplate([
    ("system",
     "당신은 요약 전문가 AI입니다. 모든 답변은 반드시 한국어로 하십시오."
    ),
    ("user",
     """
     아래 내용을 요약하십시오.

     {content}
     """
     ),
])

content = """
    AI G3'(글로벌 3강)로 꼽히던 한국이 최근 '2군' 평가를 받았다. 한국은 세계에서 3번째로 LLM(거대언어모델)을 개발했지만

    '쩐의 전쟁'에 밀려 경쟁력을 잃은 것 아니냐는 우려가 나온다. 전문가들은 개별 지표에 일희일비하기보단 큰 틀의 AI 마스터플랜을

    짜고 국가적 역량을 결집해야 한다고 주문한다.

    21일 업계에 따르면 보스턴컨설팅그룹(BCG)은 73개국 대상 'AI 성숙도 매트릭스' 평가 보고서에서 한국을 2군 격인

    'AI 안정적 경쟁국'으로 분류했다. 1군인 'AI 선도국'엔 미국·중국·영국·캐나다·싱가포르만 포함됐다.  그동안 정부는 지난 6월

    토터스미디어가 발표한 '글로벌 AI 순위'를 근거로 한국 AI 경쟁력이 세계 3위권이라고 강조해왔다.
    
    여기서 한국은 6위를 차지했는데 1,2위인 미국·중국을 제외하면 3위부터 8위까지 큰 차이가 없다는 설명이다. 그러나 이를 뒤집는

    보고서가 나오면서 한국 AI 경쟁력에 대한 위기감도 커진다.

    ** AI 쩐의전쟁, 각개전투로는 한계 **

    전문가들은 미국과 중국을 제외하면 현재 나라별 격차는 크지 않다고 입을 모은다. 다만 이들 국가가 천문학적 규모의 투자를 이어가는 만큼

    경쟁에서 뒤처지지 않으려면 국가 차원의 컨소시엄을 구성해 한국적 생태계를 만들어야 한다고 분석한다. 개별 기업의 각개전투론 승산이

    없다는 판단이다. 김명주 서울여대 정보보호학부 교수(AI안전연구소장)는 "AI는 자본력의 싸움"이라며 "그동안 우리나라는 기업이
    
    일대일로 대응해왔는데 한계가 분명한 만큼 국가에서 산학연을 연계한 원팀을 만들어야 한다"고 말했다.

    과감한 규제혁신도 요구된다. 이성엽 고려대 기술법정책센터장(한국데이터법정책학회장)은 "국내 한 IT기업에서 '규제가 많아 개발자들의

    자기검열이 강화됐다'고 하더라"라며 "AI 학습용 한국어 데이터가 부족한 가운데 개인정보와 공공데이터 활용도 제한적이다.

    다른 나라엔 없는 규제들"이라고 지적했다. 이어 "민간자본과 정부예산이 글로벌 빅테크를 따라잡기엔 부족한 만큼 데이터라도

    규제를 혁신해야 한다"고 강조했다.

    한국을 넘어 해외로 진출할 수 있는 범용 AI 모델을 개발해야 한다는 제안도 나왔다. 소프트웨어정책연구소는
    
    '글로벌 초거대 AI 모델 현황 분석' 보고서에서 "네이버(NAVER)의 하이퍼클로바X를 제외하고 대부분의 AI 모델은
    
    국내 서비스에 응용하거나 자사 제품 탑재를 위한 것"이라며 "AI 분야에서 우리나라의 글로벌 영향력을 확대하기 위해선
    
    범용성 있고 공개 가능한 AI 모델이 필수적"이라고 분석했다.
    """

for model_name in model_names:
    LangchainHelper.bind_model(model_name)
    model = LangchainHelper.load_model()

    chain = (
        prompt | model | StrOutputParser()
    )

    response = chain.invoke({"content": content})

    print(f"MODEL: {model.model_name}")
    print(f"요약 내용: {response}\n")
