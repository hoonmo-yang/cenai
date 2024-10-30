from typing import Any

import dotenv
from rapidfuzz import fuzz, process
import yaml

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hyperclovax import HyperCLOVAXChatModel, HyperCLOVAXSummarizer


def annotate_sections(
        documents: list[Document],
        keywords: list[str],
        sections: list[str],
        threshold: float
    ) -> list[Document]:
    targets = []
    section = ""
    for document in documents:
        target, section = annotate_section(
            document=document,
            section=section,
            sections=sections,
            keywords=keywords,
            threshold=threshold,
        )
        targets.append(target)

    return targets


def annotate_section(
        document: Document,
        section: str,
        sections: list[str],
        keywords: list[str],
        threshold: float
    ) -> tuple[Document, str]:
    sentence = document.page_content
    results = process.extract(
        sentence,
        keywords,
        scorer=fuzz.partial_ratio
    )
    if "sections" not in document.metadata:
        document.metadata["sections"] = []

    last = -1
    targets = []
    for result in results:
        _, score, k = result
        if score > threshold:
            targets.append(sections[k])
            last = max(k, last)

    if last == -1:
        targets.append(section)
    else:
        section = sections[last]

    document.metadata["sections"] = targets

    return document, section


def create_summary(
        documents: list[Document],
        summary_template: dict[str, Any],
        model: BaseChatModel
    ) -> tuple[str, str, str]:
    summary = {}
    for name, item in summary_template.items():
        title = item["title"]
        sections = item["sections"]

        if name == "keyword":
            content = extract_keywords(
                model=model,
                documents=documents,
                sections=sections,
            )
        else:
            content = summarize_sections(
                model=model,
                documents=documents,
                title=title,
                sections=sections,
            )

        summary[name] = {
            "title": title,
            "content": content,
        }

    return summary


def summarize_sections(
        model: BaseChatModel,
        documents: list[Document],
        title: str,
        sections: list[str]
) -> str:
    messages = [
        HumanMessage(content=document.page_content)
        for document in documents
        if set(document.metadata["sections"]) & set(sections)
    ]

    system_prompt = """
        당신은 학술논문을 요약하는 AI 전문가입니다.
        요청한 내용의 요약 외에 아무 것도 대답하지 마십시오.
        대화체나 일상어가 아닌 격식을 갖춘 요약문으로 대답해야 합니다.
        한국어를 사용해야 합니다. 전문 용어는 가급적 원문을 따르기 바랍니다.

        제시된 내용을 {title}에 알맞게 요약해야 합니다.
        요약한 내용은 1000 토큰이 넘지 않도록 해 주십시오.
        """

    human_prompt = ""

    if model.model_name != "hyperclovax-summarizer":
        human_prompt = """
            아래 입력된 요약할 내용을 요약해 주세요.

            ### 요약할 내용:
        """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
        MessagesPlaceholder(variable_name="sections"),
    ])

    chain = (
        prompt |
        model |
        StrOutputParser()
    )

    summary = chain.invoke({
        "title": title,
        "sections": messages,
    })

    return summary


def extract_keywords(
        model: BaseChatModel,
        documents: list[Document],
        sections: list[str]
) -> str:
    if model.model_name == "hyperclovax-summarizer":
        return "LLM doesn't support keyword extractions"

    messages = [
        HumanMessage(content=document.page_content)
        for document in documents
        if set(document.metadata["sections"]) & set(sections)
    ]

    system_prompt = """
    당신은 학술논문에서 중심어를 추출하는 AI 전문가입니다.
    당신은 추출한 중심어만 답변해야 합니다.
    중심어는 원문 내용에 중요한 주제이거나 자주 등장하는 전문 어휘입니다.
    추출한 중심어는 한국어와 영어로 분류해서 답변해 주시기 바랍니다.
    추출한 중심어는 각 언어 별로 5개가 넘지 않도록 해 주십시오.
    """

    human_prompt = """
    아래 입력된 내용에 대한 중심어를 추출해 주십시오.

    ### 내용:
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
        MessagesPlaceholder(variable_name="sections"),
    ])

    chain = (
        prompt |
        model |
        StrOutputParser()
    )

    keyword = chain.invoke({
        "sections": messages,
    })

    return keyword


model_name = "gpt-3.5-turbo"
model_name = "gpt-4o"
model_name = "hyperclovax-hcx-003"
model_name = "hyperclovax-summarizer"

dotenv.load_dotenv()

if "gpt" in model_name:
    model = ChatOpenAI(model=model_name)

elif model_name == "hyperclovax-hcx-003":
    model = HyperCLOVAXChatModel()

elif model_name == "hyperclovax-summarizer":
    model = HyperCLOVAXSummarizer()

with open("template.yaml") as fin:
    deserial = yaml.load(
        fin, Loader=yaml.Loader,
    )

sections = []
keywords = []

for key, value in deserial["toc"].items():
    sections.append(key)
    keywords.append(value)

summary_template = deserial["summary"]
threshold = 80.0

loader = PyPDFLoader("paper.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

sources = splitter.split_documents(
    documents=documents,
)

documents = annotate_sections(
    documents=sources,
    sections=sections,
    keywords=keywords,
    threshold=threshold,
)

summary = create_summary(
    documents=documents,
    summary_template=summary_template,
    model=model,
)

for name, item in summary.items():
    print(f"**{item['title']}**:\n")
    print(f"{item['content']}\n")
    print(f"**CATEGORY**: \"{name}\"\n\n")
