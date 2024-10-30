from __future__ import annotations
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core import cenai_path, INFO, LangchainHelper, load_dotenv
from cenai_core.dataman import load_json_yaml


class PaperSummarizer:
    data_dir = cenai_path("data") / "nrfk"
    output_dir = cenai_path("output") / "nrfk"

    def __init__(self,
                 model_name: str,
                 model: BaseChatModel,
                 embeddings: Embeddings,
                 paper_name: str
                 ):
        self._model = model
        self._embeddings = embeddings
        self._paper_name = paper_name
        self._paper_file = self.data_dir / f"{paper_name}.pdf"
        self._body = f"{model_name}_{paper_name}"

        (self._sections, self._keywords,
         self._summary_template) = self._load_templates()

        self._documents = []
        self._summary_df = pd.DataFrame()

    def _load_templates(self) -> tuple[list[str], list[str], dict[str, Any]]:
        yaml_file = self.data_dir / "template.yaml" 
        deserial = load_json_yaml(yaml_file)

        sections = []
        keywords = []

        for key, value in deserial["toc"].items():
            sections.append(key)
            keywords.append(value)

        return sections, keywords, deserial["summary"]

    def split_sections(self) -> None:
        loader = PyPDFLoader(str(self._paper_file))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        sources = splitter.split_documents(documents)

        targets = []
        section = ""
        for source in sources:
            target, section = self._classify_section(source, section)
            targets.append(target)

        self._documents = targets

    def _classify_section(self,
                          document: Document,
                          section: str
                          ) -> tuple[Document, str]:

        sentence = document.page_content
        results = process.extract(
            sentence,
            self._keywords,
            scorer=fuzz.partial_ratio,
        )
        if "sections" not in document.metadata:
            document.metadata["sections"] = []

        last = -1
        targets = []
        for result in results:
            _, score, k = result
            if score > 80.0:
                targets.append(self._sections[k])
                last = max(k, last)

        if last == -1:
           targets.append(section)
        else:
            section = self._sections[last]

        document.metadata["sections"] = targets

        return document, section

    def create_summary(self) -> None:
        self._summary_df = pd.DataFrame(
            columns=[
                "name",
                "title",
                "content",
            ],
        )

        for name, item in self._summary_template.items():
            title = item["title"]
            sections = item["sections"]

            if name == "keyword":
                content = self._extract_keywords(sections)
            else:
                content = self._summarize_sections(title, sections)

            new_df = pd.DataFrame({
                "name": [name],
                "title": [title],
                "content": [content],
            })

            self._summary_df = pd.concat(
                [self._summary_df, new_df], axis=0
            )

    def _summarize_sections(self,
                            title: str,
                            sections: list[str]
                            ) -> str:
        messages = [
            HumanMessage(content=document.page_content)
            for document in self._documents
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
            self.model |
            StrOutputParser()
        )

        summary = chain.invoke({
            "title": title,
            "sections": messages,
        })

        return summary

    def _extract_keywords(self, sections: list[str]) -> str:
        messages = [
            HumanMessage(content=document.page_content)
            for document in self._documents
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
            self.model |
            StrOutputParser()
        )

        keyword = chain.invoke({
            "sections": messages,
        })

        return keyword

    def create_report(self) -> None:
        excel_dir = self.output_dir / "excel"
        excel_dir.mkdir(parents=True, exist_ok=True)

        excel_file = excel_dir / f"{self._body}.xlsx"

        with pd.ExcelWriter(excel_file) as writer:
            self._summary_df.to_excel(
                writer, sheet_name="Summary"
            )

    @property
    def model(self) -> BaseChatModel:
        return self._model

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @classmethod
    def summarize_papers(cls,
                         model_names: list[str],
                         paper_names: list[str],
                         summarizer_class: PaperSummarizer
                         ) -> None:
        load_dotenv(False)

        for model_name in model_names:
            LangchainHelper.bind_model(model_name)
            model = LangchainHelper.load_model()
            embeddings = LangchainHelper.load_embeddings()

            for paper_name in paper_names:
                INFO(
                    f"\nmodel:{model_name} paper:{paper_name}"
                )

                summarizer = summarizer_class(
                    model_name=model_name,
                    model=model,
                    embeddings=embeddings,
                    paper_name=paper_name,
                )

                summarizer.split_sections()
                summarizer.create_summary()
                summarizer.create_report()


def main() -> None:
    model_names = ["gpt-4o"]
    model_names = ["gpt-3.5-turbo"]
    model_names = ["hyperclovax-hcx-003"]

    paper_names = [
        "nrfk-paper00",
        "nrfk-paper01",
    ]

    PaperSummarizer.summarize_papers(
        model_names=model_names,
        paper_names=paper_names,
        summarizer_class=PaperSummarizer,
    )


if __name__ == "__main__":
    main()