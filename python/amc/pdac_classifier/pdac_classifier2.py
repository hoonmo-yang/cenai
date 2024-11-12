from operator import itemgetter
import pandas as pd

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core.dataman import dedent, Q, Struct
from cenai_core.grid import GridChainContext

from amc.pdac_classifier import PDACClassifier, PDACClassifyResult


class PDACClassifier2(PDACClassifier):
    def __init__(self,
                 metadata: Struct,
                 topk: int,
                 sections: list[str],
                 **parameter
                 ):
        super().__init__(
            metadata=metadata,
            module_suffix=f"k{topk:02d}",
            sections=sections,
        )

        self.INFO(f"{self.header} prepared ....")

        self._topk = topk
        self._metadata_df.loc[0, "topk"] = self._topk

        retriever = self._create_retriever()
        self.classifier_chain = self._create_classifier_chain(retriever)

        self.INFO(f"{self.header} prepared DONE")

    def _create_retriever(self) -> BaseRetriever:
        self.INFO(f"{self.header} RAG prepared ....")

        trainset_text = self.stringfy_trainsets()
        documents = [Document(page_content=trainset_text)]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        splits = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self._topk},
        )

        self.INFO(f"{self.header} RAG prepared DONE")
        return retriever

    def _create_classifier_chain(self,
                                 retriever: BaseRetriever
                                 ) -> Runnable:
        self.INFO(f"{self.header} CHAIN prepared ....")

        system_prompt = """
        당신은 췌장암 환자의 CT 판독문이 어떤 유형에 속하는지 예측하고,
        그렇게 예측한 이유를 설명하는 역할을 맡고 있습니다.
        당신은 유형의 명칭과 검색된 문맥을 이용하여 입력된 CT 판독문의 유형을
        맞추세요. 왜 그런지 유형에 대한 근거를 명확하게 제시해 주세요.
        
        *유형의 명칭*:
        {category_text}

        *문맥*:
        {context}
        """

        human_prompt = """
        {question}

        다음 사항을 참고해 주세요:
        1. 답변은 'PDAClassifierResult' 함수의 속성대로 출력되어야 합니다.
        2. 유형 결정 근거는 한국어를 사용하세요.
        3. 전문 용어는 입력에서 사용한 원문을 그대로 유지하세요.

        *유형*:
        *근거*:
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", dedent(system_prompt)),
            ("human", dedent(human_prompt)),
        ])

        chain = (
            RunnablePassthrough().assign(
                context=itemgetter("question") |
                retriever |
                self.concat_documents
            ) |
            prompt |
            self.model.with_structured_output(PDACClassifyResult)
        )

        self.INFO(f"{self.header} CHAIN prepared DONE")
        return chain

    def classify_pre(self) -> GridChainContext:
        return GridChainContext()
