from operator import itemgetter

from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core.dataman import dedent, Q

from amc.pdac_classifier import PDACClassifier, PDACClassifyResult


class Classifier4(PDACClassifier):
    def __init__(self,
                 dataset: str,
                 model_name: str,
                 algorithm: str,
                 sections: list[str],
                 topk: int,
                 **kwargs
                 ):
        super().__init__(
            dataset=dataset,
            model_name=model_name,
            algorithm=algorithm,
            sections=sections,
            hparam=f"k{topk:02d}",
            **kwargs
        )

        self.INFO(f"RUN {Q(self.run_id)} prepared ....")

        self._topk = topk
        self._evaluation_df.loc[0, "topk"] = self._topk

        retriever = self._create_retriever()
        self._classifier_chain = self._create_classifier_chain(retriever)

        self.INFO(f"RUN {Q(self.run_id)} prepared DONE")

    def _create_retriever(self) -> BaseRetriever:
        self.INFO(f"RUN {Q(self.run_id)} RAG prepared ....")

        example_text = self.stringfy_examples()
        documents = [Document(page_content=example_text)]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        splits = splitter.split_documents(documents)

        bm25_retriever = BM25Retriever.from_documents(
            documents=splits,
        )
        bm25_retriever.k = 1

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
        )

        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self._topk},
        )

        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.7, 0.3],
        )

        self.INFO(f"RUN {Q(self.run_id)} RAG prepared DONE")
        return retriever

    def _create_classifier_chain(self,
                                 retriever: BaseRetriever
                                 ) -> Runnable:
        self.INFO(f"RUN {Q(self.run_id)} CHAIN prepared ....")

        system_prompt = """
        당신은 췌장암 환자의 CT 판독문이 어떤 유형에 속하는지 예측하고,
        그렇게 예측한 이유를 설명하는 역할을 맡고 있습니다.
        당신은 유형의 설명과 검색된 문맥을 이용하여 입력된 CT 판독문의 유형을
        맞추세요. 왜 그런지 유형에 대한 근거를 명확하게 제시해 주세요.

        *유형의 설명*:
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

        self.INFO(f"RUN {Q(self.run_id)} CHAIN prepared DONE")
        return chain

    def classify(self) -> None:
        self.INFO(f"RUN {Q(self.run_id)} CLASSIFY proceed ....")

        category_labels = self.get_category_labels()
        category_text = self.stringfy_categories()

        example_df = self._example_df["test"].head(1)

        self._result_df = example_df.apply(
            self.classify_foreach,
            chain=self._classifier_chain,
            category_text=category_text,
            category_labels=category_labels,
            sections=self.sections,
            run_id=self.run_id,
            total=example_df.shape[0],
            axis=1,
        )

        self.INFO(f"RUN {Q(self.run_id)} CLASSIFY proceed DONE")
