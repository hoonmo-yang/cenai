from operator import attrgetter, itemgetter

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core.dataman import dedent, Struct
from cenai_core.grid import GridChainContext

from amc.pdac_classifier import PDACClassifier, PDACClassifyResult


class MultiqueryQuestionHandler(BaseCallbackHandler):
    def __init__(self):
        self._question_text = ""

    def on_chain_end(self,
                     response: str,
                     **kwargs
                     ) -> None:
        self._question_text = response.text.strip()

    def __call__(self) -> str:
        return self._question_text


class PDACClassifier8(PDACClassifier):
    def __init__(self,
                 metadata: Struct,
                 sections: list[str],
                 topk: int,
                 num_questions: int,
                 **parameter
                 ):
        super().__init__(
            metadata=metadata,
            module_suffix=f"k{topk:02d}_q{num_questions:02d}",
            sections=sections,
        )

        self.INFO(f"{self.header} prepared ....")

        self._topk = topk
        self._num_questions = num_questions

        self.metadata_df.loc[0, ["topk", "num_questions"]] = [
            self._topk, self._num_questions,
        ]

        self._multiquery_question_handler = MultiqueryQuestionHandler()

        retriever = self._create_retriever()
        multiquery_retriever = self._create_multiquery_retriever(retriever)

        self.classifier_chain = self._create_classifier_chain(multiquery_retriever)

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

        self.INFO(f"{self.header} RAG prepared DONE")
        return retriever

    def _create_multiquery_retriever(self,
                                     retriever: BaseRetriever
                                     ) -> Runnable:
        self.INFO(f"{self.header} MULTI-QUERY RETRIEVER prepared ....")

        prompt = """
        사용자가 제시한 원 질문을 바탕으로, 췌장암 환자의 CT 판독문 유형을 예측하는 데 유용한
        
        다양한 관점의 질문을 {num_questions}개 생성하십시오. 이러한 질문은 벡터 DB에서 관련 문서를
        
        더 효과적으로 검색하기 위한 목적으로 작성되어야 합니다.

        사용자의 질문에는 예측해야 할 CT 판독문 내용이 포함되어 있으며, 이 판독문은 {sections}으로 구성됩니다.

        {sections}의 정보를 참고하여, CT 판독문의 유형 예측에 도움이 될 질문을 다양한 관점에서 만들어 주십시오.

        각 질문은 개행 문자(\n)로 구분되어 한 줄씩 나열되도록 작성하십시오. 예시: foo\nbar\nbaz\n

        *췌장암 환자의 CT 판독문 유형*:
        {category_text}
        
        {question}
        """

        question_prompt = ChatPromptTemplate.from_messages(
            ("human", dedent(prompt)),
        )

        prompt = """
        {multiquery_question}

        {original_question}
        """

        mixup_prompt = PromptTemplate.from_template(dedent(prompt))

        multiquery_retriever = (
            {
                "multiquery_question":
                    question_prompt | self.model | StrOutputParser(),

                "original_question":
                    itemgetter("question")
            } |
            mixup_prompt.with_config(
                callbacks=[
                    self._multiquery_question_handler,
                ],
            ) |
            attrgetter("text") |
            retriever
        )

        self.INFO(f"{self.header} MULTI-QUERY RETRIEVER prepared DONE")
        return multiquery_retriever

    def _create_classifier_chain(self,
                                 retriever: BaseRetriever
                                 ) -> Runnable:
        self.INFO(f"{self.header} CHAIN prepared ....")

        system_prompt = """
        당신은 췌장암 환자의 CT 판독문이 특정 유형에 속하는지 예측하고,

        그 이유를 설명하는 역할을 맡고 있습니다. 주어진 유형 설명과
        
        검색된 문맥을 바탕으로, 입력된 CT 판독문의 유형을 정확히 예측하십시오.
        
        또한, 예측한 근거를 명확하게 제시하여 왜 해당 유형에 속하는지 설명해 주세요.

        *췌장암 환자의 CT 판독문 유형 설명*:
        {category_text}

        *검색된 문맥*:
        {context}
        """

        human_prompt = """
        {question}

        *참고 사항*:
        답변은 반드시 'PDACClassifyResult' 함수의 속성 형식에 맞게 출력하십시오.

        유형 결정 근거는 한국어로 작성해 주세요.

        전문 용어는 입력된 원문에서 사용된 형태를 그대로 유지해 주세요.

        *유형*:
        *근거*:
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", dedent(system_prompt)),
            ("human", dedent(human_prompt)),
        ])

        chain = (
            RunnablePassthrough().assign(
                context=RunnablePassthrough() |
                retriever |
                self.reorder_documents |
                self.concat_documents
            ) |
            prompt |
            self.model.with_structured_output(PDACClassifyResult)
        )

        self.INFO(f"{self.header} CHAIN prepared DONE")
        return chain

    def classify_pre(self) -> GridChainContext:
        return GridChainContext(
            parameter={
                "num_questions": self._num_questions,
            },
            handler={
                "생성질문": self._multiquery_question_handler,
            },
        )
