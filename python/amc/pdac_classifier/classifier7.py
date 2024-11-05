from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core.dataman import dedent, Q

from amc.pdac_classifier import PDACClassifier, PDACClassifyResult


class MultiqueryQuestionHandler(BaseCallbackHandler):
    def __init__(self):
        self._questions = []

    def on_chain_end(self,
                     response: str,
                     **kwargs
                     ) -> None:
        questions = [
            question.strip() for question in response.split("\n")
        ]

        self._questions = [
            question for question in questions if question
        ]

    def __call__(self) -> str:
        return "\n".join(self._questions)


class Classifier7(PDACClassifier):
    def __init__(self,
                 dataset: str,
                 model_name: str,
                 algorithm: str,
                 sections: list[str],
                 topk: int,
                 num_questions: int,
                 **kwargs
                 ):
        super().__init__(
            dataset=dataset,
            model_name=model_name,
            algorithm=algorithm,
            sections=sections,
            hparam=f"k{topk:02d}_q{num_questions:02d}",
            **kwargs
        )

        self.INFO(f"RUN {Q(self.run_id)} prepared ....")

        self._topk = topk
        self._num_questions = num_questions

        self._evaluation_df.loc[0, ["topk", "num_questions"]] = [
            self._topk, self._num_questions
        ]

        self.multiquery_question_handler = MultiqueryQuestionHandler()

        retriever = self._create_retriever()
        multiquery_retriever = self._create_multiquery_retriever(retriever)

        self._classifier_chain = self._create_classifier_chain(multiquery_retriever)

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

    def _create_multiquery_retriever(self,
                                     retriever: BaseRetriever
                                     ) -> Runnable:
        self.INFO(f"RUN {Q(self.run_id)} MULTI-QUERY RETRIEVER prepared ....")

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

        multiquery_retriever = (
            question_prompt |
            self.model |
            StrOutputParser().with_config(
                callbacks=[self.multiquery_question_handler],
            ) |
            retriever
        )

        self.INFO(f"RUN {Q(self.run_id)} MULTI-QUERY RAG prepared DONE")
        return multiquery_retriever

    def _create_classifier_chain(self,
                                 retriever: BaseRetriever
                                 ) -> Runnable:
        self.INFO(f"RUN {Q(self.run_id)} CHAIN prepared ....")

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

        self.INFO(f"RUN {Q(self.run_id)} CHAIN prepared DONE")
        return chain

    def classify(self) -> None:
        self.INFO(f"RUN {Q(self.run_id)} CLASSIFY proceed ....")

        category_labels = self.get_category_labels()
        category_text = self.stringfy_categories()

        example_df = self._example_df["test"]

        hparam = {
            "num_questions": self._num_questions,
        }

        self._result_df = example_df.apply(
            self.classify_foreach,
            chain=self._classifier_chain,
            category_text=category_text,
            category_labels=category_labels,
            sections=self.sections,
            run_id=self.run_id,
            hparam=hparam,
            others={
                "생성질문": self.multiquery_question_handler,
            },
            total=example_df.shape[0],
            axis=1,
        )

        self.INFO(f"RUN {Q(self.run_id)} CLASSIFY proceed DONE")
