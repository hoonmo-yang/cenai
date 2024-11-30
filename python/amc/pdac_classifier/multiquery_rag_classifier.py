from typing import Sequence

from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableMap
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core.dataman import Struct
from cenai_core.langchain_helper import ChainContext, load_chatprompt

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


class MultiQueryRagClassifier(PDACClassifier):
    def __init__(self,
                 models: Sequence[str],
                 sections: Sequence[str],
                 topk: int,
                 num_questions: int,
                 query_prompt: str,
                 classify_prompt: str,
                 question: str,
                 metadata: Struct
                 ):

        case_suffix = "_".join([
            query_prompt.split(".")[0],
            classify_prompt.split(".")[0],
            question.split(".")[0],
            f"k{topk:02d}",
            f"q{num_questions:02d}",
        ])

        super().__init__(
            models=models,
            sections=sections,
            case_suffix=case_suffix,
            metadata=metadata,
        )

        self.INFO(f"{self.header} prepared ....")

        self._topk = topk
        self._num_questions = num_questions
        self.question = question

        self.metadata_df.loc[
            0,
            [
                "topk",
                "num_questions",
                "query_prompt",
                "classify_prompt",
                "question",
             ]
        ] = [
            topk,
            num_questions,
            query_prompt,
            classify_prompt,
            question,
        ]

        self.multiquery_question_handler = MultiqueryQuestionHandler()

        retriever = self._build_retriever()
        multiquery_retriever = self._build_multiquery_retriever(
            query_prompt=query_prompt,
            retriever=retriever,
        )

        self.main_chain = self._build_classify_chain(
            classify_prompt=classify_prompt,
            retriever=multiquery_retriever,
            )

        self.INFO(f"{self.header} prepared DONE")

    def _build_retriever(self) -> BaseRetriever:
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

        vectorstore = FAISS.from_documents(
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

    def _build_multiquery_retriever(self,
                                    query_prompt: str,
                                    retriever: BaseRetriever
                                    ) -> Runnable:
        self.INFO(f"{self.header} MULTI-QUERY RETRIEVER prepared ....")

        prompt_args = load_chatprompt(self.content_dir / query_prompt)
        query_prompt = ChatPromptTemplate(**prompt_args)

        multiquery_retriever = (
            query_prompt |
            self.model[0] |
            StrOutputParser().with_config(
                callbacks=[self.multiquery_question_handler],
            ) |
            retriever
        )

        self.INFO(f"{self.header} MULTI-QUERY RETRIEVER prepared DONE")
        return multiquery_retriever

    def _build_classifier_chain(self,
                                classify_prompt: str,
                                retriever: BaseRetriever
                                ) -> Runnable:
        self.INFO(f"{self.header} CHAIN prepared ....")

        prompt_args = load_chatprompt(self.content_dir / classify_prompt)
        prompt = ChatPromptTemplate(**prompt_args)

        chain = (
            RunnableMap({
                "content": itemgetter("content"),

                "context": (
                    itemgetter("question") |
                    retriever |
                    self.reorder_documents |
                    self.concat_documents
                ),
            }) |
            prompt |
            self.model[0].with_structured_output(PDACClassifyResult)
        )

        self.INFO(f"{self.header} CHAIN prepared DONE")
        return chain

    def classify_pre(self) -> ChainContext:
        return ChainContext(
            parameter={
                "num_questions": self._num_questions,
            },
            handler={
                "생성질문": self.multiquery_question_handler,
            },
        )
