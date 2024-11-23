from typing import Sequence

from operator import itemgetter

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core.dataman import Struct
from cenai_core.langchain_helper import ChainContext, load_chatprompt

from amc.pdac_classifier import PDACClassifier, PDACClassifyResult


class ReorderedRagClassifier(PDACClassifier):
    def __init__(self,
                 model: str,
                 sections: Sequence[str],
                 topk: int,
                 classify_prompt: str,
                 question: str,
                 metadata: Struct
                 ):

        case_suffix = "_".join([
            classify_prompt.split(".")[0],
            question.split(".")[0],
            f"k{topk:02d}",
        ])

        super().__init__(
            model=model,
            sections=sections,
            case_suffix=case_suffix,
            metadata=metadata,
        )

        self.INFO(f"{self.header} prepared ....")

        self._topk = topk
        self.question = question

        self.metadata_df.loc[
            0,
            [
                "topk",
                "classify_prompt",
                "question",
            ]
        ] = [
            topk,
            classify_prompt,
            question,
        ]

        retriever = self._create_retriever()

        self.classifier_chain = self._create_classifier_chain(
            classify_prompt=classify_prompt,
            retriever=retriever,
        )

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

    def _create_classifier_chain(self,
                                 classify_prompt: str,
                                 retriever: BaseRetriever
                                 ) -> Runnable:
        self.INFO(f"{self.header} CHAIN prepared ....")

        prompt_args = load_chatprompt(self.content_dir / classify_prompt)
        prompt = ChatPromptTemplate(**prompt_args)

        chain = (
            RunnablePassthrough().assign(
                context=itemgetter("question") |
                retriever |
                self.reorder_documents |
                self.concat_documents
            ) |
            prompt |
            self.model.with_structured_output(PDACClassifyResult)
        ) 

        self.INFO(f"{self.header} CHAIN prepared DONE")
        return chain

    def classify_pre(self) -> ChainContext:
        return ChainContext()