from typing import Sequence

from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableMap
from langchain.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core.dataman import Struct
from cenai_core.langchain_helper import ChainContext, load_prompt

from pdac_classifier import PDACClassifier
from pdac_template import PDACResultClassify


class VanilaRagClassifier(PDACClassifier):
    def __init__(self,
                 models: Sequence[str],
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
            models=models,
            sections=sections,
            case_suffix=case_suffix,
            metadata=metadata,
        )

        self.INFO(f"{self.header} prepared ....")

        self._topk = topk
        self._question = question

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

        retriever = self._build_retriever()

        self.main_chain = self._build_main_chain(
            classify_prompt=classify_prompt,
            retriever=retriever,
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

        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings[0],
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self._topk},
        )

        self.INFO(f"{self.header} RAG prepared DONE")
        return retriever

    def _build_main_chain(self,
                          classify_prompt: str,
                          retriever: BaseRetriever
                          ) -> Runnable:
        self.INFO(f"{self.header} MAIN CHAIN prepared ....")

        parser = PydanticOutputParser(
            pydantic_object=PDACResultClassify,
        )

        prompt_args, partials = load_prompt(self.content_dir / classify_prompt)

        full_args = prompt_args | {
            "partial_variables": {
                partials[0]: parser.get_format_instructions(),
            },
        }

        prompt = PromptTemplate(**full_args)

        chain = (
            RunnableMap({
                "content": itemgetter("content"),

                "context": (
                    itemgetter("question") |
                    retriever |
                    self.concat_documents
                ),
            }) |
            prompt |
            self.model[0] |
            parser
        )

        self.INFO(f"{self.header} MAIN CHAIN prepared DONE")
        return chain

    def classify_pre(self) -> ChainContext:
        return ChainContext()
