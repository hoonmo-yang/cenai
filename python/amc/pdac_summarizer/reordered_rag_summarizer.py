from operator import attrgetter, itemgetter

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from langchain_core.runnables import (
    Runnable, RunnableBranch, RunnableLambda, RunnableMap, RunnablePassthrough
)

from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core.dataman import Struct
from cenai_core.langchain_helper import load_chatprompt, load_prompt

from amc.pdac_summarizer import (
    PDACSummarizer, PDACClassifyResult, PDACReportTemplateFail
)

from amc.pdac_summarizer import pdac_template


class ReorderedRagSummarizer(PDACSummarizer):
    def __init__(self,
                 model: str,
                 topk: int,
                 classify_prompt: str,
                 summarize_prompt: str,
                 question: str,
                 metadata: Struct
                 ):

        case_suffix = "_".join({
            classify_prompt.split(".")[0],
            summarize_prompt.split(".")[0],
            question.split(".")[0],
            f"k{topk:02d}",
        })

        super().__init__(
            model=model,
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
                "summarize_prompt",
                "question",
            ]
        ] = [
            topk,
            classify_prompt,
            summarize_prompt,
            question,
        ]

        retriever = self._build_retriever()

        classify_chain = self._build_classify_chain(
            classify_prompt=classify_prompt,
            retriever=retriever,
        )

        summarize_chain = self._build_summarize_chain(
            summarize_prompt=summarize_prompt,
        )

        self.summarize_chain = classify_chain | summarize_chain
        
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

    def _build_classify_chain(self,
                              classify_prompt: str,
                              retriever: BaseRetriever
                              ) -> Runnable:
        self.INFO(f"{self.header} CLASSIFY CHAIN prepared ....")

        prompt_args = load_chatprompt(self.content_dir / classify_prompt)
        prompt = ChatPromptTemplate(**prompt_args)

        chain = RunnableMap({
            "type": (
                RunnablePassthrough() |
                {
                    "content":  itemgetter("content"),

                    "context": (
                        itemgetter("question") |
                        retriever |
                        self.reorder_documents |
                        self.concat_documents
                    ),

                } |
                prompt |
                self.model.with_structured_output(PDACClassifyResult) |
                attrgetter("type")
            ),

            "content": itemgetter("content"),
        })

        self.INFO(f"{self.header} CLASSIFY CHAIN prepared DONE")
        return chain

    def _build_summarize_chain(self,
                               summarize_prompt: str
                               ) -> Runnable:
        self.INFO(f"{self.header} SUMMARIZE CHAIN prepared ....")

        prompt_args, partials = load_prompt(self.content_dir / summarize_prompt)

        labels = sorted(self.get_type_labels())[:-1]

        statements = []

        for i, label in enumerate(labels):
            Class = getattr(pdac_template, f"PDACReportTemplate{i + 1}")

            parser = PydanticOutputParser(
                pydantic_object=Class,
            )

            full_args = prompt_args | {
                "partial_variables": {
                    partials[0]: parser.get_format_instructions(),
                },
            }

            prompt = PromptTemplate(**full_args)

            statement = (
                lambda x, type_=label: x["type"] == type_,
                prompt | self.model | parser
            )

            statements.append(statement)

        statements.append(
            RunnableLambda(lambda _: PDACReportTemplateFail(
                message="PDAC Report Classify Failure",
            ))
        )

        chain = RunnableBranch(*statements)

        self.INFO(f"{self.header} SUMMARIZE CHAIN prepared DONE")
        return chain
