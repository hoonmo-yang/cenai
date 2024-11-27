from __future__ import annotations
from typing import Any, Optional

from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process

from langchain_core.runnables import Runnable, RunnableLambda
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core import Timer
from cenai_core.dataman import load_json_yaml, Q, Struct
from cenai_core.grid import GridRunnable
from cenai_core.langchain_helper import load_documents


class PaperSummarizer(GridRunnable, ABC):
    logger_name = "cenai.nrf.paper_classifier"

    def __init__(self,
                 model,
                 chunk_size: int,
                 chunk_overlap: int,
                 num_keywords: int,
                 max_tokens: int,
                 case_suffix: str,
                 metadata: Struct
                 ):
        
        case_suffix = "_".join([
            case_suffix,
            f"kw{num_keywords}",
            f"cs{chunk_size}",
            f"co{chunk_overlap}",
            f"tk{max_tokens}",
        ])

        corpus_parts = "_".join([
            metadata.corpus_prefix,
            metadata.corpus_stem,
            metadata.corpus_ext.split(".")[-1],
        ])

        super().__init__(
            model=model,
            case_suffix=case_suffix,
            corpus_parts=corpus_parts,
            metadata=metadata,
        )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._num_keywords = num_keywords
        self._max_tokens = max_tokens

        self.metadata_df.loc[
            0,
            [
                "chunk_size",
                "chunk_overlap",
                "num_keywords",
                "max_tokens",
             ]
        ] = [
            chunk_size,
            chunk_overlap,
            num_keywords,
            max_tokens,
        ]

        self._layout = self._get_layout()

        self._summarize_chain = RunnableLambda(
            lambda _: ""
        )

        self._keyword_chain = RunnableLambda(
            lambda _: ""
        )

    def _get_layout(self) -> Struct:
        layout_file = self.content_dir / "paper-layout.yaml"
        layout = load_json_yaml(layout_file)

        return Struct({
            "source": layout["source_template"],
            "summary": layout["summary_template"],
        })

    def run(self, **directive) -> None:
        self.summarize_paper(**directive)

    def summarize_paper(
            self,
            num_tries: Optional[int] = None,
            recovery_time: Optional[int] = None,
            **kwargs
            ) -> None:

        self.INFO(f"{self.header} PAPER SUMMARY proceed ....")

        for file_






        file_ = self.document_files[0]

        self.INFO(f"{self.header} FILE {Q(file_)} proceed ....")

        documents = self._prepare_documents(file_)

        for key, value in self.layout.summary.items():
            title = value["title"]
            sections = value["sections"]

            content = (
                self._extract_keywords(
                    documents, sections, num_tries, recovery_time
                ) if key in ["keyword",] else

                self._summarize_sections(
                    documents, title, sections, num_tries, recovery_time
                )
            )

    def _prepare_documents(self, file_: Path) -> list[Document]:
        documents = load_documents(file_)

        content = "\n".join([
            document.page_content for document in documents
        ])

        documents = [
            Document(
                page_content=content,
                metadata={"source": str(file_),},
            )
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
        )

        documents = splitter.split_documents(documents)
        documents = self._annotate_sections(documents)

    def _annotate_sections(self, documents: list[Document]) -> list[Document]:
        targets = []
        section = ""

        for document in documents:
            target, section = self._annotate_section(document, section)
            targets.append(target)

        return targets

    def _annotate_section(self,
                          document: Document,
                          section: str
                          ) -> tuple[Document, str]:
        content = document.page_content
        results = process.extract(
            content, self.layout.titles,
            scorer=fuzz.partial_ratio,
        )
        if "sections" not in document.metadata:
            document.metadata["sections"] = []

        last = -1
        targets = []
        for result in results:
            _, score, k = result
            if score > 80.0:
                targets.append(self.layout.sections[k])
                last = max(k, last)

        if last == -1:
            targets.append(section)
        else:
            section = self.layout.sections[last]

        document.metadata["sections"] = targets

        return document, section

    def _summarize_sections(
            self,
            documents: list[Document],
            title: str,
            sections: list[str],
            num_tries: int,
            recovery_time: int
            ):

        content = "\n".join([
            document.page_content for document in documents
            if set(document.metadata["sections"]) & set(sections)
        ])











    @property
    def layout(self) -> Struct:
        return self._layout
