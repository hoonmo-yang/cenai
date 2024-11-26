from __future__ import annotations
from typing import Optional

from pathlib import Path
import pandas as pd
import re

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core import Timer

from cenai_core.dataman import (
    divide_evenly, proportionalize, Q, Struct
)

from cenai_core.langchain_helper import get_document_length, load_documents
from cenai_core.grid import GridRunnable


class QADatasetGenerator(GridRunnable):
    logger_name = "cenai.nrf.qa_dataset_generator"

    P_RESPONSE = re.compile(
        r"문제\s*:(.*?)\s정답\s*:(.*?)(?=\s*문제\s*:|$)", re.S
    )

    def __init__(self,
                 model,
                 chunk_size: int,
                 chunk_overlap: int,
                 num_datasets: int,
                 max_tokens: int,
                 case_suffix: str,
                 metadata: Struct
                 ):

        case_suffix = "_".join([
            case_suffix,
            f"cs{chunk_size}",
            f"co{chunk_overlap}",
            f"n{num_datasets}",
            f"tk{max_tokens}",
        ])

        corpus_part = "_".join([
            metadata.corpus_prefix,
            "-".join(
                [stem for stem in metadata.corpus_stem if stem]
            ),
            "-".join([
                extension[1:] for extension in metadata.corpus_ext
                if extension]),
        ])

        super().__init__(
            model=model,
            case_suffix=case_suffix,
            corpus_part=corpus_part,
            metadata=metadata,
        )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._num_datasets = num_datasets
        self._max_tokens = max_tokens

        self.metadata_df.loc[
            0, 
            [
                "chunk_size",
                "chunk_overlap",
                "num_datasets",
                "max_tokens",
            ]
        ] = [
            chunk_size,
            chunk_overlap,
            num_datasets,
            max_tokens,
        ]

        self._generate_chain = RunnableLambda(
            lambda _: ""
        )

    def run(self, **directive) -> None:
        self._generate_qa_dataset(**directive)

    def _generate_qa_dataset(
            self,
            num_tries: Optional[int] = None,
            recovery_time: Optional[int] = None,
            **kwargs
            ) -> None:

        self.INFO(f"{self.header} QA-DATASET GENERATE proceed ....")

        weights = [
            get_document_length(file_)
            for file_ in self.document_files
        ]

        sizes = proportionalize(self._num_datasets, weights)
        
        timer = Timer()

        for i, (file_, size) in enumerate(zip(self.document_files, sizes)):
            self.INFO(
                f"{self.header} FILE {Q(file_)} "
                f"[{i + 1}/{len(self.document_files)}] ...."
            )

            documents = self._load_documents(file_)

            total = len(documents)
            split_sizes = divide_evenly(size, total)

            result_df = pd.DataFrame()

            for j, (document, split_size) in enumerate(zip(documents, split_sizes)):
                if split_size == 0:
                    self.INFO(
                        f"{self.header} DOCUMENT [{j + 1}/{total}] "
                        f"size: {split_size} SKIP"
                    )
                    continue

                some_result_df = self._generate_qa_dataset_foreach(
                    document=document,
                    size=split_size,
                    num_tries=num_tries,
                    recovery_time=recovery_time,
                    file_=file_,
                )

                if some_result_df is None:
                    self.INFO(
                        f"{self.header} DOCUMENT [{j + 1}/{total}] "
                        f"size: {split_size} SKIP"
                    )
                    continue

                self.INFO(
                    f"{self.header} DOCUMENT [{j + 1}/{total}] "
                    f"size: {split_size} DONE"
                )

                result_df = pd.concat(
                    [result_df, some_result_df], axis=0
                )

            self.result_df = pd.concat(
                [self.result_df, result_df], axis=0
            )

            timer.lap()

            self.INFO(
                f"{self.header} "
                f"total_time: {timer.seconds:.1f}s "
                f"({self.result_df.shape[0]}/{self._num_datasets})"
            )

            self.INFO(
                f"{self.header} FILE {Q(file_)} "
                f"[{i + 1}/{len(self.document_files)}] DONE"
            )

        self.result_df = self.result_df.reset_index(drop=True)
        self.INFO(f"{self.header} QA-DATASET GENERATE proceed DONE")

    def _load_documents(self, file_: Path) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        documents = load_documents(file_)

        split_documents = splitter.split_documents(
            documents=documents,
        )

        return split_documents

    def _generate_qa_dataset_foreach(
            self,
            document: Document,
            size: int,
            num_tries: int,
            recovery_time: int,
            file_: Path
        ) -> Optional[pd.DataFrame]:
        for i in range(num_tries):
            try:
                timer = Timer()

                response = self.generate_chain.invoke(
                    {
                        "num_datasets": size,
                        "max_tokens": self._max_tokens,
                        "context": document.page_content,
                    }
                )

            except KeyboardInterrupt as error:
                raise error

            except BaseException:
                self.ERROR(f"LLM({self.model.model_name}) internal error")
                self.ERROR(f"number of tries {i + 1}/{num_tries}")

                Timer.delay(recovery_time)
            else:
                break
        else:
            self.ERROR(f"number of tries exceeds {num_tries}")
            return None

        timer.lap()

        records = self._parse_response(response, size)
        result_df = pd.DataFrame(records)

        result_df[["file", "time"]] = [str(file_), timer.seconds]

        return result_df

    def _parse_response(self,
                        response: str,
                        size: int
                        ) -> list[dict[str, str]]:
        if not response:
            return []

        matches = self.P_RESPONSE.findall(response)

        records = []
        for match in matches:
            question = match[0].strip()
            answer = match[1].strip()
            records.append({"문제": question, "정답": answer})

        if len(records) != size:
            self.WARNING(
                f"numbers of generated records: {Q(len(records))} "
                f"not matched to {Q(size)}"
            )

        return records

    @property
    def generate_chain(self) -> Runnable:
        return self._generate_chain

    @generate_chain.setter
    def generate_chain(self, chain: Runnable) -> None:
        self._generate_chain = chain
