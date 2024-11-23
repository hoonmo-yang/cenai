from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product
import pandas as pd
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from cenai_core.dataman import load_json_yaml, Q, Struct
from cenai_core.langchain_helper import LangchainHelper
from cenai_core.logger import Logger
from cenai_core.system import cenai_path


class GridRunnable(Logger, ABC):
    logger_name = "cenai.grid_runnable"

    log_dir = cenai_path("log")
    data_dir = cenai_path("data")

    def __init__(self,
                 model: str,
                 case_suffix: str,
                 corpus_suffix: str,
                 metadata: Struct
                 ):
        self._metadata = metadata

        self._suite_id = self.metadata.suite.id
        self._suite_prefix = self.metadata.suite.prefix

        log_file = Path(
            self.log_dir /
            self.suite_prefix /
            f"{self.suite_id}.log"
        )

        super().__init__(
            log_file=log_file,
        )

        self._case_id = "_".join([
            token for token in [
                corpus_suffix,
                model,
                self.metadata.module,
                case_suffix,
            ] if token
        ])

        prefix_dir = self.metadata.suite.prefix_dir
        self._corpus_dir = prefix_dir / "corpus"

        self._dataset_df = self._load_dataset()
        self._document_files = self._collect_document_files()

        self._source_dir = (
            self.data_dir /
            self.metadata.institution /
            self.metadata.task 
        )

        self._source_corpus_dir = self._source_dir / "corpus"
        self._content_dir = self._source_dir / "content"
        self._html_dir = self._source_dir / "html"

        LangchainHelper.bind_model(model)

        self._model = LangchainHelper.load_model()
        self._embeddings = LangchainHelper.load_embeddings()

        self._metadata_df = pd.DataFrame({
            "suite_id": [self.suite_id],
            "case_id": [self.case_id],
            "suite_prefix": [self.suite_prefix],
            "suite_create_date": [self.metadata.suite.create_date],
            "suite_index": [self.metadata.suite.index],
            "institution": [self.metadata.institution],
            "task": [self.metadata.task],
            "tags": [",".join(self.metadata.tags)],
            "model": [self.model.model_name],
            "module": [self.metadata.module],
            "corpus_mode": [self.metadata.corpus_mode],
            "corpus_prefix": [self.metadata.corpus_prefix],
            "corpus_stem": [self.metadata.corpus_stem],
            "corpus_ext": [self.metadata.corpus_ext],
            "profile_file": [self.metadata.suite.profile_file],
        })

        self._result_df = pd.DataFrame()

    def _load_dataset(self) -> dict[str, pd.DataFrame]:
        if self.metadata.corpus_mode not in ["dataset",]:
            return {}

        dataset_df = {}

        corpus_stem = self.metadata.corpus_stem
        corpus_ext = self.metadata.corpus_ext

        for tag in ["train", "test"]:
            corpus_dir = self.corpus_dir / Path(corpus_stem).parent / tag
            file_ = corpus_dir / f"{Path(corpus_stem).name}{corpus_ext}"
            records = load_json_yaml(file_)
            dataset_df[tag] = pd.DataFrame(records)

        return dataset_df

    def _collect_document_files(self) -> list[Path]:
        if self.metadata.corpus_mode not in ["aggregate", "document",]:
            return []

        corpus_prefix = self.metadata.corpus_prefix
        corpus_dir = self.corpus_dir / corpus_prefix

        corpus_stem = self.metadata.corpus_stem
        corpus_ext = self.metadata.corpus_ext

        if isinstance(corpus_stem, str):
            corpus_stem = [corpus_stem]

        if isinstance(corpus_ext, str):
            corpus_ext = [corpus_ext]

        document_files = []

        for stem, extension in product(corpus_stem, corpus_ext):
            document_files.extend(
                list(corpus_dir.glob(f"{stem}{extension}"))
            )

        return document_files

    @abstractmethod
    def run(self, **directive) -> None:
        pass

    @property
    def corpus_dir(self) -> Path:
        return self._corpus_dir

    @property
    def source_dir(self) -> Path:
        return self._source_dir

    @property
    def content_dir(self) -> Path:
        return self._content_dir

    @property
    def source_corpus_dir(self) -> Path:
        return self._source_corpus_dir

    @property
    def html_dir(self) -> Path:
        return self._html_dir

    @property
    def model(self) -> BaseChatModel:
        return self._model

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @property
    def suite_id(self) -> str:
        return self._suite_id

    @property
    def suite_prefix(self) -> str:
        return self._suite_prefix

    @property
    def case_id(self) -> str:
        return self._case_id

    @property
    def header(self) -> str:
        return f"CASE {Q(self.case_id)}[{Q(self.suite_id)}]"

    @property
    def metadata(self) -> Struct:
        return self._metadata

    @property
    def metadata_df(self) -> pd.DataFrame:
        return self._metadata_df

    @property
    def dataset_df(self) -> pd.DataFrame:
        return self._dataset_df

    @property
    def document_files(self) -> list[Path]:
        return self._document_files

    @property
    def result_df(self) -> pd.DataFrame:
        return self._result_df

    @result_df.setter
    def result_df(self, value: pd.DataFrame) -> None:
        self._result_df = value
