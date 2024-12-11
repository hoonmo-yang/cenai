from __future__ import annotations
from typing import Any, Callable, Iterator, Optional, Sequence

from abc import ABC, abstractmethod
import itertools
import pandas as pd
from pydantic import BaseModel, Field

from langchain_community.document_transformers import LongContextReorder
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from cenai_core import Timer
from cenai_core.dataman import load_text, optional, Q, Struct
from cenai_core.nlp import match_text
from cenai_core.grid import GridRunnable
from cenai_core.langchain_helper import ChainContext

from amc.pdac_classifier import PDACResultClassify


class PDACClassifier(GridRunnable, ABC):
    logger_name = "cenai.amc.pdac_classifier"

    def __init__(self,
                 models: Sequence[str],
                 sections: Sequence[str],
                 case_suffix: str,
                 metadata: Struct
                 ):

        self._sections = self._get_sections(sections)

        corpus_part = "_".join([
            self.metadata.corpus_stem,
            "".join([
                "b" if section == "본문" else "c"
                for section in self._sections
            ]),
        ])

        super().__init__(
            models=models,
            case_suffix=case_suffix,
            corpus_part=corpus_part,
            metadata=metadata,
        )

        self.metadata_df.loc[0, "sections"] = ",".join(self._sections)

        self.main_chain = RunnableLambda(
            lambda _: PDACResultClassify(
                type="Not available",
                reason="PDACClassifier not implemented",
            )
        )

    @staticmethod
    def _get_sections(sections: list[str]) -> list[str]:
        sections = [
            section for section in sorted(
                sections, reverse=True
            ) if section in ["본문", "결론"]
        ]
        return sections if sections else ["본문", "결론"]

    def run(self, **directive) -> None:
        self._classify(**directive)

    def _classify(
            self,
            num_selects: Optional[int] = None,
            num_tries: Optional[int] = None,
            recovery_time: Optional[int] = None,
            **kwargs
        ) -> None:
        self.INFO(f"{self.header} CLASSIFY proceed ....")

        sample_df = self.select_samples(
            source_df=self.dataset_df["test"],
            num_selects=num_selects,
            keywords=["유형", "sample"],
        )

        num_tries = optional(num_tries, 5)
        recovery_time = optional(recovery_time, 2)

        context = self.classify_pre()

        self.result_df = sample_df.apply(
            self._classify_foreach,
            labels=self.get_type_labels(),
            count=itertools.count(1),
            total=sample_df.shape[0],
            num_tries=num_tries,
            recovery_time=recovery_time,
            context=context,
            axis=1
        )

        self._classify_post()

        self.INFO(f"{self.header} CLASSIFY proceed DONE")

    @abstractmethod
    def classify_pre(self) -> tuple[
        dict[str, Any],
        dict[str, BaseCallbackHandler]
    ]:
        pass

    def _classify_foreach(self,
                         field: pd.Series,
                         labels: list[str],
                         count: Callable[..., Iterator[int]],
                         total: int,
                         num_tries: int,
                         recovery_time: int,
                         context: ChainContext
                         ) -> pd.Series:
        content = "\n".join([
            f"{section}: {field[section]}" for section in self.sections
        ])

        question, *_ = load_text(
            self.content_dir / self.question,
            {"content": content},
        )

        sample = field["sample"]

        for i in range(num_tries):
            try:
                timer = Timer()

                response = self.main_chain.invoke(
                    input={
                        "content": content,
                        "question": question,
                    } | context.parameter,

                    config=self.chain_config,
                )

            except KeyboardInterrupt as error:
                raise error

            except BaseException:
                self.ERROR(f"LLM({self.model[0].model_name}) internal error")
                self.ERROR(f"number of tries {i + 1}/{num_tries}")

                Timer.delay(recovery_time)
            else:
                break
        else:
            self.ERROR(f"number of tries exceeds {num_tries}")

            response = PDACResultClassify(
                type="Not available",
                reason = f"LLM({self.model[0].model_name}) internal error",
            )

        timer.lap()

        entry = pd.Series({
            "sample": sample,
        } | {
            section: field[section]
            for section in self.sections
        } | {
            "정답": field["유형"],
            "예측": match_text(response.type, labels),
            "원예측": response.type,
            "근거": response.reason,
            "소요시간": timer.seconds,
        } | {
            name: handler()
            for name, handler in context.handler.items()
        })

        self.INFO(
            f"SAMPLE [{sample:02d}]: TIME(sec):{entry['소요시간']:.2f}   "
            f"{'HIT' if entry['정답'] == entry['예측'] else 'MISS'}  "
            f"(GT:{Q(entry['정답'])} PV:{Q(entry['예측'])})"
        )

        self.INFO(
            f"{self.header} CLASSIFY proceed DONE "
            f"[{next(count):02d}/{total:02d}] proceed DONE"
        )
        return entry

    def _classify_post(self):
        self.result_df[
            [
                "suite_id",
                "case_id",
            ]
        ] = [
            self.suite_id,
            self.case_id,
        ]

        total = self.result_df.shape[0]
        hit = (self.result_df["정답"] == self.result_df["예측"]).sum()
        hit_ratio = (hit / total) * 100

        self.INFO(
            f"{self.header} hit ratio: {hit_ratio:.1f}% ({hit}/{total})"
        )

    def stringfy_trainsets(self) -> str:
        trainset_df = self.dataset_df["train"].reset_index(drop=True)

        trainsets = trainset_df.apply(
            self._stringfy_trainset_foreach,
            axis=1
        )

        return "\n\n".join(trainsets)

    def _stringfy_trainset_foreach(self, field: pd.Series) -> str:
        section_text = "\n".join([
            f"**{section}**: {field[section]}"
            for section in self.sections
        ])

        return (
            f"*예제 {field.name + 1}. CT 판독문*:\n"
            f"**유형**: {field['유형']}\n"
            f"{section_text}"
        )

    def get_type_labels(self) -> list[str]:
        return (
            self.dataset_df["train"]["유형"].unique().tolist() +
            ["Not available"]
        )

    @staticmethod
    def reorder_documents(documents: list[Document]) -> list[Document]:
        reordering = LongContextReorder()
        return reordering.transform_documents(documents)

    @property
    def sections(self) -> list[str]:
        return self._sections

    @property
    def question(self) -> str:
        return self._question

    @question.setter
    def question(self, question: str) -> None:
        self._question = question
