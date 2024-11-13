from __future__ import annotations
from typing import Any, Optional

from abc import ABC, abstractmethod
import pandas as pd
from pydantic import BaseModel, Field

from langchain_community.document_transformers import LongContextReorder
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import Runnable, RunnableLambda
from langchain.schema import Document

from cenai_core import Timer
from cenai_core.dataman import concat_texts, dedent, Q, Struct
from cenai_core.nlp import match_text
from cenai_core.grid import GridRunnableDataset, GridChainContext
from cenai_core.pandas_helper import to_json


class PDACClassifyResult(BaseModel):
    category: str = Field(
        description="AI 분류기가 예측한 CT 판독문의 유형",
    )

    reason: str = Field(
        description="AI 분류기가 CT 판독문의 유형을 예측한 근거",
    )


class PDACClassifier(GridRunnableDataset, ABC):
    logger_name = "cenai.amc.pdac_classifier"

    def __init__(self,
                 metadata: Struct,
                 module_suffix: str,
                 sections: list[str]
                 ):

        self._sections = self._get_sections(sections)

        dataset_suffix = "".join([
            "b" if section == "본문" else "c"
            for section in self._sections
        ])

        super().__init__(
            metadata=metadata,
            dataset_suffix=dataset_suffix,
            module_suffix=module_suffix,
        )

        self.metadata_df.loc[0, "sections"] = ",".join(self._sections)

        self._classifier_chain = RunnableLambda(
            lambda _: PDACClassifier(
                category="Not available",
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

    def run(self, directive: dict[str, Any]) -> None:
        self.classify(
            head=directive.get("head"),
            num_tries=directive.get("num_tries"),
            recovery_time=directive.get("recovery_time"),
        )

    def classify(self,
                 head: Optional[int],
                 num_tries: Optional[int],
                 recovery_time: Optional[int]
                 ) -> None:
        self.INFO(f"{self.header} CLASSIFY proceed ....")

        testset_df = self.dataset_df["test"]

        if isinstance(head, int):
            testset_df = testset_df.head(head)

        if num_tries is None:
            num_tries = 5

        if recovery_time is None:
            num_tries = 2

        context = self.classify_pre()

        self.result_df = testset_df.apply(
            self.classify_foreach,
            category_labels=self.get_category_labels(),
            total=testset_df.shape[0],
            num_tries=num_tries,
            recovery_time=recovery_time,
            context=context,
            axis=1,
        )

        self.classify_post()

        self.INFO(f"{self.header} CLASSIFY proceed DONE")

    @abstractmethod
    def classify_pre(self) -> tuple[
        dict[str, Any],
        dict[str, BaseCallbackHandler]
    ]:
        pass

    def classify_foreach(self,
                         field: pd.Series,
                         category_labels: list[str],
                         total: int,
                         num_tries: int,
                         recovery_time: int,
                         context: GridChainContext
                         ) -> pd.Series:

        content = "\n".join([
            f"{section}: {field[section]}" for section in self.sections
        ])

        question = (
            f"""
            *User question*: What category does the input CT report belong to, and
            what is the reasoning?

            *CT report content*:
            {content}
            """
        if self.prompt.stem.endswith("en") else
            f"""
            *사용자 질문*: 입력된 CT 판독문이 어떤 유형에 속하는지와 그 근거는 무엇입니까?

            *CT 판독문 내용*:
            {content}
            """
        )

        for i in range(num_tries):
            try:
                timer = Timer()

                answer = self.classifier_chain.invoke(
                    {
                        "content": dedent(content),
                        "question": dedent(question),
                    } | context.parameter
                )
            except BaseException:
                self.ERROR(f"LLM({self.model.model_name}) internal error")
                self.ERROR(f"number of tries {i + 1}/{num_tries}")

                Timer.delay(recovery_time)

            else:
                category = answer.category
                reason = answer.reason
                break
        else:
            self.ERROR(f"number of tries exceeds {num_tries}")

            category = "Not available"
            reason = f"LLM({self.model.model_name}) internal error"

        timer.lap()

        entry = pd.Series({
            "grid_id": self.grid_id,
            "run_id": self.run_id,
        } | {
            section: field[section]
            for section in self.sections
        } | {
            "정답": field["유형"],
            "예측": match_text(category, category_labels),
            "원예측": category,
            "근거": reason,
            "소요시간": timer.seconds,
        } | {
            name: handler()
            for name, handler in context.handler.items()
        })

        self.INFO(
            f"TIME(sec):{entry['소요시간']:.2f}   "
            f"{'HIT' if entry['정답'] == entry['예측'] else 'MISS'}  "
            f"(GT:{Q(entry['정답'])} PV:{Q(entry['예측'])})"
        )

        self.INFO(
            f"{self.header} CLASSIFY proceed DONE "
            f"[{field.name + 1:02d}/{total:02d}] proceed DONE"
        )
        return entry

    def classify_post(self):
        total = self.result_df.shape[0]

        self.result_df["hit"] = (
            self.result_df["정답"] == self.result_df["예측"]
        )

        hit = self.result_df["hit"].sum()
        hit_ratio = (hit / total) * 100

        self.INFO(
            f"{self.header} hit ratio: {hit_ratio:.1f}% ({hit}/{total})"
        )

    def save_data(self,
                  directive: dict[str, Any]
                  ) -> None:
        self.INFO(f"{self.header} DATA saved ....")

        if directive.get("save", True):
            datastore_dir = self.datastore_dir / self.grid_id
            datastore_dir.mkdir(parents=True, exist_ok=True)
            data_json = datastore_dir / f"{self.run_id}.json"

            to_json(data_json, self.metadata_df, self.result_df)

            self.INFO(f"{self.header} DATA saved DONE")
        else:
            self.INFO(f"{self.header} DATA saved SKIP")

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

    def get_category_labels(self) -> list[str]:
        return (
            self.dataset_df["train"]["유형"].unique().tolist() +
            ["Not available"]
        )

    @staticmethod
    def concat_documents(documents: list[Document]) -> str:
        return concat_texts(documents, "page_content", "\n\n")

    @staticmethod
    def reorder_documents(documents: list[Document]) -> list[Document]:
        reordering = LongContextReorder()
        return reordering.transform_documents(documents)

    @property
    def sections(self) -> list[str]:
        return self._sections

    @property
    def classifier_chain(self) -> Runnable:
        return self._classifier_chain

    @classifier_chain.setter
    def classifier_chain(self, chain: Runnable) -> None:
        self._classifier_chain = chain
