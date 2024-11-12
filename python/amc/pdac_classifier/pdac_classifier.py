from __future__ import annotations
from typing import Any

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
from cenai_core.grid import GridRunnable, GridChainContext
from cenai_core.pandas_helper import to_json


class PDACClassifyResult(BaseModel):
    category: str = Field(
        description="AI 분류기가 예측한 CT 판독문의 유형",
    )

    reason: str = Field(
        description="AI 분류기가 CT 판독문의 유형을 예측한 근거",
    )


class PDACClassifier(GridRunnable, ABC):
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

    def run(self,
            directive: dict[str, Any],
            input_df: pd.DataFrame = pd.DataFrame()
            ) -> None:

        self.classify(
            head=directive["head"],
            input_df=input_df
        )

    def classify(self,
                 head: int = 0,
                 input_df: pd.DataFrame = pd.DataFrame()
                 ) -> None:
        self.INFO(
            f"RUN {Q(self.run_id)}[{Q(self.batch_id)}] CLASSIFY proceed ...."
        )

        if input_df.empty:
            input_df = self.example_df["test"]

            if input_df.empty:
                raise ValueError(
                    "No input data for PDAC classification"
                )

            total = input_df.shape[0]
            if head > 0:
                input_df = input_df.head(head)

        context = self.classify_pre()

        self.result_df = input_df.apply(
            self.classify_foreach,
            category_text=self.get_category_text,
            category_labels=self.get_category_labels,
            partial=input_df.shape[0],
            total=total,
            context=context,
            axis=1,
        )

        self.classify_post()

        self.INFO(
            f"RUN {Q(self.run_id)}[{Q(self.batch_id)}] CLASSIFY proceed DONE"
        )

    @abstractmethod
    def classify_pre(self) -> tuple[
        dict[str, Any],
        dict[str, BaseCallbackHandler]
    ]:
        pass

    def classify_foreach(self,
                         field: pd.Series,
                         category_text: str,
                         category_labels: list[str],
                         partial: int,
                         total: int,
                         context: GridChainContext
                         ) -> pd.Series:

        content = "\n".join([
            f"{section}: {field[section]}" for section in self.sections
        ])

        question = f"""
        *사용자 질문*: 입력된 CT 판독문이 어떤 유형에 속하는지와 그 근거는 무엇입니까?

        *CT 판독문 내용*:
        {content}
        """

        timer = Timer()

        try:
            answer = self.classifier_chain.invoke(
                {
                    "question": dedent(question),
                    "category_text": category_text,
                    "sections": "과 ".join(self.sections)
                } | context.parameter
            )
        except BaseException:
            self.ERROR(f"LLM({self.model.model_name}) internal error")
            category = "Not available"
            reason = f"LLM({self.model.model_name}) internal error"
        else:
            category = answer.category
            reason = answer.reason

        timer.lap()

        entry = pd.Series({
            "batch_id": self.batch_id,
            "run_id": self.run_id,
            "partial": partial,
            "total": total,
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
            f"RUN {Q(self.run_id)}[{Q(self.batch_id)}] CLASSIFY proceed DONE "
            f"[{field.name + 1:02d}/{partial:02d}] proceed DONE"
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
            f"RUN {Q(self.run_id)}[{Q(self.batch_id)}] "
            f"hit ratio: {hit_ratio:.1f}% ({hit}/{total})"
        )

    def save_data(self,
                  directive: dict[str, Any]
                  ) -> None:
        self.INFO(f"RUN {Q(self.run_id)}[{Q(self.batch_id)}] DATA saved ....")

        datastore_dir = self.datastore_dir / self.batch_id
        datastore_dir.mkdir(parents=True, exist_ok=True)
        data_json = datastore_dir / f"{self.run_id}.json"

        to_json(data_json, self.metadata_df, self.result_df)

        self.INFO(f"RUN {Q(self.run_id)}[{Q(self.batch_id)}] DATA saved DONE")

    def stringfy_examples(self) -> str:
        example_df = self.example_df["train"].reset_index(drop=True)

        examples = example_df.apply(
            self._stringfy_example_foreach,
            axis=1
        )

        return "\n\n".join(examples)

    def _stringfy_example_foreach(self, field: pd.Series) -> str:
        section_text = "\n".join([
            f"**{section}**: {field[section]}"
            for section in self.sections
        ])

        return (
            f"*예제 {field.name + 1}. CT 판독문*:\n"
            f"**유형**: {field['유형']}\n"
            f"{section_text}"
        )

    def stringfy_categories(self) -> str:
        prefix = self.dataset.split("_")[0]
        txt_file = self.source_dir / f"{prefix}.txt"
        text = txt_file.read_text('utf-8')
        return text

    def get_category_labels(self) -> list[str]:
        return self.example_df["train"]["유형"].unique().tolist() + ["Not available"]

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
