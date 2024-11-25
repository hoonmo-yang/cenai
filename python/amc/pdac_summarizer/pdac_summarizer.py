from __future__ import annotations
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from cenai_core import Timer
from cenai_core.dataman import concat_texts, load_text, optional, Q, Struct
from cenai_core.grid import GridRunnable
from cenai_core.nlp import match_text
from amc.pdac_summarizer.pdac_template import PDACReportTemplateFail


class PDACClassifyResult(BaseModel):
    type: str = Field(
        description="AI 분류기가 예측한 CT 판독문의 유형",
    )


class PDACSummarizer(GridRunnable):
    logger_name = "cenai.amc.pdac_summarizer"

    def __init__(self,
                 model: str,
                 case_suffix: str,
                 metadata: Struct
                 ):

        super().__init__(
            model=model,
            case_suffix=case_suffix,
            corpus_suffix="",
            metadata=metadata,
        )

        self._summarize_chain = RunnableLambda(
            lambda _: PDACReportTemplateFail(message="Not initialized yet")
        )

    def run(self, **directive) -> None:
        self._summarize(**directive)

    def _summarize(
            self,
            num_selects: Optional[int] = None,
            num_tries: Optional[int] = None,
            recovery_time: Optional[int] = None,
            **kwargs
        ) -> None:
        self.INFO(f"{self.header} SUMMARIZE proceed ....")

        testset_df = self._select_testsets(
            self.dataset_df["test"], num_selects
        )

        css_file = self.html_dir / "styles.css"
        css_text = f"<style>\n{css_file.read_text()}\n</style>"

        num_tries = optional(num_tries, 5)
        recovery_time = optional(recovery_time, 2)

        self.result_df = testset_df.apply(
            self._summarize_foreach,
            labels=self.get_type_labels(),
            css_text=css_text,
            total=testset_df.shape[0],
            num_tries=num_tries,
            recovery_time=recovery_time,
            axis=1
        )

        self.INFO(f"{self.header} SUMMARIZE proceed DONE")

    def _select_testsets(self,
                         testset_df: pd.DataFrame,
                         num_selects: Optional[int],
                         ) -> pd.DataFrame:
        num_selects = optional(num_selects, 0)

        if not num_selects:
            return testset_df

        sample_df = testset_df.groupby(["유형"]).apply(
                lambda field:
                    field.sample(
                        min(len(field), num_selects),
                        random_state=0,
                    )
            ).droplevel(0).reset_index().rename(columns={"index": "sample"})

        return sample_df

    def _summarize_foreach(self,
                          field: pd.Series,
                          labels: list[str],
                          css_text: str,
                          total: int,
                          num_tries: int,
                          recovery_time: int
                         ) -> pd.Series:
        content = field["본문"]

        question, *_ = load_text(
            self.content_dir / self.question,
            {"content": content},
        )

        for i in range(num_tries):
            try:
                timer = Timer()

                pdac_report = self.summarize_chain.invoke({
                    "content": content,
                    "question": question,

                })

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

            pdac_report = PDACReportTemplateFail(
                message="LLM internal error during summarization",
            )

        timer.lap()

        gt_type = field["유형"]
        pv_type = match_text(pdac_report.type, labels)
        sample = int(field["sample"])

        summary = pdac_report.model_dump()

        is_hit = gt_type == pv_type

        html = self._generate_html(
            sample=sample,
            css_text=css_text,
            summary=summary,
            gt_type=gt_type,
            pv_type=pv_type,
            is_hit=is_hit,
        )

        entry = pd.Series({
            "suite_id": self.suite_id,
            "case_id": self.case_id,
            "gt_type": gt_type,
            "pv_type": pv_type,
            "summary": summary,
            "content": content,
            "sample": sample,
            "html": html,
            "hit": is_hit,
            "time": timer.seconds,
        })

        self.INFO(
            f"SAMPLE {int(sample):03d} GT:{Q(gt_type)} PV:{Q(pv_type)} "
            f"[{'HIT' if is_hit else "MISS"}] "
            f"TIME: {timer.seconds:.1f}s"
        )

        self.INFO(
            f"{self.header} SUMMARIZE proceed DONE "
            f"[{int(field.name) + 1:02d}/{total:02d}] proceed DONE"
        )
        return entry

    def _generate_html(self, 
                       sample: int,
                       css_text: str,
                       summary: dict[str, Any],
                       gt_type: str,
                       pv_type: str,
                       is_hit: bool
                       ) -> str:

        index = self.get_type_label_index(pv_type) + 1
        html_file = self.html_dir / f"html_template_type{index}.html"
        html_text = html_file.read_text()

        summary_df = pd.json_normalize(summary, sep="__")
        summary_args = summary_df.to_dict(orient="records")[0]

        html_args = {
            "css_content": css_text,
            "case_id": self.case_id,
            "sample": sample,
            "gt_type": gt_type,
            "pv_type": pv_type,
            "hit": "HIT" if is_hit else "MISS",
        } | summary_args

        html = html_text.format(**html_args)
        return html

    def stringfy_trainsets(self) -> str:
        trainset_df = self.dataset_df["train"].reset_index(drop=True)

        trainsets = trainset_df.apply(
            self._stringfy_trainset_foreach,
            axis=1
        )

        return "\n\n".join(trainsets)

    def _stringfy_trainset_foreach(self, field: pd.Series) -> str:
        return (
            f"*예제 {field.name + 1}. CT 판독문*:\n"
            f"**유형**: {field['유형']}\n"
            f"**본문**: {field['본문']}"
        )

    def get_type_labels(self) -> list[str]:
        return (
            sorted(self.dataset_df["train"]["유형"].unique().tolist()) +
            ["Not available"]
        )

    def get_type_label_index(self, label: str) -> int:
        indices = [
            i for i, value in enumerate(self.get_type_labels()[:-1])
            if value == label
        ]

        return indices[0] if indices else -1

    @staticmethod
    def concat_documents(documents: list[Document]) -> str:
        return concat_texts(documents, "page_content", "\n\n")

    @staticmethod
    def reorder_documents(documents: list[Document]) -> list[Document]:
        reordering = LongContextReorder()
        return reordering.transform_documents(documents)

    @property
    def summarize_chain(self) -> Runnable:
        return self._summarize_chain

    summarize_chain.setter
    def summarize_chain(self, chain: Runnable) -> None:
        self._summarize_chain = chain

    @property
    def question(self) -> str:
        return self._question

    @question.setter
    def question(self, question: str) -> None:
        self._question = question
