from __future__ import annotations
from typing import Any, Callable, Iterator, Optional, Sequence

import itertools
import pandas as pd

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from cenai_core import Timer
from cenai_core.dataman import load_text, optional, Q, Struct
from cenai_core.grid import GridRunnable
from cenai_core.nlp import match_text
from amc.pdac_summarizer.pdac_template import PDACReportTemplateFail


class PDACSummarizer(GridRunnable):
    logger_name = "cenai.amc.pdac_summarizer"

    def __init__(self,
                 models: Sequence[str],
                 case_suffix: str,
                 metadata: Struct
                 ):
        corpus_part = metadata.corpus_stem

        super().__init__(
            models=models,
            case_suffix=case_suffix,
            corpus_part=corpus_part,
            metadata=metadata,
        )

        self.main_chain = RunnableLambda(
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

        sample_df = self.select_samples(
            source_df=self.dataset_df["test"],
            num_selects=num_selects,
            keywords=["유형", "sample"],
        )

        css_file = self.html_dir / "styles.css"
        css_text = f"<style>\n{css_file.read_text()}\n</style>"

        num_tries = optional(num_tries, 5)
        recovery_time = optional(recovery_time, 2)

        self.result_df = sample_df.apply(
            self._summarize_foreach,
            labels=self.get_type_labels(),
            css_text=css_text,
            count=itertools.count(1),
            total=sample_df.shape[0],
            num_tries=num_tries,
            recovery_time=recovery_time,
            axis=1
        )

        self._summarize_post()

        self.INFO(f"{self.header} SUMMARIZE proceed DONE")

    def _summarize_foreach(self,
                          field: pd.Series,
                          labels: list[str],
                          css_text: str,
                          count: Callable[..., Iterator[int]],
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

                pdac_report = self.main_chain.invoke(
                    input={
                        "content": content,
                        "question": question,
                    },
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

            pdac_report = PDACReportTemplateFail(
                message="LLM internal error during summarization",
            )

        timer.lap()

        gt_type = field["유형"]
        pv_type = match_text(pdac_report.type, labels)
        sample = field["sample"]

        summary = pdac_report.model_dump()

        is_hit = gt_type == pv_type

        html = self._generate_html(
            sample=sample,
            css_text=css_text,
            summary=summary,
            gt_type=gt_type,
            pv_type=pv_type,
            is_hit=is_hit,
            content=content,
        )

        entry = pd.Series({
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
            f"[{next(count):02d}/{total:02d}] proceed DONE"
        )
        return entry

    def _generate_html(self, 
                       sample: int,
                       css_text: str,
                       summary: dict[str, Any],
                       gt_type: str,
                       pv_type: str,
                       is_hit: bool,
                       content: str
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
            "content": content
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

    def _summarize_post(self):
        self.result_df[
            [
                "suite_id",
                "case_id",
            ]
        ] = [
            self.suite_id,
            self.case_id,
        ]

    @staticmethod
    def reorder_documents(documents: list[Document]) -> list[Document]:
        reordering = LongContextReorder()
        return reordering.transform_documents(documents)

    @property
    def question(self) -> str:
        return self._question

    @question.setter
    def question(self, question: str) -> None:
        self._question = question
