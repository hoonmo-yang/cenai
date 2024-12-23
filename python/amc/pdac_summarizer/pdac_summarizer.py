from typing import Any, Callable, Iterator, Sequence

import itertools
import pandas as pd

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.utils import Output

from cenai_core import Timer
from cenai_core.dataman import load_text, Q, Struct
from cenai_core.grid import GridRunnable
from cenai_core.nlp import match_text

from pdac_template import PDACReportTemplateFail


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

    def stream(self,
               messages: Sequence[dict[str, str] | tuple[str, str]],
               **kwargs) -> Iterator[Output]:
        return iter([])

    def invoke(self, **directive) -> None:
        num_selects = directive.get("num_selects", 1)
        num_tries = directive.get("num_tries", 10)
        recovery_time = directive.get("recovery_time", 0.5)

        self._summarize(
            num_selects=num_selects,
            num_tries=num_tries,
            recovery_time=recovery_time,
        )

    def _summarize(
            self,
            num_selects: int,
            num_tries: int,
            recovery_time: float
        ) -> None:
        self.INFO(f"{self.header} SUMMARIZE proceed ....")

        sample_df = self.select_samples(
            source_df=self.dataset_df["test"],
            num_selects=num_selects,
            keywords=["유형", "sample_id"],
        )

        self.result_df = sample_df.apply(
            self._summarize_foreach,
            labels=self.get_type_labels(),
            count=itertools.count(1),
            total=sample_df.shape[0],
            num_tries=num_tries,
            recovery_time=recovery_time,
            axis=1
        ).pipe(
            self._prepare_htmls
        )

        self.INFO(f"{self.header} SUMMARIZE proceed DONE")

    def _summarize_foreach(self,
                          field: pd.Series,
                          labels: list[str],
                          count: Callable[..., Iterator[int]],
                          total: int,
                          num_tries: int,
                          recovery_time: float
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

            except KeyboardInterrupt:
                raise

            except BaseException:
                self.ERROR(f"LLM({self.model[0].model_name}) internal error")
                self.ERROR(f"number of tries {i + 1}/{num_tries}")

                Timer.delay(recovery_time)
                recovery_time *= 2
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
        sample_id = int(field.sample_id)

        summary = pdac_report.model_dump()

        is_hit = gt_type == pv_type

        entry = pd.Series({
            "gt_type": gt_type,
            "pv_type": pv_type,
            "summary": summary,
            "content": content,
            "sample_id": sample_id,
            "hit": is_hit,
            "time": timer.seconds,
        })

        self.INFO(
            f"SAMPLE {sample_id:03d} GT:{Q(gt_type)} PV:{Q(pv_type)} "
            f"[{'HIT' if is_hit else "MISS"}] "
            f"TIME: {timer.seconds:.1f}s"
        )

        self.INFO(
            f"{self.header} SUMMARIZE proceed DONE "
            f"[{next(count):02d}/{total:02d}] proceed DONE"
        )
        return entry

    def _prepare_htmls(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        self.INFO(f"{self.header} SUMMARY HTML PREPARATION proceed ....")

        columns = [
            "css_file",
            "html_file",
            "html_args",
        ]

        sample_df[columns] = sample_df.apply(
            self._prepare_html_foreach,
            count=itertools.count(1),
            total=sample_df.shape[0],
            axis=1
        )

        columns = [
            "suite_id",
            "case_id",
        ]
        sample_df[columns] = [self.suite_id, self.case_id]

        self.INFO(f"{self.header} SUMMARY HTML PREPARATION proceed DONE")
        return sample_df

    def _prepare_html_foreach(self, 
                              sample: pd.Series,
                              count: Callable[..., Iterator[int]],
                              total: int,
                             ) -> pd.Series:

        summary_df = pd.json_normalize(sample.summary, sep="__")
        summary_args = summary_df.to_dict(orient="records")[0]

        html_args = {
            "sample_id": sample.sample_id,
            "gt_type": sample.gt_type,
            "pv_type": sample.pv_type,
            "hit": "HIT" if sample.hit else "MISS",
            "content": sample.content
        } | summary_args

        css_file = self.html_dir / "styles.css"
        index = self.get_type_label_index(sample.pv_type) + 1
        html_file = self.html_dir / f"html_template_type{index}.html"

        entry = pd.Series({
            "css_file": str(css_file),
            "html_file": str(html_file),
            "html_args": html_args,
        })

        self.INFO(
            f"{self.header} SAMPLE HTML {sample.sample_id:03d} "
            f"[{next(count):02d}/{total:02d}] DONE"
        )

        return entry

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
    def reorder_documents(documents: list[Document]) -> list[Document]:
        reordering = LongContextReorder()
        return reordering.transform_documents(documents)

    @property
    def question(self) -> str:
        return self._question

    @question.setter
    def question(self, question: str) -> None:
        self._question = question
