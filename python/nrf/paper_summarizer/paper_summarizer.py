from __future__ import annotations
from typing import Any, Callable, Iterator, Optional, Union

import itertools
import json
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from cenai_core import Timer

from cenai_core.dataman import (
    load_json_yaml, optional, pad_list, Q, Struct
)

from cenai_core.grid import GridRunnable

from cenai_core.langchain_helper import (
    LineTextSplitter, load_documents, load_prompt
)

from nrf.paper_summarizer.paper_template import (
    PaperResultFail, PaperResultSimilarity, PaperSummaryTemplate
)


class PaperSummarizer(GridRunnable):
    logger_name = "cenai.nrf.paper_classifier"

    def __init__(self,
                 models: list[str],
                 num_keywords: int,
                 max_tokens: int,
                 extract_gt_prompt: str,
                 similarity_prompt: str,
                 case_suffix: str,
                 metadata: Struct
                 ):
        
        case_suffix = "_".join([
            extract_gt_prompt.split(".")[0],
            similarity_prompt.split(".")[0],
            case_suffix,
            f"kw{num_keywords}",
            f"tk{max_tokens}",
        ])

        corpus_stem = metadata.corpus_stem
        corpus_ext = metadata.corpus_ext

        if isinstance(corpus_stem, str):
            corpus_stem = [corpus_stem]

        if isinstance(corpus_ext, str):
            corpus_ext = [corpus_ext]

        corpus_part = "_".join([
            metadata.corpus_prefix,
            "-".join(
                [stem for stem in corpus_stem if stem]
            ),
            "-".join([
                extension[1:] for extension in corpus_ext
                if extension]),
        ])

        super().__init__(
            models=models,
            case_suffix=case_suffix,
            corpus_part=corpus_part,
            metadata=metadata,
        )

        self._num_keywords = num_keywords
        self._max_tokens = max_tokens

        self.metadata_df.loc[
            0,
            [
                "extract_gt_prompt",
                "similarity_prompt",
                "num_keywords",
                "max_tokens",
             ]
        ] = [
            extract_gt_prompt,
            similarity_prompt,
            num_keywords,
            max_tokens,
        ]

        self._layout = self._get_layout()

        self._extract_gt_chain = self._build_extract_gt_chain(
            extract_gt_prompt=extract_gt_prompt,
        )

        self._similarity_chain = self._build_similarity_chain(
            similarity_prompt=similarity_prompt,
        )

        css_file = self.html_dir / "styles.css"
        self._css_text = f"<style>\n{css_file.read_text()}\n</style>"

        html_file = self.html_dir / "html_summary.html"
        self._html_summary = html_file.read_text()

        html_file = self.html_dir / "html_eval.html"
        self._html_eval = html_file.read_text()

    def _build_extract_gt_chain(self,
                                extract_gt_prompt: str,
                                ) -> Runnable:
        self.INFO(f"{self.header} EXTRACT GT CHAIN prepared ....")

        parser = PydanticOutputParser(
            pydantic_object=PaperSummaryTemplate,
        )

        prompt_args, partials = load_prompt(
            self.content_dir / extract_gt_prompt
        )

        full_args = prompt_args | {
            "partial_variables": {
                partials[0]: parser.get_format_instructions(),
            },
        }

        prompt = PromptTemplate(**full_args)

        chain = prompt | self.model[1] | parser

        self.INFO(f"{self.header} EXTRACT GT CHAIN prepared DONE")
        return chain

    def _build_similarity_chain(self,
                                similarity_prompt: str,
                                ) -> Runnable:
        self.INFO(f"{self.header} SIMILARITY CHAIN prepared ....")

        parser = PydanticOutputParser(
            pydantic_object=PaperResultSimilarity,
        )

        prompt_args, partials = load_prompt(
            self.content_dir / similarity_prompt
        )

        full_args = prompt_args | {
            "partial_variables": {
                partials[0]: parser.get_format_instructions(),
            },
        }

        prompt = PromptTemplate(**full_args)

        chain = prompt | self.model[1] | parser

        self.INFO(f"{self.header} SIMILARITY CAHIN prepared DONE")
        return chain

    def _get_layout(self) -> Struct:
        layout_file = self.content_dir / "paper-layout.yaml"
        layout = load_json_yaml(layout_file)

        return Struct({
            "source": layout["source_template"],
            "summary": layout["summary_template"],
        })

    def run(self, **directive) -> None:
        self._summarize(**directive)

    def _summarize(
            self,
            num_tries: Optional[int] = None,
            recovery_time: Optional[int] = None,
            **kwargs
            ) -> None:

        self.INFO(f"{self.header} PAPER SUMMARY proceed ....")

        num_tries = optional(num_tries, 5)
        recovery_time = optional(recovery_time, 2)

        paper_df = self._split_papers_by_section()

        pv_df = paper_df.pipe(
            self._gather_paper_sections,
        ).pipe(
            self._summarize_papers,
            num_tries=num_tries,
            recovery_time=recovery_time,
        )

        gt_df = paper_df[
            paper_df.section == "summary"
        ].pipe(
            self._extract_summary_sections,
            num_tries=num_tries,
            recovery_time=recovery_time,
        )

        self.result_df = pd.merge(
            pv_df, gt_df,
            on=["file"],
            how="outer",
            suffixes=["_pv", "_gt"],
        ).pipe(
            self._compare_similarity,
            num_tries=num_tries,
            recovery_time=recovery_time,
        )

        self.INFO(f"{self.header} PAPER SUMMARY proceed DONE")

    def _compare_similarity(self,
                            result_df: pd.DataFrame,
                            num_tries: int,
                            recovery_time: int
                            ) -> pd.DataFrame:
        columns = ["file", "similarity", "difference", "html_eval"]

        result_df[columns] = result_df.apply(
            self._compare_similarity_foreach,
            count=itertools.count(1),
            total=result_df.shape[0],
            num_tries=num_tries,
            recovery_time=recovery_time,
            axis=1
        )

        return result_df

    def _compare_similarity_foreach(self,
                                    field: pd.Series,
                                    count: Callable[..., Iterator[int]],
                                    total: int,
                                    num_tries: int,
                                    recovery_time: int
                                    ) -> pd.Series:
        file_ = field.file
        summary_pv = json.dumps(field.summary_pv, ensure_ascii=False)
        summary_gt = json.dumps(field.summary_gt, ensure_ascii=False)
        
        for i in range(num_tries):
            try:
                timer = Timer()

                response = self.similarity_chain.invoke(
                    input={
                        "gt_content": summary_pv,
                        "pv_content": summary_gt,
                    },
                )

            except KeyboardInterrupt as error:
                raise error

            except BaseException:
                self.ERROR(f"LLM({self.model[1].model_name}) internal error")
                self.ERROR(f"number of tries {i + 1}/{num_tries}")

                Timer.delay(recovery_time)
            else:
                break
        else:
            self.ERROR(f"number of tries exceeds {num_tries}")

            response = PaperResultSimilarity(
                score=0.0,
                difference="LLM internal error",
            )

        args = {
            "file": str(field.file),
            "similarity": response.score,
            "difference":  response.difference,
        }

        html_args = {"css_content": self._css_text} | args
        html_eval = self._html_eval.format(**html_args)
        entry = pd.Series(args|{"html_eval": html_eval,})

        self.INFO(
            f"** FILE {Q(file_.name)} [{next(count):02d}/{total:02d}] "
            f"TIME: {timer.seconds:.1f}s SIMILARITY proceed DONE"
        )
        return entry

    def _split_papers_by_section(self) -> pd.DataFrame:
        splitter = LineTextSplitter(chunk_size=50)
        input_df = self.document_df

        output_df = input_df.apply(
            self._split_paper_by_section_foreach,
            splitter=splitter,
            count=itertools.count(1),
            total=input_df.shape[0],
            axis=1
        ).explode(
            ["document", "section"],
        ).reset_index(
            drop=True,
        ).groupby(
            ["file", "section"], sort=False,
        )["document"].apply(
            self._merge_paper_by_section_foreach,
        ).reset_index()

        return output_df

    def _split_paper_by_section_foreach(
            self,
            field: pd.Series,
            splitter: LineTextSplitter,
            count: Callable[..., Iterator[int]],
            total: int
        ) -> pd.Series:
        file_ = field.file

        documents = load_documents(file_)

        content = "\n".join([document.page_content for document in documents])

        merged_documents = [
            Document(
                page_content=content,
                metadata={"source": str(file_)}
            )
        ]

        split_documents = splitter.split_documents(merged_documents)
        documents, sections = self._annotate_paper_by_section(split_documents)

        self.INFO(
            f"** FILE {Q(file_.name)} SECTION SPLIT "
            f"[{next(count):02d}/{total:02d}] proceed DONE"
        )

        return pd.Series({
            "file": file_,
            "section": sections,
            "document": documents,
        })

    def _annotate_paper_by_section(
            self,
            documents: list[Document]
        ) -> tuple[list[Document], list[str]]:

        section = ""

        targets = []
        sections = []

        for document in documents:
            target, section = self._annotate_document_by_section(
                document, section
            )

            targets.append(target)
            sections.append(section)

        return targets, sections

    def _annotate_document_by_section(
            self,
            document: Document,
            section: str
        ) -> tuple[Document, str]:

        sections = list(self.layout.source.keys())
        titles = list(self.layout.source.values())

        results = process.extract(
            document.page_content,
            titles,
            scorer=fuzz.partial_ratio,
        )

        _, score, k = results[0]
        section = sections[k] if score > 80.0 else section
        document.metadata["section"] = section

        return document, section

    def _merge_paper_by_section_foreach(
            self, group: pd.Series
        ) -> pd.Series:

        page_content = "\n".join(
            group.apply(
                lambda field: field.page_content
            )
        )

        return pd.Series({
            "document": Document(
                            page_content=page_content,
                            metadata={
                                "file": str(group.name[0]),
                                "section": group.name[1],
                            },
                        ),
        })

    def _gather_paper_sections(self, paper_df: pd.DataFrame) -> pd.DataFrame:

        total = paper_df.groupby(["file"], sort=False).size().shape[0]

        output_df = paper_df.groupby(["file"], sort=False)[
            ["section", "document"]
        ].apply(
            self._gather_paper_sections_foreach,
            count=itertools.count(1),
            total=total,
        ).explode(
            ["item", "title", "content",]
        ).reset_index()

        return output_df

    def _gather_paper_sections_foreach(
            self,
            group: pd.DataFrame,
            count: Callable[..., Iterator[int]],
            total: int
        ) -> pd.Series:
        file_ = group.name

        items = []
        titles = []
        contents = []

        for item, record in self.layout.summary.items():
            title, sections = [
                record[key] for key in ["title", "sections"]
            ]

            page_contents = pd.Series()

            for section in sections:
                some_page_contents = group[
                    group.section == section
                ].document.apply(
                    lambda field: field.page_content
                )

                page_contents = pd.concat(
                    [page_contents, some_page_contents]
                )

            items.append(item)
            titles.append(title)
            contents.append("\n".join(page_contents))

            self.INFO(f"**** ITEM {Q(item)} TITLE {Q(title)}")

        entry = pd.Series({
            "item": items,
            "title": titles,
            "content": contents,
        })

        self.INFO(
            f"** FILE {Q(file_.name)} [{next(count)}/{total}] "
            "SECTION GATHER DONE"
        )

        return entry

    def _summarize_papers(self,
                          paper_df: pd.DataFrame,
                          num_tries: int,
                          recovery_time: int
                          ) -> pd.DataFrame:

        total = paper_df.groupby(["file"], sort=False).size().shape[0]

        summary_df = paper_df.groupby(["file"], sort=False)[
            ["item", "title", "content"]
        ].apply(
            self._summarize_paper_foreach,
            count=itertools.count(1),
            total=total,
            num_tries=num_tries,
            recovery_time=recovery_time,
        ).reset_index()
        
        return summary_df

    def _summarize_paper_foreach(self,
                                 group: pd.DataFrame,
                                 count: Callable[..., Iterator[int]],
                                 total: int,
                                 num_tries: int,
                                 recovery_time: int
                                ) -> pd.Series:
        file_ = group.name
        total2 = group.groupby(["item", "title"], sort=False).size().shape[0]

        timer = Timer()

        items = group.groupby(["item", "title"], sort=False)["content"].apply(
            self._summarize_paper_item_foreach,
            file_=file_,
            count=itertools.count(1),
            total=total2,
            num_tries=num_tries,
            recovery_time=recovery_time,
        ).reset_index().to_dict(orient="records")

        summary, html = self._formatize(items)

        timer.lap()

        self.INFO(
            f"** FILE {Q(file_.name)} [{next(count)}/{total}] "
            "SUMMARIZE proceed DONE"
        )

        return pd.Series({
            "summary": summary,
            "html": html,
            "time": timer.seconds,
        })

    def _summarize_paper_item_foreach(
            self,
            group: pd.Series,
            file_: Path,
            count: Callable[..., Iterator[int]],
            total: int,
            num_tries: int,
            recovery_time: int
        ) -> Union[str, list[list[str]]]:

        item, title = group.name
        content = group.iloc[0]

        for i in range(num_tries):
            try:
                timer = Timer()

                response = self.main_chain.invoke(
                    input={
                        "item": item,
                        "title": title,
                        "content": content,
                        "num_keywords": self._num_keywords,
                        "max_tokens": self._max_tokens,
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

            response = PaperResultFail(
                message="LLM internal error during summarization",
            )

        timer.lap()

        if not response.error:
            summary = (
                [response.keyword_kr, response.keyword_en]
                if item == "keyword" else

                response.summary
            )

        else:
            summary = [[], []] if item == "keyword" else ""

        self.INFO(
            f"**** FILE {Q(file_.name)} ITEM: {Q(item)} "
            f"TIME: {timer.seconds:.1f}s"
        )

        self.INFO(
            f"{self.header} SUMMARIZE proceed DONE "
            f"[{next(count):02d}/{total:02d}] proceed DONE"
        )

        return summary

    def _extract_summary_sections(
            self,
            paper_df: pd.DataGrame,
            num_tries: int,
            recovery_time: int
        ) -> pd.DataFrame:

        total = paper_df.groupby(["file"], sort=False).size().shape[0]

        summary_df = paper_df.groupby(["file"], sort=False)[
            ["document"]
        ].apply(
            self._extract_summary_section_foreach,
            count=itertools.count(1),
            total=total,
            num_tries=num_tries,
            recovery_time=recovery_time,
        ).reset_index()

        return summary_df

    def _extract_summary_section_foreach(
            self,
            group: pd.DataFrame,
            count: Callable[..., Iterator[int]],
            total: int,
            num_tries: int,
            recovery_time: int
        ) -> pd.Series:

        file_ = group.name

        content = "\n".join(
            group.document.apply(lambda field: field.page_content)
        )

        for i in range(num_tries):
            try:
                timer = Timer()

                response = self.extract_gt_chain.invoke(
                    input={
                        "content": content,
                    },
                )

            except KeyboardInterrupt as error:
                raise error

            except BaseException:
                self.ERROR(f"LLM({self.model[1].model_name}) internal error")
                self.ERROR(f"number of tries {i + 1}/{num_tries}")

                Timer.delay(recovery_time)
            else:
                break
        else:
            self.ERROR(f"number of tries exceeds {num_tries}")

            response = PaperSummaryTemplate(
                abstract="",
                outcome="",
                expectation="",
                keyword_kr=[],
                keyword_en=[],
            )

        timer.lap()

        items = self._generate_items(response)
        summary, html = self._formatize(items)

        timer.lap()

        self.INFO(
            f"FILE {Q(file_.name)} ITEM: summary "
            f"TIME: {timer.seconds:.1f}s"
        )

        self.INFO(
            f"{self.header} EXTRACT SUMMARY proceed DONE "
            f"[{next(count):02d}/{total:02d}] proceed DONE"
        )

        return pd.Series({
            "summary": summary,
            "html": html,
            "time": timer.seconds
        })

    def _generate_items(self,
                        summary: PaperSummaryTemplate
                        ) -> list[dict[str, Any]]:
        summary_dict = summary.model_dump()

        summary_dict["keyword"] = []

        for name in ["keyword_kr", "keyword_en"]:
            summary_dict["keyword"].append(summary_dict[name])
            summary_dict.pop(name)

        items = [
            {
                "item": key,
                "title": self.layout.summary[key]["title"],
                "content": value,
            } for key, value in summary_dict.items()
        ]

        return items

    def _formatize(self,
                   records: list[dict[str, Any]]
                   ) -> tuple[str, dict[str, Any]]:
        summary = {
            record["item"]: {
                "title": record["title"],
                "content": record["content"],
            } for record in records
        }

        html_args = {
            "css_content": self._css_text,
            "num_keywords": self._num_keywords,
        }

        for item, record in summary.items():
            title, content = [
                record[key] for key in ["title", "content"]
            ]

            args = {f"{item}_title": title}

            if item == "keyword":
                keyword_text = self._decorate_keywords(content)

                args |= {
                    "keyword_kr": keyword_text[0],
                    "keyword_en": keyword_text[1],
                }
            else:
                args |= {
                    f"{item}_content": content,
                }

            html_args |= args

        html = self._html_summary.format(**html_args)

        return summary, html

    def _decorate_keywords(self,
                           all_keywords: list[list[str]]
                           ) -> list[str]:
        all_keywords = [
            pad_list(keywords, self._num_keywords)
            for keywords in all_keywords
        ]

        return [
            "\n".join([
                f"<td>{keyword}</td>"
                for keyword in keywords
            ])
            for keywords in all_keywords
        ]

    @property
    def layout(self) -> Struct:
        return self._layout

    @property
    def extract_gt_chain(self) -> Runnable:
        return self._extract_gt_chain

    @property
    def similarity_chain(self) -> Runnable:
        return self._similarity_chain
