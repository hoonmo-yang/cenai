from typing import Callable, Iterator, Sequence

import itertools
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from langchain_core.runnables.utils import Output

from cenai_core import cenai_path, Timer
from cenai_core.dataman import load_text, Q, Struct
from cenai_core.grid import GridRunnable
from cenai_core.pandas_helper import excel_to_structured_dataframe

from cenai_core.postgres import (
    execute_script, from_rows, Postgres, to_columns, to_vars
)

from pj_schema import PJ_SCHEMA
from pj_template import PJSummaryTemplate


class PJSummarizer(GridRunnable):
    logger_name = "cenai.amc.pj_summarizer"

    def __init__(self,
                 models: Sequence[str],
                 case_suffix: str,
                 metadata: Struct
                 ):

        super().__init__(
            models=models,
            case_suffix=case_suffix,
            corpus_part="",
            metadata=metadata,
        )

        self._cenai_db = Postgres("cenai")
        self._cenai_db.connect()

        self._conn = self._cenai_db.conn

        self._build_db_tables()
        self._load_db_tables()

    def _build_db_tables(self) -> None:
        ddl_dir = cenai_path("python/amc/pj_summarizer/sql")
        ddl_file = ddl_dir / "pj_ddl.sql"

        with self.conn as conn, conn.cursor() as cursor:
            execute_script(cursor, ddl_file)

    def _load_db_tables(self) -> None:
        xlsx_file = self.content_dir / "pj-data.xlsx"

        tables = {
            "patient": ["Patient", {
                "birth_ym": "datetime", 
                "frst_vist_dt": "datetime",
                "dx_dt": "datetime",
                "cancer_reg_dt": "datetime",
                "death_dt": "datetime",
            }],

            "surgery": ["Surgery", {"oprt_dt": "datetime"}],
            "tissue": ["Tissue", {"tssu_exam_dt": "datetime"}],
            "blbm": ["BL&BM", {"blbm_exam_dt": "datetime"}],
            "biopsy": ["Biopsy", {"bx_exam_dt": "datetime"}],
            "ihc": ["IHC", {"paex_ihc_dt": "datetime"}],
            "sequencing": ["Sequencing", {"paex_sqnc_dt": "datetime"}],
            "ct": ["CT", {"imgx_ct_dt": "datetime"}],
            "mr": ["MR", {"imgx_mri_dt": "datetime"}],
            "petct": ["PETCT", {"imgx_petct_dt": "datetime"}],
            "us": ["US", {"imgx_us_dt": "datetime"}],
            "ercp": ["ERCP", {"ercp_act_dt": "datetime"}],
            "eus": ["EUS", {"eus_act_dt": "datetime"}],

            "ctx": ["CTx", {
                "ordr_strt_dt": "datetime",
                "ordr_end_dt": "datetime",
            }],

            "rtx": ["RTx", {
                "rad_tret_st_dt": "datetime",
                "rad_tret_end_dt": "datetime",
            }],
        }

        for table_name, (sheet_name, pgconv) in tables.items():
            data_df = excel_to_structured_dataframe(
                excel=xlsx_file,
                sheet_name=sheet_name,
                schema=PJ_SCHEMA[table_name],
            )

            if table_name in ["patient"]:
                data_df["nickname"] = [
                    "김유신",
                    "강감찬",
                    "이순신",
                    "신사임당",
                ]

            with self.conn as conn, conn.cursor() as cursor:
                cursor.execute(
                    f'''
                    SELECT {table_name}_id
                    FROM {table_name}
                    '''
                )
                recs = from_rows(cursor.fetchall())

                if not len(recs):
                    rows = execute_values(
                        cursor,
                        f'''
                        INSERT INTO
                            {table_name} AS T
                            ({to_columns(data_df.columns)})
                        VALUES %s
                        RETURNING
                            T.{table_name}_id
                        ''',
                        to_vars(data_df, pgconv), fetch=True,
                    )
                    recs = from_rows(rows)

                    self.INFO(
                        "Number of inserted items "
                        f"for {Q(table_name)} table: {len(recs)}"
                    )

    def stream(self,
               messages: Sequence[dict[str, str] | tuple[str, str]],
               **kwargs) -> Iterator[Output]:
        return iter([])

    def invoke(self, **directive) -> None:
        num_tries=directive.get("num_tries", 10)
        recovery_time=directive.get("recovery_time", 0.5)

        self._summarize(
            num_tries=num_tries,
            recovery_time=recovery_time,
        )

    def _summarize(
            self,
            num_tries: int,
            recovery_time: float
    ) -> None:
        self.INFO(f"{self.header} SUMMARIZATION proceed ....")

        self.result_df = self.patient_df.apply(
            self._summarize_foreach,
            count=itertools.count(1),
            total=self.patient_df.shape[0],
            num_tries=num_tries,
            recovery_time=recovery_time,
            axis=1
        ).pipe(
            self._prepare_htmls
        )

        self.INFO(f"{self.header} SUMMARIZATION proceed DONE")

    def _summarize_foreach(self,
                           patient: pd.Series,
                           count: Callable[..., Iterator[int]],
                           total: int,
                           num_tries: int,
                           recovery_time: float
                           ) -> pd.Series:

        question, *_ = load_text(
            self.content_dir / self.question,
            {
                "nickname": patient.nickname,
                "ct_date": patient.ct_date,
            }
        )

        for i in range(num_tries):
            try:
                timer = Timer()

                response = self.main_chain.invoke({
                    "question": question
                })

            except KeyboardInterrupt:
                raise

            except BaseException:
                self.ERROR(f"LLM({self.model[0].model_name}) internal error")
                self.ERROR(f"number of tries {i + 1}/{num_tries}")

                Timer.delay(recovery_time)
                recovery_time *= 2

                response = PJSummaryTemplate(
                    nickname=patient.nickname,
                    ct_date=patient.ct_date,
                    resch_pat_id="",
                    birth_ym="",
                    sex_cd="",
                    frst_vist_dt="",
                    dx_dt="",
                    prmr_orgn_cd="",
                    mrph_diag_cd="",
                    cancer_reg_dt="",
                    type="",
                    reason="",
                    summary="LLM 내부 오류 발생",
                )
            else:
                break
        else:
            self.ERROR(f"number of tries exceeds {num_tries}")

        timer.lap()

        entry = pd.Series({
            "nickname": response.nickname,
            "ct_date": response.ct_date,
            "resch_pat_id": response.resch_pat_id,
            "birth_ym": response.birth_ym,
            "sex_cd": response.sex_cd,
            "frst_vist_dt": response.frst_vist_dt,
            "dx_dt": response.dx_dt,
            "prmr_orgn_cd": response.prmr_orgn_cd,
            "mrph_diag_cd": response.mrph_diag_cd,
            "cancer_reg_dt": response.cancer_reg_dt,
            "type": response.type,
            "reason": response.reason,
            "summary": response.summary,
            "time": timer.seconds,
        })

        self.INFO(
            f"{self.header} SUMMARIZATION "
            f"NICKANE: {Q(patient.nickname)} TIME {timer.seconds: .1f}s "
            f"[{next(count):02d}/{total:02d}] DONE"
        )
        return entry

    def _prepare_htmls(self, summary_df: pd.DataFrame) -> pd.DataFrame:

        self.INFO(f"{self.header} SUMMARY HTML PREPARATION proceed ....")

        columns = [
            "resch_pat_id",
            "css_file",
            "html_file",
            "html_args",
        ]

        summary_df[columns] = summary_df.apply(
            self._prepare_html_foreach,
            count=itertools.count(1),
            total=summary_df.shape[0],
            axis=1
        )

        columns = [
            "suite_id",
            "case_id",
        ]
        summary_df[columns] = [self.suite_id, self.case_id]

        self.INFO(f"{self.header} SUMMARY HTML PREPARATION proceed DONE")
        return summary_df

    def _prepare_html_foreach(self,
                              summary: pd.Series,
                              count: Callable[..., Iterator[int]],
                              total: int,
                              ) -> pd.Series:
        resch_pat_id = f"{summary.resch_pat_id:010d}"

        html_args = {
            "resch_pat_id": resch_pat_id,
        } | {
            key: summary[key] for key in [
                "nickname",
                "ct_date",
                "birth_ym",
                "sex_cd",
                "frst_vist_dt",
                "dx_dt",
                "prmr_orgn_cd",
                "mrph_diag_cd",
                "cancer_reg_dt",
                "type",
                "reason",
                "summary",
            ]
        }

        css_file = self.html_dir / "styles.css"
        html_file = self.html_dir / "html_template.html"

        entry = pd.Series({
            "resch_pat_id": resch_pat_id,
            "css_file": str(css_file),
            "html_file": str(html_file),
            "html_args": html_args,
        })

        self.INFO(
            f"{self.header} REPORT HTML {Q(resch_pat_id)} "
            f"[{next(count):02d}/{total:02d}] DONE"
        )

        return entry

    @property
    def cenai_db(self) -> Postgres:
        return self._cenai_db

    @property
    def conn(self) -> psycopg2.extensions.connection:
        return self._conn

    @property
    def patient_df(self) -> pd.DataFrame:
        return self._patient_df

    @patient_df.setter
    def patient_df(self, patient_df: pd.DataFrame) -> None:
        self._patient_df = patient_df

    @property
    def question(self) -> str:
        return self._question

    @question.setter
    def question(self, question: str) -> None:
        self._question = question