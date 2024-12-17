from __future__ import annotations
from typing import Callable, Iterator, Optional, Sequence

import itertools
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from cenai_core import cenai_path, Timer
from cenai_core.dataman import load_text, Q, Struct
from cenai_core.grid import GridRunnable
from cenai_core.pandas_helper import excel_to_structured_dataframe

from cenai_core.postgres import (
    execute_script, from_rows, Postgres, to_columns, to_vars
)

from amc.pj_summarizer.pj_schema import PJ_SCHEMA


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

        self._resch_pat_ids = pd.Series()

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

    def run(self, **directive) -> None:
        self._summarize(**directive)

    def _summarize(
            self,
            num_tries: Optional[int] = None,
            recovery_time: Optional[int] = None,
            **kwargs
    ) -> None:
        self.INFO(f"{self.header} SUMMARIZATION proceed ....")

        self.result_df = self.resch_pat_ids.apply(
            self._summarize_foreach,
            count=itertools.count(1),
            total=self.resch_pat_ids.shape[0],
            num_tries=num_tries,
            recovery_time=recovery_time,

        )

        self.INFO(f"{self.header} SUMMARIZATION proceed DONE")

    def _summarize_foreach(self,
                           resch_pat_id: int,
                           count: Callable[..., Iterator[int]],
                           total: int,
                           num_tries: int,
                           recovery_time: int
                           ) -> pd.Series:

        question, *_ = load_text(
            self.content_dir / self.question,
            {"resch_pat_id": resch_pat_id}
        )

        for i in range(num_tries):
            try:
                timer = Timer()

                response = self.main_chain.invoke(question)

            except KeyboardInterrupt:
                raise

            except BaseException:
                self.ERROR(f"LLM({self.model[0].model_name}) internal error")
                self.ERROR(f"number of tries {i + 1}/{num_tries}")

                Timer.delay(recovery_time)
            else:
                break
        else:
            self.ERROR(f"number of tries exceeds {num_tries}")

            response = ""

        timer.lap()

        entry = pd.Series({
            "resch_pat_id": resch_pat_id,
            "summary": response["output"],
        })

        self.INFO(
            f"{self.header} CLASSIFY proceed DONE "
            f"[{next(count):02d}/{total:02d}] proceed DONE"
        )
        return entry

    @property
    def cenai_db(self) -> Postgres:
        return self._cenai_db

    @property
    def conn(self) -> psycopg2.extensions.connection:
        return self._conn

    @property
    def resch_pat_ids(self) -> list[int]:
        return self._resch_pat_ids

    @resch_pat_ids.setter
    def resch_pat_ids(self, resch_pat_ids: list[int]) -> None:
        self._resch_pat_ids = resch_pat_ids

    @property
    def question(self) -> str:
        return self._question

    @question.setter
    def question(self, question: str) -> None:
        self._question = question