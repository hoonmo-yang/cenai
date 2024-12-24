from typing import Sequence

from collections.abc import Iterator
import psycopg2
from psycopg2.extras import execute_values

from langchain_core.runnables.utils import Output

from cenai_core import cenai_path
from cenai_core.dataman import Q, Struct
from cenai_core.grid import GridRunnable
from cenai_core.pandas_helper import excel_to_structured_dataframe

from cenai_core.postgres import (
    execute_script, from_rows, Postgres, to_columns, to_vars
)

from pj_schema import PJ_SCHEMA


class PJChatbot(GridRunnable):
    logger_name = "cenai.amc.pj_chatbot"

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
        ddl_dir = cenai_path("python/amc/pj_chatbot/sql")
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
        return self._converse(messages)

    def _converse(self,
                  messages: Sequence[dict[str, str] | tuple[str, str]],

                  ) -> Iterator[Output]:

        self.INFO(f"{self.header} CONVERSATION proceed ....")

        stream = self.main_chain.stream({
            "messages": messages,
        })

        self.INFO(f"{self.header} CONVERSATION proceed DONE")
        return stream

    def invoke(self, **kwargs) -> None:
        pass

    @property
    def cenai_db(self) -> Postgres:
        return self._cenai_db

    @property
    def conn(self) -> psycopg2.extensions.connection:
        return self._conn
