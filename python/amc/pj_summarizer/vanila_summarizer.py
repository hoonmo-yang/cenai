from typing import Sequence
import ast
import pandas as pd

from langchain.agents import AgentExecutor, create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase 

from cenai_core.dataman import Struct

from amc.pj_summarizer import PJSummarizer


class VanilaSummarizer(PJSummarizer):
    def __init__(self,
                 models: Sequence[str],
                 question: str,
                 metadata: Struct
                ):

        case_suffix = "_".join([
            question.split(".")[0],
        ])

        super().__init__(
            models=models,
            case_suffix=case_suffix,
            metadata=metadata,
        )

        self.INFO(f"{self.header} prepared ....")

        self.metadata_df.loc[
            0,
            [
                "question",
            ]
        ] = [
            question,
        ]

        db = SQLDatabase.from_uri(self.cenai_db.url)

        self._tables = db.get_usable_table_names()
        self.question = question
        self.resch_pat_ids = self._get_resch_pat_ids(db)

        self.main_chain = self._build_agent_executor(db)

    def _get_resch_pat_ids(self, db: SQLDatabase) -> pd.Series:
        ids = db.run("SELECT resch_pat_id FROM patient")
        return pd.Series([id[0] for id in ast.literal_eval(ids)])

    def _build_agent_executor(self,
                              db: SQLDatabase,
                              ) -> AgentExecutor:
        agent_executor = create_sql_agent(
            llm=self.model[0],
            toolkit=SQLDatabaseToolkit(db=db, llm=self.model[0]),
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        agent_executor.handle_parsing_errors = True

        return agent_executor
