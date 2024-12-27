from typing import Sequence

import ast
from operator import itemgetter
import pandas as pd

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.output_parsers import PydanticOutputParser

from cenai_core.dataman import Struct
from cenai_core.langchain_helper import load_prompt, AgentExecutorRunnable

from pj_summarizer import PJSummarizer
from pj_template import PJSummaryTemplate


class VanilaSummarizer(PJSummarizer):
    def __init__(self,
                 models: Sequence[str],
                 agent_prompt: str,
                 summarize_prompt: str,
                 question: str,
                 metadata: Struct
                ):

        case_suffix = "_".join([
            agent_prompt.split(".")[0],
            summarize_prompt.split(".")[0],
            question.split(".")[0],
        ])

        super().__init__(
            models=models,
            case_suffix=case_suffix,
            metadata=metadata,
        )

        self.INFO(f"{self.header} prepared ....")

        self.question = question

        self.metadata_df.loc[
            0,
            [
                "agent_prompt",
                "summarize_prompt",
                "question",
            ]
        ] = [
            agent_prompt,
            summarize_prompt,
            question,
        ]

        db = SQLDatabase.from_uri(self.cenai_db.url)

        self.question = question
        self.patient_df = self._get_patients()

        agent_chain = self._build_agent_chain(db)

        self.main_chain = self._build_main_chain(
            agent_chain=agent_chain,
            agent_prompt=agent_prompt,
            summarize_prompt=summarize_prompt,
        )

    def _get_patients(self) -> pd.DataFrame:
        patient_df = pd.DataFrame({
            "nickname": [
                "김유신",
                "강감찬",
                "이순신",
                "신사임당",
            ],
            "ct_date": [
                "2018-01-06",
                "2017-08-06",
                "2017-08-03",
                "2011-05-02",
            ],
        })

        return patient_df

    def _build_agent_chain(self, db: SQLDatabase) -> Runnable:
        self.INFO(f"{self.header} AGENT CHAIN prepared ....")

        agent = create_sql_agent(
            llm=self.model[0],
            toolkit=SQLDatabaseToolkit(db=db, llm=self.model[0]),
            verbose=True,
            agent_type="tool-calling",
            max_iterations=30,
        )

        agent_chain = AgentExecutorRunnable(agent)

        self.INFO(f"{self.header} AGENT CHAIN prepared DONE")
        return agent_chain

    def _build_main_chain(self, 
                          agent_chain: Runnable,
                          agent_prompt: str,
                          summarize_prompt: str
                          ) -> Runnable:
        self.INFO(f"{self.header} MAIN CHAIN prepared ....")

        agent_args, *_ = load_prompt(self.content_dir / agent_prompt)
        agent_prompt = PromptTemplate(**agent_args)

        parser = PydanticOutputParser(
            pydantic_object=PJSummaryTemplate,
        )

        summarize_args, partials = load_prompt(
            self.content_dir / summarize_prompt
        )

        full_summarize_args = summarize_args | {
            "partial_variables": {
                partials[0]: parser.get_format_instructions(),
            },
        }

        summarize_prompt = PromptTemplate(**full_summarize_args)

        chain = (
            agent_prompt |
            agent_chain | {
                "content": itemgetter("output")
            } |
            summarize_prompt |
            self.model[0] |
            parser
        )

        self.INFO(f"{self.header} MAIN CHAIN prepared DONE")
        return chain
