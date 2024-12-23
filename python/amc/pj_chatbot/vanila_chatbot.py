from typing import Sequence

from operator import itemgetter

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from cenai_core.dataman import Struct
from cenai_core.langchain_helper import load_chatprompt, AgentExecutorRunnable

from pj_chatbot import PJChatbot


class VanilaChatbot(PJChatbot):
    def __init__(self,
                 models: Sequence[str],
                 agent_prompt: str,
                 metadata: Struct
                 ):

        case_suffix = "_".join([
            agent_prompt.split(".")[0],
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
                "agent_prompt",
            ]
        ] = [
            agent_prompt,
        ]

        db = SQLDatabase.from_uri(self.cenai_db.url)

        agent_chain = self._build_agent_chain(db)

        self.main_chain = self._build_main_chain(
            agent_chain=agent_chain,
            agent_prompt=agent_prompt,
        )

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
                          ) -> Runnable:
        self.INFO(f"{self.header} MAIN CHAIN prepared ....")

        agent_args, *_ = load_chatprompt(self.content_dir / agent_prompt)
        agent_prompt = ChatPromptTemplate(**agent_args)

        chain = (
            agent_prompt |
            agent_chain | {
                "content": itemgetter("output")
            } |
            self.model[0] |
            StrOutputParser()
        )

        self.INFO(f"{self.header} MAIN CHAIN prepared DONE")
        return chain
