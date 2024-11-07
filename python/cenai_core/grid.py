from __future__ import annotations

from cenai_core.logger import Logger

from cenai_core.langchain_helper import LangchainHelper, load_dotenv



class GridWorkflow(Logger):
    logger_name = "cenai.grid_workflow"

    def __init__(self,
                 **kwargs
                 ):
        super().__init__()

    def execute(self) -> None:
        pass

