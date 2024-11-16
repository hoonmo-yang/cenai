from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd

from cenai_core.grid_ import GridRunnable, GridChainContext
from cenai_core.dataman import Q, Struct


class PaperSummarizer(GridRunnable, ABC):
    logger_name = "cenai.nrf.paper_classifier"

    def __init__(self,
                 metadata: Struct,
                 module_suffix: str
                 ):
        super().__init__(
            metadata=metadata,
            dataset_suffix="",
            module_suffix=module_suffix,
        )

        self._classifier_chain = RunnableLambda(

        )

    def run(self, directive: dict[str, Any]) -> None:
        self.summarize()

    def summarize(self) -> None:
        self.INFO(f"{self.header} SUMMARIZE proceed ....")


        self.INFO(f"{self.header} SUMMARIZE proceed DONE")
