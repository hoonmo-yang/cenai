from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from cenai_core.dataman import Struct
from cenai_core.langchain_helper import load_chatprompt

from nrf.paper_summarizer import PaperSummarizer

class StuffSummarizer(PaperSummarizer):
    def __init__(self,
                 model: str,
                 chunk_size: int,
                 chunk_overlap: int,
                 max_tokens: int,
                 num_keywords: int,
                 summarize_prompt: str,
                 keyword_prompt: str,
                 metadata: Struct
                 ):

        case_suffix = "_".join([
            summarize_prompt.split(".")[0],
            keyword_prompt.split(".")[0],
        ])

        super().__init__(
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_keywords=num_keywords,
            max_tokens=max_tokens,
            case_suffix=case_suffix,
            metadata=metadata,
        )

        self.INFO(f"{self.header} prepared ....")

        self.metadata_df.loc[
            0, 
            [
                "summarize_prompt",
                "keyword_prompt",
            ]] = [
                summarize_prompt,
                keyword_prompt,
            ]

        self.summarizer_chain = self._create_summarizer_chain(
            summarize_prompt=summarize_prompt,
        )

        self.keyword_chain = self._create_keyword_chain(
            keyword_prompt=keyword_prompt,
        )

        self.INFO(f"{self.header} prepared DONE")

    
    def _create_summarizer_chain(self,
                                 summarize_prompt: str
                                 ) -> Runnable:
        self.INFO(f"{self.header} SUMMARIZE CHAIN prepared ....")

        prompt_args = load_chatprompt(self.content_dir / summarize_prompt)
        prompt = ChatPromptTemplate(**prompt_args)

        chain = (
            prompt |
            self.model |
            StrOutputParser()
        )

        self.INFO(f"{self.header} SUMMARIZE CHAIN prepared DONE")
        return chain


    def _create_keyword_chain(self,
                              keyword_prompt: str
                              ) -> Runnable:
        self.INFO(f"{self.header} KEYWORD CHAIN prepared ....")

        prompt_args = load_chatprompt(self.content_dir / keyword_prompt)
        prompt = ChatPromptTemplate(**prompt_args)

        chain = (
            prompt |
            self.model |
            StrOutputParser()
        )

        self.INFO(f"{self.header} KEYWORD CHAIN prepared DONE")
        return chain
