from typing import Any

from pathlib import Path
from pydantic import BaseModel, Field

from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import (ChatOpenAI, OpenAIEmbeddings)

from cenai_core.hyperclovax import HyperCLOVAXChatModel, HyperCLOVAXEmbeddings
from cenai_core.dataman import load_json_yaml

default_model_name = "llama3.1:latest"


class LangchainHelper:
    model_name = default_model_name

    @classmethod
    def bind_model(cls, model_name: str) -> None:
        cls.model_name = model_name

    @classmethod
    def load_model(cls, **kwargs) -> BaseChatModel:
        caller = (
            ChatOpenAI if "gpt" in cls.model_name.lower() else
            HyperCLOVAXChatModel if "hyperclovax" else
            ChatOllama
        )

        return caller(
            model=cls.model_name, **kwargs
        )

    @classmethod
    def load_embeddings(cls, **kwargs) -> Embeddings:
        embeddings = (
            OpenAIEmbeddings(**kwargs)
            if "gpt" in cls.model_name.lower() else

            HyperCLOVAXEmbeddings()
            if "hyperclovax" in cls.model_name.lower() else
            
            HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True},
                **kwargs
            )
        )

        return embeddings


class LineListOutputParser(BaseOutputParser[list[str]]):
    def parse(self, text: str) -> list[str]:
        lines = [
            line.strip() for line in text.strip().split("\n")
        ]

        return [line for line in lines if line]


def load_chatprompt(prompt_file: Path) -> dict[str, Any]:
    parameter = load_json_yaml(prompt_file)

    messages = parameter.pop("messages", [])

    parameter["messages"] = [
        (message["role"], message["content"])
        for message in messages
    ]

    return parameter


class ChainContext(BaseModel):
    parameter: dict[str, Any] = Field(
        default={},
    )

    handler: dict[str, BaseCallbackHandler] = Field(
        default={},
    )

    class Config:
        arbitrary_types_allowed = True
