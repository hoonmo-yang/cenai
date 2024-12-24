from typing import Any, Sequence

from collections.abc import Iterator
from fnmatch import fnmatch
from lxml import etree
import os
from pathlib import Path
from pydantic import BaseModel, Field
from xml.etree.ElementTree import parse, tostring
import zipfile
import warnings

from llama_index.readers.file import HWPReader

from langchain.agents import AgentExecutor
from langchain_community.chat_models import ChatClovaX
from langchain_ollama import ChatOllama
from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Input, Output

from langchain_community.document_loaders import (
    Docx2txtLoader, PyMuPDFLoader, TextLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader
)

warnings.filterwarnings(
    "ignore",
    message=r"Field \"model_name\" in ClovaXEmbeddings "
            r"has conflict with protected namespace \"model_\"."
)

from langchain_community.embeddings import ClovaXEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.document_loaders import BaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from cenai_core.dataman import load_json_yaml, split_by_length


default_model_name = "llama3.2:latest"


class CenaiChatOllama(ChatOllama):
    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

        
class LangchainHelper:
    model_name = default_model_name

    @classmethod
    def bind_model(cls, model_name: str) -> None:
        cls.model_name = model_name

    @classmethod
    def load_model(cls, **kwargs) -> BaseChatModel:
        vendor = (
            "openai" if "gpt" in cls.model_name.lower() else
            "clovax" if "HCX" in cls.model_name.upper() else
            "ollama"
        )

        model = (
            ChatOpenAI(
                model=cls.model_name,
                **kwargs
            )
            if vendor == "openai" else

            ChatClovaX(
                model=cls.model_name,
                max_tokens=2500,
                **kwargs
            )
            if vendor == "clovax" else

            CenaiChatOllama(
                model=cls.model_name,
                **kwargs
            )
            # if vendor == "ollama"
        )

        return model

    @classmethod
    def load_embeddings(cls, **kwargs) -> Embeddings:
        vendor = (
            "openai" if "gpt" in cls.model_name.lower() else
            "clovax" if "HCX" in cls.model_name.upper() else
            "ollama"
        )

        embeddings = (
            OpenAIEmbeddings(**kwargs)
            if vendor == "openai" else

            ClovaXEmbeddings(
                model="clir-emb-dolphin",
                appid=os.environ["NCP_CLOVASTUDIO_APP_EMBEDDING_ID"],
                **kwargs
            )
            if vendor == "clovax" else

            HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True},
                **kwargs
            )
            # if vendor == "ollama"
        )

        return embeddings


class LineListOutputParser(BaseOutputParser[list[str]]):
    def parse(self, text: str) -> list[str]:
        lines = [
            line.strip() for line in text.strip().split("\n")
        ]

        return [line for line in lines if line]


def load_prompt(
        prompt_file: Path
    ) -> tuple[dict[str, Any], list[str]]:
    parameter = load_json_yaml(prompt_file)

    keys = [
        "input_variables",
        "template",
    ]

    return (
        {key: parameter[key] for key in keys},
        parameter.get("partial_variables", [])
    )


def load_chatprompt(prompt_file: Path) -> tuple[dict[str, Any], list[str]]:
    parameter = load_json_yaml(prompt_file)

    messages = [
        (message["role"], message["content"])
        if "role" in message else

        ("placeholder", message["placeholder"])
        if "placeholder" in message else

        ("human", message)
        for message in parameter["messages"]
    ]

    return (
        {
            "input_variables": parameter.get("input_variables", []),
            "messages": messages,
        },
        parameter.get("partial_variables", [])
    )


class ChainContext(BaseModel):
    parameter: dict[str, Any] = Field(
        default={},
    )

    handler: dict[str, BaseCallbackHandler] = Field(
        default={},
    )

    class Config:
        arbitrary_types_allowed = True


class LangchainHWPLoader(BaseLoader):
    def __init__(self, file_path: str):
        self._file_path = Path(file_path)
        self._reader = HWPReader()

    def load(self) -> list[Document]:
        documents = self._reader.load_data(self._file_path)

        documents = [
            Document(
                page_content=document.text,
                metadata=document.metadata | {"source": str(self._file_path)},
            ) for document in documents
        ]
        return documents


class HWPXLoader(BaseLoader):
    def __init__(self, file_path: str):
        self._file_path = Path(file_path)

    def load(self) -> list[Document]:
        xmls = self._read_hwpx()
        text = self._extract_text(xmls)

        documents = [
            Document(
                page_content=text,
                metadata={
                    "source": str(self._file_path),
                }
            )
        ]
        return documents

    def _read_hwpx(self) -> list[str]:
        with zipfile.ZipFile(self._file_path, "r") as z:
            names = [
                name for name in z.namelist()
                if fnmatch(name, "*/section*.xml")
            ]

            xmls = []

            for name in names:
                with z.open(name) as fin:
                    tree = parse(fin)
                    root = tree.getroot()
                    xmls.append(tostring(root, encoding="utf-8"))
        return xmls

    def _extract_text(self, xmls: list[str]) -> str:
        lines = []

        for xml in xmls:
            root = etree.fromstring(xml)
            lines.extend(self._walk(root))

        return "".join(lines)

    def _walk(self, element) -> list[str]:
        lines = []

        if element.text:
            lines.append(element.text)

        for child in element:
            lines.extend(self._walk(child))

        if element.tail:
            lines.append(element.tail)

        return lines


def load_documents(source_file: Path) -> list[Document]:
    loader = (
        LangchainHWPLoader(str(source_file))
        if source_file.suffix in [".hwp"] else

        HWPXLoader(str(source_file))
        if source_file.suffix in [".hwpx"] else

        Docx2txtLoader(str(source_file))
        if source_file.suffix in [".docx",] else

        PyMuPDFLoader(str(source_file))
        if source_file.suffix in [".pdf",] else

        TextLoader(str(source_file))
        if source_file.suffix in [".txt",] else

        UnstructuredExcelLoader(str(source_file))
        if source_file.suffix in [".xlsx",] else

        UnstructuredHTMLLoader(str(source_file))
        if source_file.suffix in [".html",] else

        UnstructuredMarkdownLoader(str(source_file))
        if source_file.suffix in [".md",] else

        UnstructuredPowerPointLoader
        if source_file.suffix in [".pptx",] else

        lambda _: []
    )

    return loader.load()


def get_document_length(source_file: Path) -> int:
    documents = load_documents(source_file)
    return sum(len(document.page_content) for document in documents)


class LineTextSplitter:
    def __init__(self, chunk_size: int):
        self._chunk_size = chunk_size

    def split_documents(self, documents: list[Document]) -> Document:
        text = "\n".join([
            document.page_content for document in documents
        ])

        page_contents = []
        lines = text.split("\n")

        for line in lines:
            page_contents.extend(split_by_length(line.strip(), self._chunk_size))

        return [
            Document(page_content=page_content)
            for page_content in page_contents
        ]


class AgentExecutorRunnable(Runnable):
    def __init__(self, agent_executor: AgentExecutor):
        self._agent_executor = agent_executor

    def invoke(self, input: Input, *args, **kwargs) -> Output:
        return self._agent_executor.invoke(input, *args, **kwargs)

    def batch(self, inputs: Sequence[Input], *args, **kwargs) -> list[Output]:
        return self._agent_executor.batch(inputs, *args, **kwargs)

    def stream(self, input: Input, *args, **kwargs) -> Iterator[Output]:
        return self._agent_executor.stream(input, *args, **kwargs)
