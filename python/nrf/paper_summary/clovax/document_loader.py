from typing import Union

from fnmatch import fnmatch
from lxml import etree
from pathlib import Path
from xml.etree.ElementTree import parse, tostring
import zipfile

from llama_index.readers.file import HWPReader

from langchain_community.document_loaders import (
    Docx2txtLoader, PyMuPDFLoader, TextLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader
)

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class LangchainHWPLoader(BaseLoader):
    def __init__(self, file_path: str):
        self._file_path = Path(file_path)
        self._reader = HWPReader()

    def load(self) -> list[Document]:
        documents = self.reader.load_data(self._file_path)

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


def load_documents(source_file: Union[Path, str]) -> list[Document]:
    if isinstance(source_file, str):
        source_file = Path(source_file)

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


def get_document_length(source_file: Union[str, Path]) -> int:
    if isinstance(source_file, str):
        source_file = Path(source_file)

    documents = load_documents(source_file)
    return sum(len(document.page_content) for document in documents)
