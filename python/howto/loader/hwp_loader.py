from llama_index.readers.file import HWPReader
from cenai_core import cenai_path
from pathlib import Path
from langchain_core.documents import Document

hwp_file = cenai_path("data/nrf/paper-summarizer/corpus/IR_00000000012159757_20240320093725_965639.hwp")

class LangChainHWPReader:
    def __init__(self, hwp_file:str):
        self.hwp_file = Path(hwp_file)
        self.reader = HWPReader()

    def load(self) -> list[Document]:
        documents = self.reader.load_data(self.hwp_file)

        return [
            Document(
                page_content=document.text,
                metadata=document.metadata,
            ) for document in documents
        ]


loader = LangChainHWPReader(hwp_file)
documents = loader.load()

print(documents)