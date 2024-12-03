from cenai_core import cenai_path
from cenai_core.langchain_helper import load_documents

dir_ = cenai_path("data/nrf/paper-summarizer/corpus/sample")

for file_ in dir_.glob("*.hwp"):
    print(file_)
    documents = load_documents(file_)
    content = "\n".join([document.page_content for document in documents])
