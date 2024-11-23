from typing import Annotated, Literal, TypedDict
import operator

from langchain.chains.combine_documents.reduce import (
    collapse_docs,
    split_list_of_docs
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph


from cenai_core import cenai_path, LangchainHelper, load_dotenv, Timer

model_name = "hcx-003"

load_dotenv()
LangchainHelper.bind_model(model_name)
model = LangchainHelper.load_model()
embeddings = LangchainHelper.load_embeddings()

paper_file = cenai_path("data/nrf/paper-summarizer/corpus/nrf-paper00.pdf")

loader = PyPDFLoader(str(paper_file))
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

split_documents = splitter.split_documents(documents)[0:3]

map_prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        """
        다음 내용의 논문을 요약하시오:

        {context}
        """
    )
])

map_chain = map_prompt | model | StrOutputParser()

reduce_template = """
다음은 문서 요약 목록입니다.

{documents}

이 문서 요약 목록의 내용을 가지고 최종적인 요약을 만들기 바랍니다.
"""

reduce_prompt = ChatPromptTemplate([
    ("human", reduce_template)
])

reduce_chain = reduce_prompt | model | StrOutputParser()

token_max = 500

def length_function(documents: list[Document]) -> int:
    return sum(
        model.get_num_tokens(document.page_content)
        for document in documents
    )


class OverallState(TypedDict):
    contents: list[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: list[Document]
    final_summary: str


class SummaryState(TypedDict):
    content: str


def generate_summary(state: SummaryState):
    response = map_chain.invoke(state["content"])
    Timer.delay(1)
    return {"summaries": [response]}


def map_summaries(state: OverallState):
    return [
        Send("generate_summary", {"content": content})
        for content in state["contents"]
    ]


def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [
            Document(summary) for summary in state["summaries"]
        ]
    }


def collapse_summaries(state: OverallState):
    all_documents = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    results = []
    for documents in all_documents:
        results.append(
            collapse_docs(documents, reduce_chain.invoke)
        )
        Timer.delay(1)

    return {"collapsed_summaries": results}


def should_collapse(
    state: OverallState
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"


def generate_final_summary(state: OverallState):
    response = reduce_chain.invoke(state["collapsed_summaries"])
    Timer.delay(1)
    return {"final_summary": response}


graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_final_summary", generate_final_summary)

graph.add_conditional_edges(
    START, map_summaries,
    ["generate_summary"],
)

graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)

app = graph.compile()

contents = [
    document.page_content for document in split_documents
]

response = app.invoke(
    {"contents": contents},
    {"recursion_limit": 10},
)

print(response)
