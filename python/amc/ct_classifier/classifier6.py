from operator import itemgetter
import pandas as pd
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_community.document_transformers import LongContextReorder
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from langchain_core.runnables import (
    Runnable, RunnableLambda, RunnablePassthrough
)

from langchain.retrievers import (
    BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
)

from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cenai_core.dataman import clean_text

from amc.ct_classifier.classifier import BaseClassifier


class Classification(BaseModel):
    group: str = Field(description="CT 판독문의 유형입니다.")
    why: str = Field(description="CT 판독문의 유형에 대한 근거입니다")


class Classifier6(BaseClassifier):
    output_class_dir = BaseClassifier.output_dir / __qualname__.lower()

    def __init__(self,
                 model_name: str,
                 model: BaseChatModel,
                 embeddings: Embeddings,
                 dataset_name: str,
                 test_size: float,
                 random_state: int,
                 topk: int
                 ):
        super().__init__(
            model_name, model, embeddings,
            dataset_name, test_size, random_state, topk
        )

        self._retriever = self._create_retriever()

    def _create_retriever(self) -> BaseRetriever:
        documents = [Document(page_content=self._generate_formatted_text())]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        splits = splitter.split_documents(documents)

        bm25_retriever = BM25Retriever.from_documents(
            documents=splits,
        )
        bm25_retriever.k = 1

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
        )

        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self._topk},
        )

        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.7, 0.3],
        )

        compressor = LLMChainExtractor.from_llm(self.model)
        compressor_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever,
        )

        return compressor_retriever

    def _generate_formatted_text(self):
        example_df = self._example_df["train"]
        texts = example_df.apply(
            lambda row: (
                f"### CT 판독문 {row.name + 1}\n"
                f"-## 유형: {row['유형']}\n"
                f"-## 본문:\n{row['본문']}\n"
                f"-## 결론:\n{row['결론']}"
            ),
            axis=1
        ).tolist()

        return "\n\n".join(texts)

    def classify(self) -> None:
        group_names = "\n".join(
            pd.Series(self._example_df["train"]["유형"].unique()).apply(
                lambda col: f"- {col}"
            )
        )

        group_file = self.data_dir / f"{self._dataset_name}_group.txt"
        group_description = group_file.read_text(encoding="utf-8")

        system_prompt = """
        당신은 췌장암 환자의 CT 판독문을 분석하여 그 판독문이 어떤 유형에 속하는지 판단하고,
        그 근거를 설명하는 역할을 맡고 있습니다. CT 판독문은 '본문'과 '결론'으로 나뉩니다.
        당신은 아래의 유형의 설명과 검색된 문맥을 사용하여 새로운 CT 판독문의 유형을 결정하세요.
        또한, 결정한 유형에 대한 근거를 명확하게 제시해 주세요.
        대답할 유형의 명칭은 반드시 아래 제시한 명칭 중 하나와 일치해야 합니다.

        ### 유형의 명칭:
        {group_names}

        ### 유형의 설명:
        {group_description}

        ### 문맥:
        {context}
        """

        human_prompt = """
        ### 입력:
        {input}

        다음 사항을 참고해 주세요:
        1. 결정할 유형은 반드시 제공된 문맥에서 발견하는 유형들 중 하나야 합니다.
        2. 설명할 유형 결정 이유(즉 근거)는 **본문**에서 사용한 언어를 사용해야 합니다.
           단 전문 용어는 입력에서 사용한 원 언어를 그대로 유지하세요.
        3. 유형 결정 이유(즉 근거)의 변수 이름은 'why'입니다.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])

        chain = (
            RunnablePassthrough().assign(
                context=itemgetter("input") |
                self._retriever |
                self._reorder_documents |
                self._concat_documents
            ) |
            prompt |
            self.model.with_structured_output(Classification)
        )

        self._experiment_df = self._example_df["test"].apply(
            self._classify,
            chain=chain,
            group_names=group_names,
            group_description=group_description,
            axis=1,
        )

        self._experiment_df = self._experiment_df.reset_index(drop=True)

    @staticmethod
    def _concat_documents(documents: list[Document]) -> str:
        return "\n\n".join(document.page_content for document in documents)

    @staticmethod
    def _reorder_documents(documents: list[Document]) -> list[Document]:
        reordering = LongContextReorder()
        return reordering.transform_documents(documents)

    def _classify(self,
                  row: pd.Series,
                  chain: Runnable,
                  group_names: str,
                  group_description: str
                 ) -> pd.Series:
        answer = chain.invoke({
            "input":
                "다음 입력된 CT 판독문의 유형을 결정하고,\n"
                "그렇게 결정한 이유를 설명하세요:\n"
                f"본문:{row['본문']}\n결론:{row['결론']}\n",
            "group_names": group_names,
            "group_description": group_description,
        })

        return pd.Series({
            "원유형": row["원유형"],
            "본문": row["본문"],
            "결론": row["결론"],
            "정답": row["유형"].strip().lower(),
            "예측": answer.group.strip().lower(),
            "근거": answer.why,
        })

    def create_report(self) -> None:
        report_dir = self.output_class_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"{self._body}_k{self._topk:02d}.txt"

        total = self._experiment_df.shape[0]

        hit_ok = self._experiment_df.apply(
            lambda row:
                clean_text(row["정답"]).lower() == clean_text(row["예측"]).lower(),
            axis=1,
        )

        hit = hit_ok.sum()
        hit_ratio = (hit / total) * 100

        deviate_df = self._experiment_df[~hit_ok]

        with report_file.open("wt", encoding="utf-8") as fout:
            fout.write(
                "\n## EXPERIMENT RESULT\n\n"
                f"HIT RATIO:{hit_ratio:0.2f}% HIT:{hit} TOTAL:{total}\n\n"
            )

            deviate_df.apply(
                lambda row: fout.write(
                    f"#### 테스트 {row.name + 1}:\n"
                    f"-**정답**: {row['정답']}\n"
                    f"-**예측**: {row['예측']}\n\n"
                    f"-**근거**:{row['근거']}\n\n"
                    f"-**원유형**: {row['원유형']}\n"
                    f"-**본문**:\n{row['본문']}\n"
                    f"-**결론**:\n{row['결론']}\n\n\n"
                ),
                axis=1,
            )


def main() -> None:
    dataset_stem = "pdac-report"
    test_sizes=[0.2]
    num_tries = 8
    topks = [8, 10]

    model_names = ["llama3.1:latest"]
    model_names = ["llama3.1:70b"]
    model_names = ["gpt-4o-mini"]
    model_names = ["gpt-3.5-turbo"]
    model_names = ["gpt-4o"]
    model_names = ["gpt-4o", "gpt-3.5-turbo"]

    dataset_names = BaseClassifier.create_dataset(
        dataset_stem=dataset_stem,
        test_sizes=test_sizes,
        num_tries=num_tries,
    )


    BaseClassifier.evaluate(
        model_names=model_names,
        dataset_names=dataset_names,
        test_sizes=test_sizes,
        num_tries=num_tries,
        topks=topks,
        classifier_class=Classifier6,
    )


if __name__ == "__main__":
    main()