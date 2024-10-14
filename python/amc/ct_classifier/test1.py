import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from cenai_core import (cenai_path, LangchainHelper, load_dotenv)


class Classifier:
    def __init__(
            self,
            model_name: str,
            json_file: Path,
            keyword: str,
            test_size: float,
            random_state: int,
    ):
        LangchainHelper.bind_model(model_name)
        self._model = LangchainHelper.load_model()
        self._embeddings = LangchainHelper.load_embeddings()

        self._keyword = keyword

        self._example_df, self._train_df, self._test_df = self._load_dataset(
            json_file=json_file, 
            test_size=test_size, 
            random_state=random_state,
        )

        self._sample_df = self._create_reference()

    def _load_dataset(
        self,
        json_file: Path,
        test_size: float,
        random_state: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        example_df = pd.read_json(json_file)

        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        for _, data in example_df.groupby(self._keyword):
            train, test = train_test_split(
                data,
                test_size=test_size,
                random_state=random_state,
            )
            train_df = pd.concat([train_df, train], axis=0)
            test_df = pd.concat([test_df, test], axis=0)

        return example_df, train_df, test_df

    def _create_reference(self) -> pd.DataFrame:
        reference_df = pd.DataFrame(columns=["유형", "본문", "결론"])

        for group, content in self._train_df.groupby(self._keyword):
            report, conclusion = self._create_reference_per_group(
                group, content
            )

            row = pd.DataFrame({
                "유형": [group],
                "본문": [report],
                "결론": [conclusion],
            })

            reference_df = pd.concat(
                [reference_df, row],
                ignore_index=True,
            )

        return reference_df

    def _create_reference_per_group(
            self,
            group: str,
            content: pd.DataFrame
    ) -> str:
        examples = content.reset_index(drop=True).apply(
            lambda row: f'''
            ** 예제 {row.name + 1} **
               [[ 본문 ]]
               {row["본문"]}
               [[ 결론 ]]
               {row["결론"]}
            ''',
            axis=1,
        )       

        example_pool = "\n\n".join(examples)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 사용자가 요구한 내용 외에는 답변하지 마십시요."
            )
            ("human",
            """
            아래의 예제 목록은 본문과 결론으로 구성된 형식의 CT 판독문입니다.
            목록의 모든 예제들은 항상 <{group}> 유형에 속합니다.
            예제를 조합하여 <{group}> 유형을 대표할 수 있는
            동일한 형식의 CT 판독문 샘플을 1개 만드십시오.
            결과 샘플의 사용 용도는 CT 판독문이 **{group}** 유형에 속하는지 판별하기 위한 것입니다.

            예제:
            {example_pool}
            """
            ),
        ])

        chain = prompt | self.model | StrOutputParser()
        reference = chain.invoke({
            "group": group,
            "example_pool": example_pool,
        })

        return reference

    @staticmethod
    def _concat_group(row: pd.Series) -> str:
        return \
f'''
** 예제 {row.name + 1} **
[[ 본문 ]]
{row["본문"]}
[[ 결론 ]]
{row["결론"]}
'''

    def execute(self) -> None:
        features = self._feature_df.apply(
            self._concat_features,
            axis=1,
        )

        feature_pool = "\n\n".join(features)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 CT 판독문의 유형과 유형을 판별한 근거를 답변해야 합니다."
            ),
            ("human",
'''
입력한 CT 판독문을 아래 유형 특징을 근거로 분류하여 가장 가까운 유형으로 선택해야 합니다.

** 입력한 CT 판독문: **
{user_input}

** 유형별 특징: **
{feature_pool}
'''
             )
        ])

        chain = prompt | self.model | StrOutputParser()

        self.scoreboard_df = self._test_df.apply(
            self._classify_group,
            feature_pool=feature_pool,
            chain=chain,
            axis=1,
        )

    def _concat_features(self, row: pd.Series) -> str:
        return \
f'''
** 특징 {row.name + 1}: **
[[ 유형 ]]
{row["유형"]}
[[ 요약 ]]
{row["요약"]}
'''

    def _classify_group(
            self,
            row: pd.Series,
            feature_pool: str,
            chain: Runnable
    ) -> pd.Series:
        user_input = \
f"""
[[ 본문 ]]
{row["본문"]}
[[ 결론 ]]
{row["결론"]}
"""
        result = chain.invoke({
            "user_input": user_input,
            "feature_pool": feature_pool,
        })

        return pd.Series({
            "유형": row["유형"],
            "결과": result,
        })

    def evaluate(self) -> None:
        pass

    @property
    def model(self) -> BaseChatModel:
        return self._model

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings


def main() -> None:
    load_dotenv()

    model_name = "llama3.1:latest"
    model_name = "llama3.1:70b"
    model_name = "gpt-4o-mini"
    model_name = "gpt-3.5-turbo"

    json_file = cenai_path("data/ct_report.json")

    classifier = Classifier(
        model_name=model_name,
        json_file=json_file,
        keyword = "유형",
        test_size=0.2,
        random_state=44,
    )

    classifier.execute()
    classifier.evaluate()


if __name__ == "__main__":
    main()
