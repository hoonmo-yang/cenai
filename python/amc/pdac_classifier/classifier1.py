import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from amc.ct_classifier.classifier import BaseClassifier


class Report(BaseModel):
    body: str = Field(description="CT 판독문의 본문입니다.")
    conclusion: str = Field(description="CT 판독문의 결론입니다.")
    why: str = Field(description="CT 판독문의 유형에 대한 근거입니다.")


class Feature(BaseModel):
    body: str = Field(description="CT 판독문의 본문입니다.")
    conclusion: str = Field(description="CT 판독문의 결론입니다.")
    group: str = Field(description="CT 판독문의 유형입니다.")
    why: str = Field(description="CT 판독문의 유형에 대한 근거입니다.")


class Classification(BaseModel):
    group: str = Field()
    why: str = Field()


class Classifier1(BaseClassifier):
    output_class_dir = BaseClassifier.output_dir / __qualname__.lower()

    def __init__(self,
                 model: BaseChatModel,
                 embeddings: Embeddings,
                 dataset_name: str,
                 test_size: float,
                 random_state: int
                 ):
        super().__init__(
            model, embeddings,
            dataset_name, test_size, random_state
        )

        self._feature_df = self._extract_features()

    def _extract_features(self) -> pd.DataFrame:
        feature_df = self._retrieve_cache()

        if not feature_df.empty:
            return feature_df

        feature_df = pd.DataFrame(
            columns=["유형", "본문", "결론", "근거"]
        )

        example_df = self._example_df["train"]

        for group, dataframe in example_df.groupby("유형"):
            body, conclusion, why = self._extract_feature(
                group, dataframe
            )

            row = pd.DataFrame({
                "유형": [group],
                "본문": [body],
                "결론": [conclusion],
                "근거": [why],
            })

            feature_df = pd.concat(
                [feature_df, row],
                ignore_index=True
            )

        feature_df = feature_df.apply(
            self._refine_feature,
            feature_df=feature_df,
            axis=1,
        )

        self._save_cache(feature_df)
        return feature_df

    def _refine_feature(self,
                        feature: pd.Series,
                        feature_df: pd.DataFrame
                        ) -> pd.Series:

        others = feature_df[
            feature_df["유형"] != feature["유형"]
        ].apply(
            lambda row: f"""
            ### {row.name + 1}번째 유형:
            - **본문**: {row['본문']}
            - **결론**: {row['결론']}
            - **유형**: {row['유형']}
            - **근거**: {row['근거']}
            """,
            axis=1,
        ).tolist()

        other_parts = "\n\n".join(others)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """ 당신은 총 {n}개의 췌장암 CT 판독문의 유형을 고려해야 합니다.
                이중 하나는 사용자의 요구대로 내용을 정제해야 할 유형이며
                나머지 {n} - 1개의 유형은 입력된 유형을 정제하기 위해 비교할 다른 유형들입니다. """
            ),
            (
                "human",
                """
                정제하기 위해 입력된 유형을 제외한 나머지 유형은 다음과 같습니다.

                {other_parts}

                ### 입력 유형
                입력된 유형은 아래와 같습니다.
                입력된 유형을 다음의 작업 단계를 거쳐서 정제하기 바랍니다. 내용만 바뀌게 되며
                항목은 그대로 유지되어야 합니다.

                작업 단계:
                1. 다른 유형과 공통되는 내용은 삭제하십시오.
                2. 이 유형이 다른 유형과 구분되는 고유한 차이점과 중요한 요소만 남기십시오.
                3. 특히 중요한 차이점이 되는 정보에 집중하십시오.

                - **본문**: {본문}
                - **결론**: {결론}
                - **유형**: {유형}
                - **근거**: {근거}

                결과: 각 유형 간의 차이점과 고유한 내용만 남긴 상태로 출력해 주십시오.
                """
            ),
        ])

        chain = prompt | self.model.with_structured_output(Feature)

        response = chain.invoke({
            "n": feature_df.shape[0],
            "other_parts": other_parts,
            "본문": feature["본문"],
            "결론": feature["결론"],
            "유형": feature["유형"],
            "근거": feature["근거"],
        })

        return pd.Series({
            "본문": response.body,
            "결론": response.conclusion,
            "유형": response.group,
            "근거": response.why,
        })

    def _save_cache(self, source_df: pd.DataFrame) -> None:
        feature_dir = self.output_class_dir / "feature"
        feature_dir.mkdir(parents=True, exist_ok=True)
        target_json = feature_dir / f"{self._body}.json"
        source_df.to_json(target_json)

    def _retrieve_cache(self) -> pd.DataFrame:
        feature_dir = self.output_class_dir / "feature"
        source_json = feature_dir / f"{self._body}.json"

        return (
            pd.read_json(source_json) if source_json.is_file() else
            pd.DataFrame()
        )

    def _extract_feature(self,
                         group: str,
                         example_df: pd.DataFrame
                         ) -> tuple[str, str, str]:
        examples = example_df.reset_index(drop=True).apply(
            lambda row: (f"### 예제 {row.name + 1}:\n"
                         f"**body**: {row['본문']}\n"
                         f"**conclusion**: \n{row['결론']}\n\n"),
            axis=1,
        )

        example_parts = "\n\n".join(examples)
        example_body = example_df.iloc[0]["본문"]
        example_conclusion = example_df.iloc[0]["결론"]

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """ 당신은 주어진 췌장암 검사를 위한 CT 판독문들의 내용을 분석하여 하나의 유형으로
                묶을 수 있는 공통된 내용을 뽑아서 표준 샘플을 생성하는데 특화된 언어 모델입니다.
                CT 판독문의 유형은 췌장암의 단계별 진행 정도와 CT 촬영의 목적에 따라 분류됩니다.
                또한, 생성된 표준 샘플이 왜 CT 판독문들의 유형을 잘 나타내는지 근거를 설명하는
                역할을 합니다. """
             ),
            (
                "human",
                """ 다음 텍스트들은 CT 판독문들이며, 각 판독문은 'body'와 'conclusion' 항목으로
                구성되어 있습니다. 이들은 모두 동일한 유형입니다.
                예제로 주어진 판독문들의 특징을 총괄하는 표준 샘플을 하나 만들어 주세요.
                만들어진 표준 샘플은 이후에 들어올 새로운 CT 판독문(구조는 동일하나 내용이 다름)이
                이 유형에 해당하는지 검사하는 유형 분류에 사용할 것입니다.
                그리고 만들어진 표준 샘플이 어떻게 해당 유형을 가장 잘 판별할 수 있는 표준이 되는지
                근거를 함께 제시해 주세요.

                다음 사항을 참고해 주세요:
                1. 'body'와 'conclusion' 항목을 대표할 수 있는 공통된 내용이 모두 포함되어야 합니다.
                2. CT 판독문의 유형은 췌장암의 단계별 진행 정도와 CT를 촬영하게 된 목적에 따라 분류됩니다.
                   표준 샘플은 직접적인 요약이 아니라 유형을 잘 나타내는 예제로 만들어져야 합니다.
                3. 새로운 판독문이 입력되었을 때, 이 표준 샘플과 비교하여 쉽게 유형을 판단할 수 있도록
                   명확하게 작성해 주세요.\n"
                4. 표준 샘플이 왜 이 유형의 CT 판독문을 잘 대표하는지 근거도 설명해 주세요.
                5. 근거의 변수 이름은 'why'입니다.

                ## CT 판독문 예제:
                {example_parts} """
            ),
        ])

        chain = prompt | self.model.with_structured_output(Report)

        report = chain.invoke({
            "group": group,
            "example_parts": example_parts,
            "example_body": example_body,
            "example_conclusion": example_conclusion,
        })

        return (report.body, report.conclusion, report.why)

    def classify(self) -> None:
        examples = self._feature_df.apply(
            lambda row: HumanMessage(
                content=f"""
                ### {row.name + 1}번 예제:
                - **본문**: {row['본문']}
                - **결론**: {row['결론']}
                - **유형**: {row['유형']}
                - **근거**: 이 판독문은 {row['유형']}에 해당하며, 그 이유는 {row['근거']}
                """
            ),
            axis=1
        ).tolist()

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """ 당신은 의학적 CT 판독문을 분석하여 그 판독문이 어떤 유형에 속하는지 판단하고,
                그 근거를 설명하는 역할을 맡고 있습니다. CT 판독문은 '본문'과 '결론'으로 나뉘며,
                주어진 유형을 바탕으로 새로운 CT 판독문의 유형을 결정하세요.
                또한, 결정한 유형에 대한 근거를 명확하게 제시해 주세요. """
            ),
            (
                "human",
                """ 다음은 각각 본문, 결론, 유형, 근거를 포함한 예제들입니다.
                """
            ),
            MessagesPlaceholder(variable_name="examples"),
            (
                "human",
                """ ### 입력:
                다음 입력된 CT 판독문의 유형을 결정하고, 그렇게 결정한 이유를 설명하세요:

                **본문**: {본문}
                **결론**: {결론}

                다음 사항을 참고해 주세요:
                1. 결정할 유형은 반드시 제공된 유형들 중 하나야 합니다.
                2. 설명할 유형 결정 이유(즉 근거)는 **본문**과 동일한 언어를 사용하세요.
                   단 전문 용어는 원래의 언어를 그대로 유지하세요.
                3. 유형 결정 이유(즉 근거)의 변수 이름은 'why'입니다.
                """
            ),
        ])

        self._experiment_df = self._example_df["test"].apply(
            self._classify,
            prompt=prompt,
            examples=examples,
            axis=1,
        )
        self._experiment_df = self._experiment_df.reset_index(drop=True)

    def _classify(self,
                  row: pd.Series,
                  prompt: ChatPromptTemplate,
                  examples: list[HumanMessage]
                  ) -> pd.Series:

        chain = prompt | self.model.with_structured_output(Classification)

        answer = chain.invoke({
            "본문": row["본문"],
            "결론": row["결론"],
            "examples": examples,
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
        report_file = report_dir / f"{self._body}.txt"

        total = self._experiment_df.shape[0]
        hit = (self._experiment_df["정답"] == self._experiment_df["예측"]).sum()
        hit_ratio = (hit / total) * 100

        deviate_df = self._experiment_df[
            self._experiment_df["정답"] != self._experiment_df["예측"]
        ]

        with report_file.open("wt", encoding="utf-8") as fout:
            fout.write(
                "\n## EXPERIMENT RESULT\n\n"
                f"HIT RATIO:{hit_ratio}% HIT:{hit} TOTAL:{total}\n\n"
            )

            deviate_df.apply(
                lambda row: fout.write(
                    f"#### 테스트 {row.name + 1}:\n"
                    f"-**정답**: {row['정답']}\n"
                    f"-**예측**: {row['예측']}\n"
                    f"-**근거**:\n{row['근거']}\n"
                    f"-**원유형**: {row['원유형']}\n"
                    f"-**본문**:\n{row['본문']}\n"
                    f"-**결론**:\n{row['결론']}\n\n"
                ),
                axis=1,
            )

            fout.write("\n\n## FEATURES FOR GROUPS\n")

            self._feature_df.apply(
                lambda row: fout.write(
                    f"#### 샘플 {row.name + 1}:\n"
                    f"-**유형**: {row['유형']}\n"
                    f"-**본문**:\n{row['본문']}\n"
                    f"-**결론**:\n{row['결론']}\n"
                    f"-**근거**:\n{row['근거']}\n\n"
                ),
                axis=1,
            )


def main() -> None:
    dataset_stem = "pdac-report"
    test_sizes = [0.2]
    num_tries = 4
    topks = [0]

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
        classifier_class=Classifier1,
    )


if __name__ == "__main__":
    main()
