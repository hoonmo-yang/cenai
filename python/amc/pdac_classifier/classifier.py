from __future__ import annotations
from typing import Any, Optional, Union

from abc import ABC, abstractmethod
import importlib
from itertools import product
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from langchain_community.document_transformers import LongContextReorder
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain.schema import Document

from cenai_core import cenai_path, LangchainHelper, load_dotenv, Logger, Timer
from cenai_core.dataman import concat_texts, load_json_yaml, match_text, Q
from cenai_core.pandas_helper import to_json


class PDACClassifyResult(BaseModel):
    category: str = Field(
        description="AI 분류기가 예측한 CT 판독문의 유형",
    )

    reason: str = Field(
        description="AI 분류기가 CT 판독문의 유형을 예측한 근거",
    )


class PDACClassifier(Logger, ABC):
    logger_name = "amc.pdac_classifier"

    output_dir = cenai_path("output") / "amc"
    data_dir = cenai_path("data") / "amc"

    @classmethod
    def load_grid(cls,
                  grid: Path
                  ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        deserialized = load_json_yaml(grid)
        return (
            deserialized.get("dataset"),
            deserialized.get("hparam"),
            deserialized.get("aux"),
        )

    @classmethod
    def create_datasets(cls, dataset_info: dict[str,Any]) -> list[str]:
        dataset_dir = cls.output_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        datasets = []

        for values in product(*dataset_info.values()):
            arg_dict = dict(zip(dataset_info.keys(), values))
            datasets.extend(cls._create_datasets(dataset_dir, **arg_dict))

        return datasets

    @classmethod
    def _create_datasets(cls,
                         dataset_dir: Path,
                         dataset_prefix: str,
                         test_size: float,
                         pick: Union[int,list[int]]
                         ) -> list[str]:

        json_file = cls.data_dir / f"{dataset_prefix}.json"
        source_df = pd.read_json(json_file)

        pick = [pick] if isinstance(pick, int) else pick

        test = int(test_size * 10)
        train = 10 - test
        dataset_base = f"{dataset_prefix}_{train}-{test}"

        datasets = []

        for i in range(*pick):
            dataset = f"{dataset_base}_{i:02d}"

            target_df = {key: pd.DataFrame() for key in ["train", "test"]}
            
            for _, dataframe in source_df.groupby("유형"):
                train_df, test_df = train_test_split(
                    dataframe,
                    test_size=test_size,
                    random_state=i,
                )

                target_df["train"] = pd.concat(
                    [target_df["train"], train_df], axis=0
                )

                target_df["test"] = pd.concat(
                    [target_df["test"], test_df], axis=0
                )

            for tag in ["train", "test"]:
                dataframe = target_df[tag].reset_index(drop=True)
                target_dir = dataset_dir / tag
                target_dir.mkdir(parents=True, exist_ok=True)
                target_json = target_dir / f"{dataset}.json"
                dataframe.to_json(target_json)

            datasets.append(dataset)

        return datasets

    @classmethod
    def evaluate(cls,
                 datasets: list[str],
                 spec_info: dict[str, Any],
                 aux_info: dict[str, Any]
                 ) -> None:
        load_dotenv(aux_info.get("langsmith"))

        for values in product(*spec_info.values()):
            arg_dict = dict(zip(spec_info.keys(), values))

            for dataset in datasets:
                classifier = cls._get_classifier(
                    dataset=dataset,
                    **arg_dict
                )

                if classifier is None:
                    if "algorithm" in arg_dict:
                        cls.ERROR(
                            f"invalid algorithm: {Q(arg_dict['algorithm'])} SKIP"
                        )
                    else:
                        cls.ERROR("algorithm missing SKIP")

                    continue

                classifier.classify()
                classifier.save_data()

    @classmethod
    def _get_classifier(cls,
                        dataset: str,
                        **kwargs
                        ) -> Optional[PDACClassifier]:
        if "algorithm" not in kwargs:
            return None

        algorithm = kwargs["algorithm"]
        module = importlib.import_module(algorithm)

        try:
            Class = getattr(module, algorithm.capitalize())

        except (ModuleNotFoundError, AttributeError):
            return None

        return Class(
            dataset=dataset,
            **kwargs
        )

    def __init__(self,
                 dataset: str,
                 model_name: str,
                 algorithm: str,
                 hparam: str
                 ):
        super().__init__()

        self._algorithm = algorithm
        self._dataset = dataset

        LangchainHelper.bind_model(model_name)

        self._model = LangchainHelper.load_model()
        self._embeddings = LangchainHelper.load_embeddings()

        if hparam:
            hparam = f"_{hparam}"

        self._run_id = f"{self._dataset}_{self._algorithm}{hparam}"
        self._run_id += f"_{self._model.model_name}"

        self._example_df = self._load_dataset()

        self._result_df = pd.DataFrame()

        self._evaluation_df = pd.DataFrame({
            "run_id": [self._run_id],
            "dataset": [self._dataset],
            "model": [self._model.model_name],
            "algorithm": [self._algorithm],
        })

    def _load_dataset(self) -> dict[str, pd.DataFrame]:
        dataset_dir = self.output_dir / "dataset"

        example_df = {}
        for tag in ["train", "test"]:
            json_file = dataset_dir / tag / f"{self.dataset}.json"
            example_df[tag] = pd.read_json(json_file)

        return example_df

    @abstractmethod
    def classify(self) -> None:
        pass

    def classify_foreach(self,
                         field: pd.Series,
                         chain: Runnable,
                         category_text: str,
                         category_labels: list[str],
                         total: int,
                         hparam: dict[str, Any] = {},
                         others: dict[str, BaseCallbackHandler] = {}
                         ) -> pd.Series:
        question = f"""
        *사용자 질문*: 입력된 CT 판독문이 어떤 유형에 속하는지와 그 근거는 무엇입니까?

        *CT 판독문 내용*:
        본문: {field["본문"]}
        결론: {field["결론"]}
        """

        timer = Timer()

        answer = chain.invoke(
            {
                "question": question,
                "category_text": category_text,
            } | hparam
        )

        timer.lap()

        entry = pd.Series({
            "run_id": self.run_id,
            "본문": field["본문"],
            "결론": field["결론"],
            "정답": field["유형"],
            "예측": match_text(answer.category, category_labels),
            "원예측": answer.category,
            "근거": answer.reason,
            "소요시간": timer.seconds,
        })

        entry = pd.Series({
            name: handler()
            for name, handler in others.items()
        }).combine_first(entry)

        self.INFO(
            f"TIME(sec):{entry['소요시간']:.2f}   "
            f"{'HIT' if entry['정답'] == entry['예측'] else 'MISS'}  "
            f"(GT:{Q(entry['정답'])} PV:{Q(entry['예측'])})"
        )

        self.INFO(
            f"RUN {Q(self.run_id)} CLASSIFY "
            f"[{field.name + 1:02d}/{total:02d}] proceed DONE"
        )

        return entry

    def save_data(self) -> None:
        self.INFO(f"RUN {Q(self.run_id)} DATA saved....")

        total = self._result_df.shape[0]

        self._result_df["hit"] = (
            self._result_df["정답"] == self._result_df["예측"]
        )

        hit = self._result_df["hit"].sum()
        hit_ratio = (hit / total) * 100

        self._evaluation_df[
            ["hit_ratio", "hit", "total"]
        ] = [hit_ratio, hit, total]

        self.INFO(
            f"RUN {Q(self.run_id)} hit ratio: {hit_ratio:.1f}% ({hit}/{total})"
        )

        datastore_dir = self.output_dir / "datastore"
        datastore_dir.mkdir(parents=True, exist_ok=True)
        data_json = datastore_dir / f"{self.run_id}.json"

        to_json(data_json, self._evaluation_df, self._result_df)

        self.INFO(f"RUN {Q(self.run_id)} DATA saved DONE")

    def stringfy_examples(self) -> str:
        example_df = self._example_df["train"].reset_index(drop=True)

        examples = example_df.apply(
            lambda field: (
                f"*예제 {field.name + 1}. CT 판독문*:\n"
                f"**유형**: {field['유형']}\n"
                f"**본문**: {field['본문']}\n"
                f"**결론**: {field['결론']}"
            ),
            axis=1
        )

        return "\n\n".join(examples)

    def stringfy_categories(self) -> str:
        prefix = self.dataset.split("_")[0]
        txt_file = self.data_dir / f"{prefix}.txt"
        text = txt_file.read_text('utf-8')
        return text

    def get_category_labels(self) -> list[str]:
        return self._example_df["train"]["유형"].unique().tolist()

    def get_sample_question(self,
                            tag: str = "train",
                            k: int = 0
                            ) -> str:
        example_df = self._example_df[tag]
        body, conclusion = example_df.loc[k, ["본문", "결론"]]

        question = f"""
            *사용자 질문*: 입력된 CT 판독문이 어떤 유형에 속하는지와 그 근거는 무엇입니까?

            *CT 판독문 내용*:
            본문: {body}
            결론: {conclusion}
            """

        return question

    @staticmethod
    def concat_documents(documents: list[Document]) -> str:
        return concat_texts(documents, "page_content", "\n\n")

    @staticmethod
    def reorder_documents(documents: list[Document]) -> list[Document]:
        reordering = LongContextReorder()
        return reordering.transform_documents(documents)

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @property
    def model(self) -> BaseChatModel:
        return self._model

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @property
    def run_id(self) -> str:
        return self._run_id
