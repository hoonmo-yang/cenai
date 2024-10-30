from __future__ import annotations
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from cenai_core import cenai_path, INFO, LangchainHelper, load_dotenv


class BaseClassifier:
    output_dir = cenai_path("output") / "amc"
    data_dir = cenai_path("data") / "amc"

    def __init__(self,
                 model_name: str,
                 model: BaseChatModel,
                 embeddings: Embeddings,
                 dataset_name: str,
                 test_size: float,
                 random_state: int,
                 topk: int
                 ):
        self._model = model
        self._embeddings = embeddings

        test = int(test_size * 10)
        train = 10 - test
        suffix = f"{train}-{test}_{random_state:02d}"
        self._dataset_name = dataset_name
        self._dataset_body = f"{dataset_name}_{suffix}"
        self._body = f"{model_name}_{dataset_name}_{suffix}"
        self._topk = topk

        self._example_df = self._load_dataset()

        self._experiment_df = pd.DataFrame(
            columns=("본문", "결론", "정답", "예측", "근거",)
        )

    @classmethod
    def create_dataset(cls,
                       dataset_stem: str,
                       test_sizes: list[float],
                       num_tries: int
                       ) -> list[str]:
        dataset_dir = cls.output_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        json_file = cls.data_dir / f"{dataset_stem}.json"
        source_df = cls._create_sources(json_file)
        target_df = {}
        dataset_names =[]

        for version, source_df in source_df.items():
            dataset_name = f"{dataset_stem}_{version}"
            dataset_names.append(dataset_name)

            for test_size in test_sizes:
                for k in range(num_tries):
                    target_df["train"] = pd.DataFrame()
                    target_df["test"] = pd.DataFrame()

                    for _, dataframe in source_df.groupby("유형"):
                        train_df, test_df = train_test_split(
                            dataframe,
                            test_size=test_size,
                            random_state=k,
                        )

                        target_df["train"] = pd.concat(
                            [target_df["train"], train_df], axis=0
                        )

                        target_df["test"] = pd.concat(
                            [target_df["test"], test_df], axis=0
                        )

                    test = int(test_size * 10)
                    train = 10 - test
                    suffix = f"{train}-{test}_{k:02d}"

                    for tag in ["train", "test"]:
                        target_dir = dataset_dir / tag
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_json = target_dir / f"{dataset_name}_{suffix}.json"
                        target_df[tag].to_json(target_json)

        return dataset_names

    @classmethod
    def _create_sources(cls,
                        json_file: Path,
                        ) -> dict[str, pd.DataFrame]:
        original_df = pd.read_json(json_file)
        original_df["원유형"] = original_df["유형"]

        include_df = original_df.replace({
            "유형": {
                "initial diagnosis": "initial staging",
                "unresected cancer follow up": "restaging",
            }
        })

        exclude_df = original_df[
            ~original_df["유형"].isin([
                "initial diagnosis",
                "restaging",
            ]) 
        ]

        return {
            "org": original_df,
            "inc": include_df,
            "exc": exclude_df,
        }

    @property
    def model(self) -> BaseChatModel:
        return self._model

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    def _load_dataset(self) -> dict[str, pd.DataFrame]:
        dataset_dir = self.output_dir / "dataset"

        example_df = {}
        for tag in ["train", "test"]:
            json_file = dataset_dir / tag / f"{self._dataset_body}.json"
            example_df[tag] = pd.read_json(json_file)

        return example_df

    @classmethod
    def evaluate(cls,
                 model_names: list[str],
                 dataset_names: list[str],
                 test_sizes: list[float],
                 num_tries: int,
                 topks: list[int],
                 classifier_class: BaseClassifier
                 ) -> None:
        load_dotenv(False)

        for model_name in model_names:
            LangchainHelper.bind_model(model_name)
            model = LangchainHelper.load_model()
            embeddings = LangchainHelper.load_embeddings()

            for dataset_name in dataset_names:
                for test_size in test_sizes:
                    for i in range(num_tries):
                        for topk in topks:
                            INFO(
                                f"\nmodel:{model_name} dataset:{dataset_name} "
                                f"seed:{i}, test_size:{test_size} topk:{topk}"
                            )

                            classifier = classifier_class(
                                model_name=model_name,
                                model=model,
                                embeddings=embeddings,
                                dataset_name=dataset_name,
                                test_size=test_size,
                                random_state=i,
                                topk=topk,
                            )

                            classifier.classify()
                            classifier.create_report()
