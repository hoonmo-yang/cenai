from __future__ import annotations
from typing import Any, Sequence, Union

from abc import ABC, abstractmethod
from datetime import datetime
import importlib
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
import re
from itertools import product
from sklearn.model_selection import train_test_split
import sys

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from cenai_core.dataman import load_json_yaml, ordinal, Q, Struct, to_camel
from cenai_core.langchain_helper import LangchainHelper
from cenai_core.logger import Logger
from cenai_core.system import cenai_path, load_dotenv


class Gridsuite(BaseModel):
    prefix: str = Field(description="prefix of gridsuite name")
    create_date: str = Field(description="date when the grid suite is created")
    index: int = Field(description="chronological order per date")
    artifact_dir: Path = Field(description="dir of gridsuite artifact")
    profile_file: str = Field(description="config profile file of gridsuite")

    @property
    def id(self) -> str:
        return f"{self.prefix}_{self.create_date}_{self.index:03d}"

    @property
    def prefix_dir(self) -> str:
        return self.artifact_dir / self.prefix

    @property
    def base_dir(self) -> str:
        return self.prefix_dir / self.id


class GridRunnable(Logger, ABC):
    logger_name = "cenai.grid_runnable"

    def __init__(self,
                 model: str,
                 case_suffix: str,
                 dataset_suffix: str,
                 metadata: Struct
                 ):
        self._metadata = metadata

        self._suite_id = self.metadata.suite.id
        self._suite_prefix = self.metadata.suite.prefix

        log_file = (
            cenai_path("log") /
            self._suite_prefix /
            f"{self._suite_id}.log"
        )

        super().__init__(
            log_file=log_file,
        )

        self._case_id = "_".join([
            token for token in [
                self.metadata.dataset_stem,
                dataset_suffix,
                model,
                self.metadata.module,
                case_suffix,
            ] if token
        ])

        suite_prefix_dir = self.metadata.suite.prefix_dir
        self._dataset_dir = suite_prefix_dir / "dataset"
        self._datastore_dir = suite_prefix_dir / "datastore" / self.case_id

        self._source_dir = (
            cenai_path("data") /
            self.metadata.institution /
            self.metadata.task 
        )

        self._corpus_dir = self._source_dir / "corpus"
        self._content_dir = self._source_dir / "content"

        LangchainHelper.bind_model(model)

        self._model = LangchainHelper.load_model()
        self._embeddings = LangchainHelper.load_embeddings()

        self._case_df = pd.DataFrame()

        self._metadata_df = pd.DataFrame({
            "suite_id": [self.suite_id],
            "case_id": [self.case_id],
            "suite_prefix": [self.metadata.suite.prefix],
            "suite_create_date": [self.metadata.suite.create_date],
            "suite_index": [self.metadata.suite.index],
            "institution": [self.metadata.institution],
            "task": [self.metadata.task],
            "tags": [",".join(self.metadata.tags)],
            "model": [self.model.model_name],
            "module": [self.metadata.module],
            "dataset_prefix": [self.metadata.dataset_prefix],
            "dataset_stem": [self.metadata.dataset_stem],
            "dataset_ext": [self.metadata.dataset_ext],
            "profile_file": [self.metadata.suite.profile_file],
        })

    @abstractmethod
    def run(self, **directive) -> None:
        pass

    @abstractmethod
    def save_data(self, **directive) -> None:
        pass

    @property
    def dataset_dir(self) -> Path:
        return self._dataset_dir

    @property
    def datastore_dir(self) -> Path:
        return self._datastore_dir

    @property
    def source_dir(self) -> Path:
        return self._source_dir

    @property
    def corpus_dir(self) -> Path:
        return self._corpus_dir

    @property
    def content_dir(self) -> Path:
        return self._content_dir

    @property
    def model(self) -> BaseChatModel:
        return self._model

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @property
    def suite_id(self) -> str:
        return self._suite_id

    @property
    def case_id(self) -> str:
        return self._case_id

    @property
    def header(self) -> str:
        return f"CASE {Q(self.case_id)}[{Q(self.suite_id)}]"

    @property
    def metadata(self) -> Struct:
        return self._metadata

    @property
    def metadata_df(self) -> pd.DataFrame:
        return self._metadata_df

    @property
    def result_df(self) -> pd.DataFrame:
        return self._result_df

    @result_df.setter
    def result_df(self, value: pd.DataFrame) -> None:
        self._result_df = value


class EvaluateGridRunnable(GridRunnable):
    def __init__(self,
                 model: str,
                 case_suffix: str,
                 dataset_suffix: str,
                 metadata: Struct
                 ):

        super().__init__(
            model=model,
            case_suffix=case_suffix,
            dataset_suffix=dataset_suffix,
            metadata=metadata,
        )

        self._dataset_df = self._load_dataset()

    def _load_dataset(self) -> dict[str, pd.DataFrame]:
        dataset_df = {}

        for tag in ["train", "test"]:
            json_file = (
                self.dataset_dir / tag /
                f"{self.metadata.dataset_stem}{self.metadata.dataset_ext}"
            )

            dataset_df[tag] = pd.read_json(json_file)

        return dataset_df

    @property
    def dataset_df(self) -> pd.DataFrame:
        return self._dataset_df

class GridRunner(Logger):
    logger_name = "cenai.system"

    artifact_dir = cenai_path("artifact")
    P_GRID_CASE = re.compile(r"[a-zA-Z0-9-]+_\d{4}-\d{2}-\d{2}_(\d+)")

    def __init__(self, module_paths: Sequence[Union[Path, str]] = []):
        super().__init__()

        self._add_module_paths(module_paths)

    @staticmethod
    def _add_module_paths(
        module_paths: Sequence[Union[Path, str]]
        ) -> None:
        for module_path in module_paths:
            module_path = Path(module_path).resolve()

            if module_path == Path.cwd().resolve():
                continue

            if str(module_path) not in sys.path:
                sys.path.insert(0, str(module_path))

    @staticmethod
    def _get_gridsuite_recipe(
            data_source: Union[Path, dict[str, Any]]
            ) -> Struct:

        if isinstance(data_source, Path):
            profile_file = data_source.resolve()
            profile = load_json_yaml(data_source)
            location = f"file {Q(profile_file)}"
        else:
            profile_file = ""
            profile = data_source
            location = f"var {Q('data_source')}",

        type_checks = [
            ["metadata", "", dict],
            ["version", "metadata", str],
            ["name", "metadata", str],
            ["institution", "metadata", str],
            ["task", "metadata", str],
            ["tags", "metadata", list],
            ["directive", "", dict],
            ["templates", "", list],
            ["model", "", list],
            ["dataset", "", dict],
            ["corpus", "dataset", list],
            ["test_size", "dataset", list],
            ["keywords", "dataset", list],
            ["pick", "dataset", list],
        ]

        profile[""] = profile

        for key, node, type_ in type_checks:
            node_name = Q(node) if node else "root"

            if key not in profile[node]:
                raise KeyError(
                    f"{Q(key)} key missing on {node_name} node in {location}"
                )

            if not isinstance(profile[node][key], type_):
                raise ValueError(
                    f"value of {Q(key)} key not {Q(type_)} type "
                    f"on {node_name} node in {location}"
                )

        profile.pop("")

        default_keys = [
            "model",
        ]

        default_params = {
            key: profile[key] for key in default_keys
        }

        templates = profile.pop("templates")
        all_params = []

        for i, template in enumerate(templates):
            params = {}

            modules = template.pop("module", [])
            if not modules:
                raise ValueError(
                    f"{Q('module')} key does not exist or is empty "
                    f"in {ordinal(i + 1)} element of {Q('templates')} key "
                    f"in {location}"
                )

            params["module"] = modules

            for category, keys in template.items():
                if category not in profile:
                    raise KeyError(
                        f"{Q(category)} node missing in {location}"
                    )

                if not keys:
                    keys = profile[category].keys()

                for key in keys:
                    branch = profile[category]
                    if key not in branch:
                        raise KeyError(
                            f"{Q(key)} key missing in {Q(category)} branch "
                            f"in {location}"
                        )

                    if key in params:
                        raise KeyError(
                            f"{Q(category)} key contains "
                            f"a duplicate name for {Q(key)} key"
                            "Change the duplicate keys to resolve it"
                        )

                    params[key] = branch[key]

            all_params.append(default_params | params)

        recipe = {
            "cases": all_params,
            "profile_file": str(profile_file),
        } | {
            key: profile[key] for key in [
                "metadata",
                "directive",
                "dataset",
            ]
        }

        return Struct(recipe)

    @classmethod
    def replicate_gridsuite(cls, recipe: Struct) -> Gridsuite:
        name = recipe.metadata["name"]

        datastore_dir = cls.artifact_dir / name / "datastore"
        datastore_dir.mkdir(parents=True, exist_ok=True)

        date = recipe.directive.get("fixed_data")

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        indices = [
            cls._extract_gridsuite_index(path)
            for path in datastore_dir.glob(f"{name}_{date}*")
        ]

        replicate = recipe.directive.get("replicate", True)

        index = max(indices) + int(replicate) if indices else 1

        suite = Gridsuite(
            prefix=name,
            index=index,
            create_date=date,
            artifact_dir=cls.artifact_dir,
            profile_file=recipe.profile_file,
        )
        return suite

    @classmethod
    def _extract_gridsuite_index(cls, path: Path) -> int:
        if not path.is_dir():
            return -1

        match = cls.P_GRID_CASE.fullmatch(path.name)
        return int(match[1]) if match else -1

    def get_instance(
            self,
            case_args: dict[str, Any],
            dataset_args: dict[str, Any],
            metadata: dict[str, Any],
            suite: Gridsuite
            ) -> GridRunnable:

        module = case_args.pop("module")

        class_name = to_camel(module.replace("-", "_"))
        module_name = module.replace("*", "").replace("-", "_")

        try:
            module = importlib.import_module(module_name)
            Class = getattr(module, class_name)
        
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Can't import the module {Q(module_name)}"
            )

        except AttributeError:
            raise AttributeError(
                f"Can't import the class {module_name}.{class_name}"
            )

        metadata = {
            "module": module_name.replace("_", "-"),
            "suite": suite,
        } | metadata | dataset_args

        return Class(
            metadata=Struct(metadata),
            **case_args
        )

    def invoke(self, data_source: Union[Path, dict[str,Any]]) -> None:
        suite = None

        try:
            recipe = self._get_gridsuite_recipe(data_source)
            suite = self.replicate_gridsuite(recipe)

            self.INFO(f"SUITE {Q(suite.id)} proceed ....")

            all_dataset_args = self.prepare_datasets(
                dataset=recipe.dataset,
                metadata=recipe.metadata,
                suite=suite,
            )

            load_dotenv(recipe.directive.get("langsmith"))

            for case in recipe.cases:
                for values in product(*case.values()):
                    case_args = dict(
                        zip(case.keys(), values)
                    )

                    for dataset_args in all_dataset_args:
                        try:
                            instance = self.get_instance(
                                case_args=dict(case_args),
                                dataset_args=dataset_args,
                                metadata=recipe.metadata,
                                suite=suite,
                            )

                            instance.run(**recipe.directive)
                            instance.save_data(**recipe.directive)

                        except Exception as error:
                            self.ERROR(error)
                            self.ERROR(f"CASE proceed SKIP")

        except Exception as error:
            self.ERROR(error)
            suite_id = "Unknown" if suite is None else suite.id
            self.INFO(f"SUITE {Q(suite_id)} proceed SKIP")
        else:
            self.INFO(f"SUITE {suite.id} proceed DONE")

    def __call__(self, data_source: Union[Path, dict[str, Any]]) -> None:
        self.invoke(data_source)


class EvaluateGridRunner(GridRunner):
    def __init__(self, module_paths: Sequence[Union[Path, str]] = []):
        super().__init__(
            module_paths=module_paths,
        )

    def prepare_datasets(
            self,
            dataset: dict[str, Any],
            metadata: dict[str, Any],
            suite: Gridsuite
            ) -> list[dict[str, str]]:
        dataset_dir = suite.prefix_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        all_dataset_args = []

        for values in product(*dataset.values()):
            split_args = dict(
                zip(dataset.keys(), values)
            )

            all_dataset_args.extend(
                self._split_datasets(
                    dataset_dir=dataset_dir,
                    metadata=Struct(metadata),
                    **split_args
                )
            )

        return all_dataset_args

    def _split_datasets(
        self,
        dataset_dir: Path,
        metadata: Struct,
        corpus: str,
        test_size: float,
        keywords: Union[str, list[str]],
        pick: Union[int, list[int]]
        ) -> list[dict[str, Any]]:

        corpus_dir = (
            cenai_path("data") /metadata.institution / metadata.task / "corpus"
        )

        corpus_file = corpus_dir / f"{corpus}"
        records = load_json_yaml(corpus_file)
        source_df = pd.DataFrame(records)

        all_dataset_args = []

        pick = [pick] if isinstance(pick, int) else pick
        test = int(test_size * 10)
        train = 10 - test
        corpus = corpus.split(".")[0]

        keywords = keywords if isinstance(keywords, list) else [keywords]

        for i in range(*pick):
            dataset_prefix = f"{corpus}_{train}-{test}"
            dataset_stem = f"{corpus}_{train}-{test}_{i:02d}"

            target_df = {key: pd.DataFrame() for key in ["train", "test"]}

            for _, dataframe in source_df.groupby(keywords):
                trainset_df, testset_df = train_test_split(
                    dataframe,
                    test_size=test_size,
                    random_state=i,
                )

                target_df["train"] = pd.concat(
                    [target_df["train"], trainset_df], axis=0
                )

                target_df["test"] = pd.concat(
                    [target_df["test"], testset_df], axis=0
                )

            for tag in ["train", "test"]:
                dataframe = target_df[tag].reset_index(drop=True)
                target_dir = dataset_dir / tag
                target_dir.mkdir(parents=True, exist_ok=True)
                target_json = target_dir / f"{dataset_stem}.json"
                dataframe.to_json(target_json)

            all_dataset_args.append({
                "dataset_prefix": dataset_prefix,
                "dataset_stem": dataset_stem,
                "dataset_ext": ".json",
            })

        return all_dataset_args
