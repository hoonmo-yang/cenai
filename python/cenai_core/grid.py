from __future__ import annotations
from typing import Any, Union

from abc import ABC, abstractmethod
from datetime import datetime
import importlib
import pandas as pd
import re
from itertools import product
from pathlib import Path
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
import sys

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from cenai_core.dataman import load_json_yaml, Q, Struct, to_camel
from cenai_core.langchain_helper import LangchainHelper
from cenai_core.logger import Logger
from cenai_core.system import cenai_path, load_dotenv


class Grid(BaseModel):
    name: str = Field(description="Grid name")

    date: str = Field(description="date of grid")

    index: int = Field(
        description="Chronological order per grid",
        ge=0,)

    prefix_dir: Path = Field(description="prefix dir of grid")

    @property
    def grid_id(self) -> str:
        return f"{self.name}_{self.date}_{self.index:03d}"

    @property
    def grid_dir(self) -> Path:
        return self.prefix_dir / self.name


class GridChainContext(BaseModel):
    parameter: dict[str, Any] = Field(
        default={},
    )

    handler: dict[str, BaseCallbackHandler] = Field(
        default={},
    )

    class Config:
        arbitrary_types_allowed = True


class GridRunner(Logger):
    logger_name = "cenai.system"

    def __init__(self,
                 module_paths: list[Union[Path,str]] = []
                 ):
        super().__init__()

        self._prefix_dir = cenai_path("gridout")
        self._add_module_paths(module_paths)

        self.P_GRID = re.compile(r"\w+_\d{4}-\d{2}-\d{2}_(\d+)")

    @staticmethod
    def _add_module_paths(module_paths: list[Union[Path,str]]) -> None:
        for module_path in module_paths:
            module_path = Path(module_path).resolve()

            if module_path == Path.cwd().resolve():
                continue

            if str(module_path) not in sys.path:
                sys.path.insert(0, str(module_path))

    def load_config(self,
                    config: Union[Path, Struct]
                    ) -> tuple[Struct, str]:
        if isinstance(config, Struct):
            return config, ""

        grid_yaml = config.resolve()
        deserialized = load_json_yaml(grid_yaml)
        return Struct(deserialized), str(grid_yaml)

    def replicate_grid(self, config: Struct) -> list[str]:
        top_dir = self.prefix_dir / config.name / "datastore"
        top_dir.mkdir(parents=True, exist_ok=True)

        date = datetime.now().strftime("%Y-%m-%d")

        numbers = [
            self._extract_grid_index(grid_dir)
            for grid_dir in top_dir.glob(f"{config.name}_{date}*")
        ]

        replicate = config.directive["replicate"]
        index = max(numbers) + int(replicate) if numbers else 1

        grid = Grid(
            name=config.name,
            index=index,
            date=date,
            prefix_dir=self.prefix_dir,
        )

        self.INFO(f"GRID {Q(grid.grid_id)} replicated DONE")
        return grid

    def _extract_grid_index(self, grid_dir: Path) -> int:
        if not grid_dir.is_dir():
            return -1

        match = self.P_GRID.fullmatch(grid_dir.stem)
        return int(match[1]) if match else -1

    def get_instance(self,
                     institution: str,
                     dataset: str,
                     tags: list[str],
                     grid: Grid,
                     grid_yaml: Path,
                     **parameter
                     ) -> GridRunnable:
        
        module = parameter.pop("module", None)
        model_name = parameter.pop("model", None)

        if module is None:
            raise KeyError(
                f"{Q('parameter.module')} missing in file {Q(grid_yaml)}"
            )

        if model_name is None:
            raise KeyError(
                f"No {Q('parameter.model_name')} missing in file {Q(grid_yaml)}"
            )

        class_name = to_camel(module.replace("-", "_"))
        module_name = module.replace("*", "").replace("-", "_")

        try:
            module = importlib.import_module(module_name)
            Class = getattr(module, class_name)
        
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Can't import the module {Q(module_name)} "
                f"specified in file {Q(grid_yaml)}"
            )
        
        except AttributeError:
            raise AttributeError(
                f"Can't import the class {module_name}.{class_name} "
                f"specified in file {Q(grid_yaml)}"
            )

        metadata = Struct({
            "module": module_name.replace("_", "-"),
            "institution": institution,
            "dataset": dataset,
            "model": model_name,
            "grid": grid,
            "tags": tags,
            "grid_yaml": grid_yaml,
        })

        return Class(
            metadata=metadata,
            **parameter,
        )

    def __call__(self, config: Union[Path, Struct]) -> None:
        grid = None

        try:
            config, grid_yaml = self.load_config(config)
            grid = self.replicate_grid(config)

            self.INFO(f"GRID {Q(grid.grid_id)} proceed ....")

            dataset_pairs = self.prepare_datasets(grid, config)

            load_dotenv(config.directive.get("langsmith"))

            for values in product(*config.parameter.values()):
                parameter = dict(zip(config.parameter.keys(), values))

                for institution, dataset in dataset_pairs:
                    try:
                        instance = self.get_instance(
                            institution=institution,
                            dataset=dataset,
                            tags=config.tags,
                            grid=grid,
                            grid_yaml=grid_yaml,
                            **parameter
                        )

                        instance.run(config.directive)
                        instance.save_data(config.directive)

                    except Exception as error:
                        self.ERROR(error)
                        self.ERROR(f"RUN proceed SKIP")

        except Exception as error:
            self.ERROR(error)
            grid_id = "unknown" if grid is None else grid.grid_id
            self.INFO(f"GRID {Q(grid_id)} proceed SKIP")
        else:
            self.INFO(f"GRID {grid.grid_id} proceed DONE")

    @property
    def prefix_dir(self) -> Path:
        return self._prefix_dir


class GridGenerator(GridRunner):
    def __init__(self,
                 module_paths: list[Union[Path,str]] = [],
                 ):
        super().__init__(
            module_paths=module_paths,
        )

    def prepare_datasets(self,
                         grid: Grid,
                         config: Struct
                         ) -> list[tuple[str, str]]:
        pass


class GridEvaluator(GridRunner):
    def __init__(self,
                 module_paths: list[Union[Path,str]] = [],
                 ):
        super().__init__(
            module_paths=module_paths,
        )

    def prepare_datasets(self,
                         grid: Grid,
                         config: Struct
                         ) -> list[tuple[str, str]]:
        dataset_dir = grid.grid_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        dataset_pairs = []
        dataset_param = config.dataset
        for values in product(*dataset_param.values()):
            parameter = dict(zip(dataset_param.keys(), values))

            dataset_pairs.extend(
                self._split_datasets(dataset_dir, **parameter)
            )

        return dataset_pairs

    def _split_datasets(self,
                        dataset_dir: Path,
                        institution: str,
                        dataset_prefix: str,
                        test_size: float,
                        pick: Union[int, list[int]],
                        keywords: Union[str, list[str]]
                        ) -> list[str]:
        source_dir = cenai_path("data") / institution
        json_file = source_dir / f"{dataset_prefix}.json"
        source_df = pd.read_json(json_file)

        dataset_pairs = []

        pick = [pick] if isinstance(pick, int) else pick
        test = int(test_size * 10)
        train = 10 - test
        keywords = keywords if isinstance(keywords, list) else [keywords]

        for i in range(*pick):
            dataset = f"{dataset_prefix}_{train}-{test}_{i:02d}"
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
                target_json = target_dir / f"{dataset}.json"
                dataframe.to_json(target_json)

            dataset_pairs.append([institution, dataset])

        return dataset_pairs


class GridRunnable(Logger, ABC):
    logger_name = "cenai.gridrunnable"

    def __init__(self,
                 metadata: Struct,
                 dataset_suffix: str,
                 module_suffix: str
                 ):
        self._metadata = metadata

        log_file = (
            cenai_path("log") /
            self.metadata.grid.name /
            f"{self.metadata.grid.grid_id}.log"
        )

        super().__init__(
            log_file=log_file,
        )

        if dataset_suffix:
            dataset_suffix = f"_{dataset_suffix}"

        if module_suffix:
            module_suffix = f"_{module_suffix}"

        self._run_id = (
            f"{self.metadata.dataset}{dataset_suffix}"
            f"_{self.metadata.module}{module_suffix}"
        )

        self._grid_dir = self._metadata.grid.grid_dir
        self._dataset_dir = self._grid_dir / "dataset"
        self._datastore_dir = self._grid_dir / "datastore"
        self._source_dir = cenai_path("data") / self._metadata.institution

        LangchainHelper.bind_model(metadata.model)

        self._model = LangchainHelper.load_model()
        self._embeddings = LangchainHelper.load_embeddings()

        self._dataset_df = self._load_dataset()
        self._result_df = pd.DataFrame()

        self._metadata_df = pd.DataFrame({
            "grid_id": [self._metadata.grid.grid_id],
            "run_id": [self._run_id],
            "grid_name": [self._metadata.grid.name],
            "grid_index": [self._metadata.grid.index],
            "module": [self._metadata.module],
            "institution": [self._metadata.institution],
            "dataset": [self._metadata.dataset],
            "model": [self._metadata.model],
            "tags": [",".join(self._metadata.tags)],
            "grid": [str(self._metadata.grid)],
        })

    def _load_dataset(self) -> dict[str, pd.DataFrame]:
        dataset_df = {}

        for tag in ["train", "test"]:
            json_file = self.dataset_dir / tag / f"{self.metadata.dataset}.json"

            if json_file.is_file():
                dataset_df[tag] = pd.read_json(json_file)
            else:
                dataset_df[tag] = pd.DataFrame()

        return dataset_df

    @abstractmethod
    def run(self, directive: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def save_data(self, directive: dict[str, Any]) -> None:
        pass

    @property
    def source_dir(self) -> Path:
        return self._source_dir

    @property
    def dataset_dir(self) -> Path:
        return self._dataset_dir

    @property
    def datastore_dir(self) -> Path:
        return self._datastore_dir

    @property
    def model(self) -> BaseChatModel:
        return self._model

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def header(self) -> str:
        return f"RUN {Q(self.run_id)}[{Q(self.grid_id)}]"

    @property
    def metadata(self) -> Struct:
        return self._metadata

    @property
    def grid_id(self) -> str:
        return self.metadata.grid.grid_id

    @property
    def metadata_df(self) -> pd.DataFrame:
        return self._metadata_df

    @property
    def dataset_df(self) -> pd.DataFrame:
        return self._dataset_df

    @property
    def result_df(self) -> pd.DataFrame:
        return self._result_df

    @result_df.setter
    def result_df(self, value: pd.DataFrame) -> None:
        self._result_df = value