from __future__ import annotations
from typing import Any, Sequence, Union

import copy
from datetime import datetime
import importlib
from itertools import product
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
import re
import shutil
from sklearn.model_selection import train_test_split
import sys

from cenai_core.dataman import (
    load_json_yaml, optional, ordinal, Q, Struct, to_camel
)

from cenai_core.logger import Logger
from cenai_core.pandas_helper import to_json
from cenai_core.grid.runnable import GridRunnable
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


class BaseRunner(Logger):
    logger_name = "cenai.system"

    def __init__(self):
        super().__init__()


class GridRunner(BaseRunner):
    version = "v2"

    data_dir = cenai_path("data")
    artifact_dir = cenai_path("artifact")

    P_GRID_CASE = re.compile(r"[a-zA-Z0-9-]+_\d{4}-\d{2}-\d{2}_(\d+)\.json")

    def __init__(
            self,
            profile: Union[Path, dict[str, Any]],
            module_paths: Sequence[Union[Path, str]] = []
        ):

        super().__init__()

        self._add_module_paths(module_paths)

        self.renew(profile)

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

    def renew(self, profile: Union[Path, dict[str, Any]]) -> None:
        self._recipe = self._load_gridsuite_recipe(profile)
        self._suite = self._replicate_gridsuite()
        self._datastore_dir = self.suite.prefix_dir / "datastore"

        self._corpus_dir = self.suite.prefix_dir / "corpus"
        self._corpus_dir.mkdir(parents=True, exist_ok=True)

        self._export_dir = self.suite.prefix_dir / "export"
        self._export_dir.mkdir(parents=True, exist_ok=True)

        source_dir = Path(
            self.data_dir /
            self.recipe.metadata.institution /
            self.recipe.metadata.task
        )

        self._source_corpus_dir = source_dir / "corpus"
        self._html_dir = source_dir / "html"

        self._metadata_df = pd.DataFrame()
        self._result_df = pd.DataFrame()

    @classmethod
    def _load_gridsuite_recipe(
            cls,
            profile: Union[Path, dict[str, Any]]
        ) -> Struct:

        if isinstance(profile, Path):
            profile_file = profile.resolve()
            profile = load_json_yaml(profile_file)
            location = f"file {Q(profile_file)}"
        else:
            profile_file = ""
            profile = copy.deepcopy(profile)
            location = f"var {Q('profile')}",

        cls._check_profile(profile, location)

        entity = {
            "cases": Struct({
                "keyword": "module",
                "implicit_param_keys": [
                    "models",
                ],
                "all_params": [],
            }),
            "corpora": Struct({
                "keyword": "mode",
                "implicit_param_keys": [],
                "all_params": [],
            }),
        }

        for key, context in entity.items():
            templates = profile.pop(key)

            for i, template in enumerate(templates):
                params = {}
                keyword_values = template.pop(context.keyword, [])

                if keyword_values is None:
                    raise ValueError(f"{Q(context.keyword)} key does not exist "
                                     f"or is empty in {ordinal(i + 1)} element"
                                     f"of {Q(key)} key in {location}")

                params[context.keyword] = keyword_values

                if key in ["cases",]:
                    for category, keys in template.items():
                        if category not in profile:
                            raise KeyError(f"{Q(category)} node missing"
                                           f"in {location}")

                        if keys is None:
                            keys = profile[category].keys()

                        for key in keys:
                            branch = profile[category]

                            if key not in branch:
                                raise KeyError(f"{Q(key)} key missing in "
                                               f"{Q(category)} branch "
                                               f"in {location}")

                            if key in params:
                                raise KeyError(f"{Q(category)} key contains "
                                               "a duplicate name for "
                                               f"{Q(key)} key. Change the "
                                               "duplicate keys to resolve it")

                            params[key] = branch[key]

                elif key in ["corpora",]:
                    if len(keyword_values) > 1:
                        raise ValueError(f"{Q(context.keyword)} key has "
                                         "a list with more than 1 element."
                                         f"in {ordinal(i + 1)} element of "
                                         f"{Q(key)} key in {location}")

                    params.update(template)

            implicit_params = {
                key: profile[key] for key in context.implicit_param_keys
            }

            context.all_params.append(implicit_params | params)

        recipe = {
            key: context.all_params
            for key, context in entity.items()
        } | {
            key: Struct(profile[key]) for key in [
                "metadata",
                "directive",
            ]
        } | {
            key: profile[key] for key in [
                "export",
            ]
        } | {
            "profile_file": str(profile_file),
        }

        return Struct(recipe)

    @classmethod
    def _check_profile(
            cls,
            profile: dict[str, Any],
            location: str
        ) -> None:

        type_checks = [
            ["metadata", "", dict],
            ["version", "metadata", str],
            ["name", "metadata", str],
            ["institution", "metadata", str],
            ["task", "metadata", str],
            ["tags", "metadata", list],
            ["directive", "", dict],
            ["export", "", dict],
            ["models", "", list],
            ["cases", "", list],
            ["corpora", "", list],
        ]

        profile[""] = profile

        for key, node, type_ in type_checks:
            node_name = Q(node) if node else "root"

            if key not in profile[node]:
                raise KeyError(f"{Q(key)} key missing on {node_name} node"
                               f"in {location}")

            if not isinstance(profile[node][key], type_):
                raise ValueError(f"value of {Q(key)} key not {Q(type_)} type "
                                 f"on {node_name} node in {location}")

        profile.pop("")

        version = profile["metadata"].get("version", "").lower()

        if version != cls.version:
            raise ValueError("Profile version not matched: "
                             f"{Q(version)} in {location} != {Q(cls.version)}")

    def _replicate_gridsuite(self) -> Gridsuite:
        prefix = self.recipe.metadata.name
        datastore_dir = self.artifact_dir / prefix / "datastore"
        datastore_dir.mkdir(parents=True, exist_ok=True)

        date = self.recipe.directive.get("fixed_data")

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        indices = [
            self._extract_gridsuite_index(path)
            for path in datastore_dir.glob(f"{prefix}_{date}*.json")
        ]

        replicate = self.recipe.directive.get("replicate", True)

        index = max(indices) + int(replicate) if indices else 1

        suite = Gridsuite(
            prefix=prefix,
            index=index,
            create_date=date,
            artifact_dir=self.artifact_dir,
            profile_file=self.recipe.profile_file,
        )

        return suite

    @classmethod
    def _extract_gridsuite_index(cls, path: Path) -> int:

        if not path.is_file():
            return -1

        match = cls.P_GRID_CASE.fullmatch(path.name)
        return int(match[1]) if match else -1

    def _prepare_corpora(self) -> list[dict[str, str]]:
        all_corpus_args = []

        corpora = self.recipe.corpora

        for i, corpus in enumerate(corpora):
            if corpus["mode"][0] in ["aggregate",]:
                some_corpus_args = self._prepare_aggregate(
                    order=i + 1,
                    corpus=corpus,
                )

                all_corpus_args.extend(some_corpus_args)
                continue

            for values in product(*corpus.values()):
                corpus_args = dict(zip(corpus.keys(), values))

                some_corpus_args = (
                    self._prepare_dataset
                    if corpus_args["mode"] in ["dataset",] else

                    self._prepare_document
                    # if corpus_args["mode"] in ["document",] else
                )(**corpus_args)

                all_corpus_args.extend(some_corpus_args)

        return all_corpus_args

    def _prepare_aggregate(
            self,
            order: int,
            corpus: dict[str, Any]
        ) -> list[dict[str, Any]]:

        if len(corpus["prefix"]) > 1:
            raise ValueError(
                f"{Q('prefix')} key has "
                "a list with more than 1 element "
                f"in {ordinal(order)} element of {Q('corpora')} key"
                )

        mode = corpus.pop("mode")[0]
        prefix = corpus.pop("prefix")[0]

        source_dir = self.source_corpus_dir / prefix

        for values in product(*corpus.values()):
            corpus_args = dict(zip(corpus.keys(), values))

            stem = corpus_args["stem"]
            extension = corpus_args["extension"]

            for file_ in source_dir.glob(f"{stem}{extension}"):
                target = self.corpus_dir / prefix / file_.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file_, target)

                self.INFO(f"File {Q(file_)} copied to {Q(target.parent)} DONE")

        return [{
            "corpus_mode": mode,
            "corpus_prefix": prefix,
            "corpus_stem": corpus["stem"],
            "corpus_ext": corpus["extension"],
        }]

    def _prepare_dataset(
            self,
            mode: str,
            prefix: str,
            stem: str,
            extension: str,
            test_size: float,
            keywords: Union[str, list[str]],
            seeds: Union[int, list[Union[int, list[int]]]]
        ) -> list[dict[str, Any]]:

        source_corpus_dir = self.source_corpus_dir / prefix 
        corpus_dir = self.corpus_dir / prefix

        seeds = self._fanout_seeds(seeds)

        test = int(test_size * 10)
        train = 10 - test

        keywords = keywords if isinstance(keywords, list) else [keywords]

        all_corpus_args = []

        for file_ in source_corpus_dir.glob(f"{stem}{extension}"):
            records = load_json_yaml(file_)
            source_df = pd.DataFrame(records)

            for seed in seeds:
                corpus_prefix = "/".join([
                    token for token in [
                        prefix,
                        file_.stem,
                    ] if token
                ])

                corpus_prefix += f"_{train}-{test}"
                corpus_stem = f"{corpus_prefix}_{seed:02d}"

                target_df = {key: pd.DataFrame() for key in ["train", "test"]}

                for _, dataframe in source_df.groupby(keywords):
                    trainset_df, testset_df = train_test_split(
                        dataframe,
                        test_size=test_size,
                        random_state=seed,
                    )

                    target_df["train"] = pd.concat(
                        [target_df["train"], trainset_df], axis=0
                    )

                    target_df["test"] = pd.concat(
                        [target_df["test"], testset_df], axis=0
                    )

                for tag in ["train", "test"]:
                    dataframe = target_df[tag].reset_index().rename(
                        columns={"index": "sample"}
                    )

                    dataframe["sample"] = dataframe["sample"].astype(int)

                    target_dir = corpus_dir / Path(corpus_stem).parent / tag
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_file = target_dir / f"{Path(corpus_stem).name}{extension}"
                    dataframe.to_json(target_file)

                    self.INFO(
                        f"File {Q(target_file)} copied to "
                        f"{Q(target_file.parent)} DONE"
                    )

                all_corpus_args.append({
                    "corpus_mode": mode,
                    "corpus_prefix": corpus_prefix,
                    "corpus_stem": corpus_stem,
                    "corpus_ext": extension,
                })

        return all_corpus_args

    def _fanout_seeds(
            self,
            seeds: Union[int, list[Union[int, list[int]]]]
        ) -> list[int]:

        if isinstance(seeds, int):
            return [seeds]

        targets = []

        for seed in seeds:
            if isinstance(seed, int):
                targets.append(seed)

            elif isinstance(seed, list):
                if len(seed) > 2:
                    seed[1] += seed[0]

                targets.extend(range(*seed[:3]))

        return list(set(targets))

    def _prepare_document(
            self,
            mode: str,
            prefix: str,
            stem: str,
            extension: str,
        ) -> list[dict[str, Any]]:

        source_corpus_dir = self.source_corpus_dir / prefix 
        corpus_dir = self.corpus_dir / prefix

        all_corpus_args = []

        for file_ in source_corpus_dir.glob(f"{stem}{extension}"):
            target = corpus_dir / file_.name
            target.parent.mkdir(parents=True, exist_ok=True)

            shutil.copyfile(file_, target)

            self.INFO(f"File {Q(file_)} copied to {Q(target.parent)} DONE")

            all_corpus_args.append({
                "corpus_mode": mode,
                "corpus_prefix": prefix,
                "corpus_stem": file_.stem,
                "corpus_ext": file_.suffix,
            })

        return all_corpus_args

    def get_instance(
            self,
            case_args: dict[str, Any],
            corpus_args: dict[str, Any],
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

        metadata = Struct({
            "module": module_name.replace("_", "-"),
            "suite": self.suite,
        } | self.recipe.metadata | corpus_args)

        return Class(
            metadata=metadata,
            **case_args
        )

    def yield_result(self) -> pd.DataFrame:
        self.invoke()
        return pd.DataFrame(self.result_df)

    def invoke(self) -> None:
        self.INFO(f"{self.header} proceed ....")

        all_corpus_args = self._prepare_corpora()

        load_dotenv(self.recipe.directive.get("langsmith"))

        for case in self.recipe.cases:
            for values in product(*case.values()):
                case_args = dict(zip(case.keys(), values))

                for corpus_args in all_corpus_args:
                    instance = self.get_instance(
                        case_args=dict(case_args),
                        corpus_args=corpus_args,
                    )

                    instance.run(**self.recipe.directive)

                    self.metadata_df = pd.concat(
                        [self.metadata_df, instance.metadata_df], axis=0
                    )

                    self.result_df = pd.concat(
                        [self.result_df, instance.result_df], axis=0
                    )

        self.INFO(f"{self.header} proceed DONE")

    def save(self) -> None:
        self.INFO(f"{self.header} DATA saved ....")

        save = optional(self.recipe.directive.get("save"), True)

        if save:
            data_json = self.datastore_dir / f"{self.suite_id}.json"
            to_json(data_json, self.metadata_df, self.result_df)

            self.INFO(f"{self.header} DATA saved DONE")
        else:
            self.INFO(f"{self.header} DATA saved SKIP")

    def export(self) -> None:
        self.INFO(f"{self.header} FILE EXPORT proceed ....")

        export = dict(self.recipe.export)

        if not export.pop("enable", False):
            return

        stem = optional(export.pop("stem", None), self.suite_id)
        columns = optional(export.pop("columns", None), {})

        all_columns = {
            key: columns[key]
            for key in ["all", "metadata", "result"]
            if columns.get(key) is not None
        }

        for values in product(*export.values()):
            export_args = dict(zip(export.keys(), values))

            self._export_data(stem, all_columns, **export_args)

        self.INFO(f"{self.header} FILE EXPORT proceed DONE")

    def _export_data(
            self,
            stem: str,
            all_columns: dict[str, Sequence[str]],
            mode: str,
            extension: str
        ) -> None:

        if mode in ["all",]:
            export_df = pd.merge(
                self.result_df,
                self.metadata_df,
                on=["suite_id", "case_id"],
                how="outer",
            )

        elif mode in ["metadata",]:
            export_df = self.metadata_df

        elif mode in ["result",]:
            export_df = self.result_df

        columns = all_columns.get(mode, [])

        columns = [
            column in export_df.columns
            for column in columns
        ]

        if columns:
            export_df = export_df[columns]

        target = self.export_dir / f"{stem}-{mode}{extension}"

        if extension in [".csv",]:
            export_df.to_csv(
                target,
                index=False,
                encoding="utf-8",
            )

        elif extension in [".json",]:
            export_df.to_json(
                target,
                orient="records",
                force_ascii=False,
            )

        elif extension in [".xlsx",]:
            with pd.ExcelWriter(target) as writer:
                export_df.to_excel(
                    writer, sheet_name=mode
                )
        else:
            raise ValueError("export not supported for "
                             f"file type {Q(extension)}")

        self.INFO(f"File {Q(target)} saved Done")

    def __call__(self) -> None:
        self.invoke()
        self.save()
        self.export()

    @property
    def recipe(self) -> Struct:
        return self._recipe

    @property
    def suite(self) -> Gridsuite:
        return self._suite

    @property
    def suite_id(self) -> str:
        return self.suite.id

    @property
    def header(self) -> str:
        return f"SUITE {Q(self.suite_id)}"

    @property
    def source_corpus_dir(self) -> Path:
        return self._source_corpus_dir

    @property
    def html_dir(self) -> Path:
        return self._html_dir

    @property
    def datastore_dir(self) -> Path:
        return self._datastore_dir

    @property
    def corpus_dir(self) -> Path:
        return self._corpus_dir

    @property
    def export_dir(self) -> Path:
        return self._export_dir

    @property
    def metadata_df(self) -> pd.DataFrame:
        return self._metadata_df

    @metadata_df.setter
    def metadata_df(self, value: pd.DataFrame) -> None:
        self._metadata_df = value

    @property
    def result_df(self) -> pd.DataFrame:
        return self._result_df

    @result_df.setter
    def result_df(self, value: pd.DataFrame) -> None:
        self._result_df = value
