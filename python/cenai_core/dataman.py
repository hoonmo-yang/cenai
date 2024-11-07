from typing import Any, Hashable, Sequence

import json
from operator import attrgetter, itemgetter
from pathlib import Path
from rapidfuzz import process
import textwrap
import yaml


def match_text(keyword: str, candidates: list[str]) -> str:
    best_match = process.extractOne(
        keyword, candidates
    )
    return best_match[0]


def load_json_yaml(src: Path, **kwargs) -> dict[Hashable, Any]:
    '''
    Loads json or yaml file to a dict and returns it.
    It is determined by file extension whether the src is
    a json or yaml file.
    '''
    if not src.is_file() or src.suffix not in [".json", ".yaml", ".yml"]:
        raise TypeError(f"{src} isn't either yaml or json file")

    with src.open() as fin:
        deserial = (
            json.load(fin, **kwargs) if src.suffix == ".json" else
            yaml.load(fin, Loader=yaml.Loader, **kwargs)
        )
    return deserial


def dump_json_yaml(content: dict[str, Any], dst: Path, **kwargs) -> None:
    '''
    Dumps the content dict to the dst file. The file serialization format
    (either json or yaml) is deterimed by the file extension of the dst file.
    '''
    if dst.suffix not in [".json', '.yaml', '.yml"]:
        raise TypeError(f"{dst} isn't either yaml or json file")

    with dst.open("wt", encoding="utf-8") as fout:
        if dst.suffix == ".json":
            json.dump(content, fout, ensure_ascii=False, indent=4, **kwargs)
        else:
            yaml.dump(content, fout, sort_keys=False, **kwargs)


def concat_texts(
        chunks: Sequence[Any],
        field: str,
        seperator: str = "\n"
) -> str:
    if not chunks:
        return ""

    sample = chunks[0]

    selector = (
        itemgetter(field)
        if hasattr(sample, "__getitem__") and not isinstance(sample, str) else
        attrgetter(field) if hasattr(sample, field) else
        None
    )

    if selector is None:
        return ""

    return seperator.join([selector(chunk) for chunk in chunks])


def Q(text: str) -> str:
    return f"'{text}'"


def QQ(text: str) -> str:
    return f"\"{text}\""


def dedent(source: str) -> str:
    return textwrap.dedent(source).strip()


class Struct:
    def __init__(self, data: dict[str, Any]):
        self.__dict__.update(data)
