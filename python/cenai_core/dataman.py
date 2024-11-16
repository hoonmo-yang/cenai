from typing import Any, Hashable, Sequence

import json
from operator import attrgetter, itemgetter
from pathlib import Path
import re
import textwrap
import yaml


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


def ordinal(n: int) -> str:
    suffix = (
        "th" if 10 <= n % 100 <= 13 else
        "st" if n % 10 == 1 else
        "nd" if n % 10 == 2 else
        "rd" if n % 10 == 3 else
        "th"
    )

    return f"{n}{suffix}"


class Struct:
    def __init__(self, data: dict[str, Any]):
        self.__dict__.update(data)

    def __repr__(self) -> str:
        return str(self.__dict__)

def to_camel(literal: str) -> str:
    return "".join([
        word[1:-1].upper() if word[0] == "*" else word.capitalize()
        for word in literal.split("_")
    ])


def to_snake(literal: str) -> str:
    snake = re.sub(r"(?<=[a-z])([A-Z])", r'_\1', literal)
    snake = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", snake)
    return snake.lower()


def optional(value: Any, other: Any) -> Any:
    return other if value is None else value


def load_text(src: Path,
              input_variables: dict[str, str],
              ) -> tuple[str, list[str]]:
    deserial = load_json_yaml(src)

    content = deserial.pop("content", "")
    required_variables = deserial.pop("input_variables", [])

    variables = {}
    missings = []

    for key in required_variables:
        value = input_variables.get(key)

        if value is None:
            value = ""
            missings.append()

        variables[key] = value

    result = content.format(**variables)
    return result, missings