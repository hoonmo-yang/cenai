import string
from typing import Any, Hashable

import json
from pathlib import Path
import string
import yaml


def clean_text(text: str) -> str:
    text = "".join(text.split())
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    return text


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
