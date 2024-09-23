import dotenv
import os
from pathlib import Path


def cenai_path(*args) -> Path:
    return Path(os.environ["CENAI_DIR"]).joinpath(*args)


def load_dotenv() -> None:
    dotenv_path = cenai_path("cf/.env")
    dotenv.load_dotenv(dotenv_path)
