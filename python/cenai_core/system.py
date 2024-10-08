from datetime import timedelta
import dotenv
import os
from pathlib import Path
from timeit import default_timer


def cenai_path(*args) -> Path:
    return Path(os.environ["CENAI_DIR"]).joinpath(*args)


def load_dotenv() -> None:
    dotenv_path = cenai_path("cf/.env")
    dotenv.load_dotenv(dotenv_path)


class Timer:
    def __init__(self):
        self.start()

    def start(self) -> None:
        self.current = default_timer()
        self.previous = self.current

    def lap(self) -> None:
        self.current = default_timer()

    @property
    def seconds(self) -> float:
        return self.current - self.previous

    @property
    def delta(self) -> timedelta:
        return timedelta(seconds=self.seconds)