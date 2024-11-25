import argparse
import os
from pathlib import Path
import sys

from cenai_core.logger import Logger
from cenai_core.dataman import Q
from cenai_core.grid.runner import GridRunner


class GridCLI(Logger):
    profile_dir = Path("profile")
    logger_name = "cenai.system"

    def __init__(self,
                 title: str,
                 argvs: list[str] = [],
                 environ: dict[str, str] = {}
                 ):

        super().__init__()

        os.environ |= environ

        if not argvs:
            argvs = sys.argv[1:]

        self._title = title
        self._option = self._get_option(argvs)

    def _get_option(self,
                    argvs: list[str]
                    ) -> argparse.Namespace:

        parser = argparse.ArgumentParser(
            description=f"{self.title}"
        )

        parser.add_argument(
            "profiles",
            nargs="*",
            default=[],
            help="profile files"
        )

        return parser.parse_args(argvs)

    def _choose_profiles(self) -> list[Path]:
        profiles = [
            self.profile_dir / f"{profile}.yaml"
            for profile in self.option.profiles
        ]

        if profiles:
            return profiles

        profiles = list(self.profile_dir.glob("*.yaml"))

        if not profiles:
            raise RuntimeError(
                f"no profile file in {Q(self.profile_dir)}"
            )

        items = [
            (f"[{i + 1}] {profile.stem}")
            for i, profile in enumerate(profiles)
        ]

        while True:
            answer = input(
                f"\n{'\n'.join(items)}\n\n"
                f"Choose a profile yaml for {Q(self.title)} "
                "by number (q for exit): "
            )

            answer = answer.strip()

            if answer.lower() == "q":
                return []

            if answer.isdigit():
                k = int(answer) - 1
                if k < len(items):
                    profile = profile[k]
                    break

            self.ERROR(f"\nwrong selection - {Q(answer)}\n")

        return [profile]

    def __call__(self) -> None:
        self.invoke()

    def invoke(self) -> None:
        profiles = self._choose_profiles()

        for profile in profiles:
            self.INFO(f"PROFILE {Q(profile)} for {Q(self.title)} proceed ....")

            runner = GridRunner(profile)

            runner()
            runner.save()
            runner.export()

            self.INFO(f"PROFILE {Q(profile)} for {Q(self.title)} proceed DONE")

        self.INFO("bye")

    @property
    def title(self) -> str:
        return self._title

    @property
    def option(self) -> argparse.Namespace:
        return self._option
