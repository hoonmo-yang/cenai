import sys


def eprint(mesg: str, *args, **kwargs) -> None:
    print(mesg, *args, **kwargs, file=sys.stderr)
