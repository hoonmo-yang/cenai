import logging

logger = logging.getLogger("cenai")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def DEBUG(mesg: str, *args, **kwargs) -> None:
    logger.debug(mesg, *args, stacklevel=2, **kwargs)


def INFO(mesg: str, *args, **kwargs) -> None:
    logger.info(mesg, *args, stacklevel=2, **kwargs)


def WARNING(mesg: str, *args, **kwargs) -> None:
    logger.warning(mesg, *args, stacklevel=2, **kwargs)


def ERROR(mesg: str, *args, **kwargs) -> None:
    logger.error(mesg, *args, stacklevel=2, **kwargs)


def CRITICAL(mesg: str, *args, **kwargs) -> None:
    logger.critical(mesg, *args, stacklevel=2, **kwargs)
