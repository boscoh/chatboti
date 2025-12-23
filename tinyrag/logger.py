import logging
from typing import Union

from rich.console import Console
from rich.logging import RichHandler

_logging_configured = False


def setup_logging(
    level: Union[int, str] = logging.INFO,
) -> None:
    """
    Set up logging with Rich formatter and switching off noisy modules. Call
    as early as possible before AWS or other service calls. Safe to call multiple
    times - only configures once.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    if isinstance(level, str):
        level = getattr(logging, level.upper())
    rich_handler = RichHandler(
        console=Console(stderr=True),
        log_time_format="[%X]",
    )
    rich_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    root_logger.addHandler(rich_handler)

    logger_configs = [
        ("__main__", logging.INFO, True),
        ("boto3", logging.INFO, True),
        ("botocore", logging.WARNING, True),
        ("aioboto3", logging.INFO, True),
        ("aiobotocore", logging.WARNING, True),
        ("urllib3", logging.WARNING, True),
        ("httpx", logging.WARNING, True),
        ("httpcore", logging.WARNING, True),
        ("openai", logging.WARNING, True),
        ("h11", logging.WARNING, True),
        ("uvicorn", logging.INFO, True),
        ("uvicorn.access", logging.WARNING, True),
        ("uvicorn.error", logging.INFO, True),
        ("uvicorn.server", logging.INFO, True),
        ("mcp.server", logging.WARNING, True),
    ]

    for name, lvl, propagate in logger_configs:
        logger = logging.getLogger(name)
        if name != "":
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        logger.setLevel(lvl)
        logger.propagate = propagate
