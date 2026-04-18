import logging

import accelerate
import datasets
import httpx
import numpy
import torch
import transformers
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

_suppress = [
    transformers,
    accelerate,
    datasets,
    torch,
    httpx,
    numpy,
]

_console = Console(width=80)

_handler = RichHandler(
    show_path=False,
    console=_console,
    rich_tracebacks=True,
    tracebacks_show_locals=True,
    tracebacks_suppress=_suppress,
)

install(show_locals=True, console=_console, suppress=_suppress)

logging.basicConfig(format="%(message)s", handlers=[_handler])

_tf_logger = logging.getLogger("transformers")
_tf_logger.handlers = [_handler]
_tf_logger.propagate = True

logger = logging.getLogger("icftsc")
