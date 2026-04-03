from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

from typer import Option, Typer

from ptperf.types import Task

app = Typer(no_args_is_help=True)


def timed(func: Callable) -> Callable:
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        from ptperf.logging import logger

        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info("time elapsed %02d:%02d:%02d", hours, minutes, seconds)

        return result

    return wrapper


@app.command()
@timed
def main(
    model_path: Annotated[str, Option()],
    task: Annotated[Task.__value__, Option()],
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    from ptperf.logging import logger
    from ptperf.scripts import fine_tune

    logger.setLevel(log_level)
    fine_tune(model_path, task)


if __name__ == "__main__":
    app()
