from collections.abc import Callable
from typing import Literal

from typer import Typer

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


@app.callback()
def callback(log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"):
    from ptperf.logging import logger

    logger.setLevel(log_level)


@app.command()
@timed
def main():
    from ptperf.logging import logger

    logger.info("i'm not impressed by your performance")


if __name__ == "__main__":
    app()
