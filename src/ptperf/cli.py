from __future__ import annotations

from typing import Annotated, Literal

from typer import Option, Typer

from ptperf.types import Task

app = Typer(no_args_is_help=True)


@app.command(no_args_is_help=True)
def main(
    model_path: Annotated[str, Option()],
    task: Annotated[Task.__value__, Option()],
    epochs: Annotated[int, Option()] = 3,
    batch_size: Annotated[int, Option()] = 8,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
) -> None:
    import time

    from ptperf.logging import logger
    from ptperf.scripts import fine_tune

    logger.setLevel(log_level)

    start = time.time()
    run_name = "notimplemented"
    fine_tune(model_path, task, run_name, epochs, batch_size)
    elapsed = time.time() - start

    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info("time elapsed %02d:%02d:%02d", hours, minutes, seconds)


if __name__ == "__main__":
    app()
