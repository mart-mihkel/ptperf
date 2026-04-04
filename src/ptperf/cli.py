from __future__ import annotations

from typing import Annotated, Literal

from typer import Option, Typer

from ptperf.types import Method, Task

app = Typer(no_args_is_help=True)


@app.command(no_args_is_help=True)
def main(
    model_path: Annotated[str, Option()],
    task: Annotated[Task.__value__, Option()],
    method: Annotated[Method.__value__, Option()],
    run_name: Annotated[str, Option()],
    epochs: Annotated[int, Option()] = 3,
    batch_size: Annotated[int, Option()] = 8,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
) -> None:
    import os
    import time

    import mlflow

    from ptperf.logging import logger
    from ptperf.scripts import fine_tune

    logger.setLevel(log_level)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        tracking_uri = "sqlite:///mlflow.db"
        logger.warning("MLFLOW_TRACKING_URI is unset")

    logger.info("tracking run %s of experiment ptperf at %s", run_name, tracking_uri)

    mlflow.set_experiment("ptperf")
    mlflow.start_run(run_name=run_name)
    mlflow.log_param("task", task)
    mlflow.log_param("method", method)
    mlflow.log_param("model", model_path)

    start = time.time()

    if method == "fine-tune":
        fine_tune(model_path, task, run_name, epochs, batch_size)
    else:
        raise NotImplementedError(f"Method: {method}")

    elapsed = time.time() - start

    mlflow.end_run(run_name)

    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info("time elapsed %02d:%02d:%02d", hours, minutes, seconds)


if __name__ == "__main__":
    app()
