from __future__ import annotations

from typing import Annotated, Literal

from typer import Option, Typer

from ptperf.types import Method, Task

app = Typer(no_args_is_help=True)


@app.command(no_args_is_help=True)
def main(
    model: Annotated[str, Option(help="HuggingFace model or path to checkpoint")],
    task: Annotated[Task.__value__, Option(help="Type of NLP task")],
    method: Annotated[Method.__value__, Option(help="Fine tuning method")],
    run_name: Annotated[str, Option(help="Run name for experiment tracking")],
    epochs: int = 3,
    batch_size: int = 8,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
) -> None:
    import os

    import mlflow

    from ptperf.logging import logger
    from ptperf.scripts import fine_tune

    logger.setLevel(log_level)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        tracking_uri = "sqlite:///mlflow.db"
        logger.warning("MLFLOW_TRACKING_URI is unset")

    logger.info(
        'tracking run "%s" of experiment "ptperf" at %s',
        run_name,
        tracking_uri,
    )

    mlflow.set_experiment("ptperf")
    mlflow.start_run(run_name=run_name)
    mlflow.log_param("task", task)
    mlflow.log_param("method", method)
    mlflow.log_param("model", model)

    fine_tune(model, task, method, run_name, epochs, batch_size, tracking=True)

    mlflow.end_run()


if __name__ == "__main__":
    app()
