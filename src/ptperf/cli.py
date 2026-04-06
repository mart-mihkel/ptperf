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
    experiment: Annotated[str, Option(help="Experiment name for tracking")] = "ptperf",
    run_name: Annotated[
        str | None,
        Option(help="Run name for tracking, inferred from parameters by default"),
    ] = None,
    epochs: int = 3,
    batch_size: int = 8,
    grad_chkpt: bool = False,
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

    if run_name is None:
        run_name = f"{model}/{task}/{method}"
        logger.debug('inferred run name "%s"', run_name)

    logger.info(
        'tracking run "%s" of experiment "%s" at %s',
        run_name,
        experiment,
        tracking_uri,
    )

    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    mlflow.log_param("task", task)
    mlflow.log_param("model", model)
    mlflow.log_param("method", method)

    fine_tune(
        model_path=model,
        task=task,
        method=method,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        grad_chkpt=grad_chkpt,
        tracking=True,
    )

    mlflow.end_run()


if __name__ == "__main__":
    app()
