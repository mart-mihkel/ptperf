from __future__ import annotations

from typing import Annotated, Literal

from typer import Option, Typer

app = Typer()


@app.command()
def main(
    experiment: Annotated[str, Option(help="Experiment name")] = "ptperf",
    tracking_uri: Annotated[
        str,
        Option(help="Experiment tracking host or path", envvar="MLFLOW_TRACKING_URI"),
    ] = "sqlite:///mlflow.db",
    log_level: Literal["debug", "info", "warning", "error"] = "info",
) -> None:
    import os
    from pathlib import Path

    from mlflow.tracking import MlflowClient
    from polars import DataFrame

    from ptperf.logging import logger

    logger.setLevel(log_level.upper())

    logger.info("connecting to %s", tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    logger.info("finding experiment %s", experiment)
    exp = client.get_experiment_by_name(experiment)

    if exp is None:
        raise RuntimeError(f"experiment '{experiment}' not found")

    logger.info("collecting metrics")
    runs = client.search_runs(exp.experiment_id, "")
    rows = []
    for run in runs:
        run_data = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }

        metrics = run.data.metrics
        for key, value in metrics.items():
            run_data[key] = value

        params = run.data.params
        for key, value in params.items():
            run_data[key] = value

        rows.append(run_data)

    logdir = Path("log")
    metricdir = logdir / "metrics"
    path = metricdir / f"{experiment}.csv"
    os.makedirs(metricdir, exist_ok=True)

    df = DataFrame(rows)
    df.write_csv(path)

    logger.info("found %d runs with %d params", df.shape[0], df.shape[1])
    logger.info("saved metrics to '%s'", path)


if __name__ == "__main__":
    app()
