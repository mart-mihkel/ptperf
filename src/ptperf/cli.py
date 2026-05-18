from __future__ import annotations

from typing import Annotated, Literal

from typer import Option, Typer

from ptperf.types import Method, Task

app = Typer(no_args_is_help=True)


@app.command(no_args_is_help=True, help="Run experiments")
def main(
    model: Annotated[str, Option(help="HuggingFace model or path to checkpoint")],
    task: Annotated[Task.__value__, Option(help="Type of NLP task")],
    method: Annotated[Method.__value__, Option(help="Fine tuning method")],
    lora_rank: Annotated[int | None, Option(help="LoRA attention dimension")] = None,
    lora_alpha: Annotated[int | None, Option(help="LoRA scaling parameter")] = None,
    virtual_tokens: Annotated[
        int | None, Option(help="Number of virtual tokens for soft prompt methods")
    ] = None,
    encoder_dim: Annotated[
        int | None, Option(help="Prompt encoder hidden dimension for p-tuning")
    ] = None,
    experiment: Annotated[str, Option(help="Experiment name for tracking")] = "ptperf",
    run_name: Annotated[
        str | None,
        Option(help="Run name for tracking, inferred from parameters by default"),
    ] = None,
    tracking_uri: Annotated[
        str,
        Option(help="Experiment tracking host or path", envvar="MLFLOW_TRACKING_URI"),
    ] = "sqlite:///mlflow.db",
    max_steps: Annotated[int, Option(help="Number of training steps")] = 1024,
    batch_size: int = 8,
    grad_chkpt: Annotated[bool, Option(help="Gradient checkpointing")] = False,
    log_level: Literal["debug", "info", "warning", "error"] = "info",
) -> None:
    import mlflow

    from ptperf.logging import logger
    from ptperf.modeling import fine_tune

    logger.setLevel(log_level.upper())

    if (lora_rank is None or lora_alpha is None) and method == "lora":
        raise ValueError("`lora_rank` nad `lora_alpha` must be set for lora")

    if virtual_tokens is None and method in ["prefix-tune", "prompt-tune", "p-tune"]:
        raise ValueError("`virtual_tokens` must be set for soft prompt methods")

    if encoder_dim is None and method == "p-tune":
        raise ValueError("`encoder_dim` must be set for p-tuning")

    lora_alpha = lora_alpha or 0
    lora_rank = lora_rank or 0
    virtual_tokens = virtual_tokens or 0
    encoder_dim = encoder_dim or 0

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
    mlflow.log_param("lora_alpha", lora_alpha)
    mlflow.log_param("lora_rank", lora_rank)
    mlflow.log_param("virtual_tokens", virtual_tokens)
    mlflow.log_param("encoder_dim", encoder_dim)

    fine_tune(
        model_path=model,
        task=task,
        method=method,
        run_name=run_name,
        lora_alpha=lora_alpha,
        lora_rank=lora_rank,
        virtual_tokens=virtual_tokens,
        encoder_dim=encoder_dim,
        max_steps=max_steps,
        batch_size=batch_size,
        grad_chkpt=grad_chkpt,
        tracking=True,
    )

    mlflow.end_run()


if __name__ == "__main__":
    app()
