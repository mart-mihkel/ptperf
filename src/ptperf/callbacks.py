import mlflow
import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class HWMetricsCallback(TrainerCallback):
    def __init__(self, tracking: bool = False) -> None:
        self.tracking = tracking

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: dict,
    ) -> None:
        _ = args, state, control, kwargs
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: dict,
    ) -> None:
        _ = args, state, control, kwargs
        metrics: dict[str, float] = {}

        if not torch.cuda.is_available():
            return

        metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated()
        metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved()
        metrics["gpu_peak_memory_allocated"] = torch.cuda.max_memory_allocated()
        metrics["gpu_peak_memory_reserved"] = torch.cuda.max_memory_reserved()
        metrics = metrics | torch.cuda.memory_stats()

        if self.tracking:
            mlflow.log_metrics(metrics, step=state.global_step)
