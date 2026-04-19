from typing import Literal

import mlflow
import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ptperf.logging import logger


class HWMetricsCallback(TrainerCallback):
    tracking: bool
    phase: Literal["train", "inference"]

    def __init__(self, tracking: bool = False) -> None:
        self.tracking = tracking
        self.phase = "train"

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: dict,
    ) -> None:
        _ = args, state, control, kwargs
        self.reset_cuda_stats()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: dict,
    ) -> None:
        _ = args, control, kwargs
        if not torch.cuda.is_available():
            return

        if not self.tracking:
            return

        mem_stats = torch.cuda.memory_stats()
        metrics = {f"{self.phase}.{k}": v for k, v in mem_stats.items()}
        mlflow.log_metrics(metrics, step=state.global_step)

    @staticmethod
    def reset_cuda_stats() -> None:
        logger.debug("reset cuda stats")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
