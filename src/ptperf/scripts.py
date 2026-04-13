import time
from typing import cast

import mlflow
import torch
from peft import LoraConfig, PrefixTuningConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator
)

from ptperf.datasets import load_data
from ptperf.logging import logger
from ptperf.types import Method, Task


def fine_tune(
    model_path: str,
    task: Task,
    method: Method,
    run_name: str,
    num_virtual_tokens: int,
    epochs: int = 1,
    batch_size: int = 8,
    grad_chkpt: bool = False,
    tracking: bool = False,
) -> None:
    logger.debug('load "%s" config', model_path)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = _load_tokenizer(model_path)

    logger.info("prepare model")
    model = _load_base_model(model_path, task)
    model = _prepare_model(model, task, method)
    _log_params(model, tracking)

    logger.info("prepare data")
    data = load_data(tokenizer, config, task, num_virtual_tokens)

    collator = _load_collator(tokenizer, task)
    args = _get_training_args(run_name, epochs, batch_size, grad_chkpt, tracking)

    trainer = Trainer(
        args=args,
        model=model,
        data_collator=collator,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logger.info("start trainer")
    t0 = time.perf_counter()
    trainer.train()
    wall_time = time.perf_counter() - t0

    _log_hardware_metrics(trainer, wall_time, tracking)


def _load_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    logger.debug('load tokenizer for "%s"', model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)

    if tokenizer.pad_token is None:
        logger.warning("tokenizer doesn't have a padding token, using eos")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def _load_base_model(model_path: str, task: Task) -> PreTrainedModel:
    logger.debug('load "%s" for %s', model_path, task)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if task == "causal-lm":
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    elif task == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=dtype)
    else:
        raise NotImplementedError(f"Model for task: {task}")

    model = cast(PreTrainedModel, model)
    if model.config.pad_token_id is None:
        logger.warning("model doesn't have a padding token, using eos")
        model.config.pad_token_id = model.config.eos_token_id

    return model


def _load_collator(tokenizer: PreTrainedTokenizerFast, task: Task) -> DataCollator:
    logger.debug("prepare data collator for %s", task)

    if task == "causal-lm":
        return default_data_collator
    if task == "seq2seq":
        return DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    raise NotImplementedError(f"Collator for task: {task}")


def _prepare_model(
    model: PreTrainedModel,
    task: Task,
    method: Method,
    num_virtual_tokens: int = 0,
) -> PreTrainedModel:
    if method == "fine-tune":
        return model

    task_type = _get_peft_task(task)

    if method == "lora":
        logger.info("prepare lora model")
        peft_config = LoraConfig(task_type=task_type)
        return get_peft_model(model, peft_config)

    if method == "prefix-tune":
        logger.info("prepare prefix tuning model")
        peft_config = PrefixTuningConfig(
            task_type=task_type,
            num_virtual_tokens=num_virtual_tokens,
        )

        return get_peft_model(model, peft_config)

    raise NotImplementedError(f"Method: {method}")


def _get_training_args(
    run_name: str,
    epochs: int,
    batch_size: int,
    grad_chkpt: bool = False,
    tracking: bool = False,
) -> TrainingArguments:
    report_to = "mlflow" if tracking else "none"
    have_cuda = torch.cuda.is_available()

    if have_cuda:
        logger.debug("have accelerator")
        logger.debug("using brain floating point 16")
        optim = "adamw_8bit"
    else:
        logger.warning("no accelerator available")
        logger.debug("using full precision floating point")
        optim = "adamw_torch_fused"

    if grad_chkpt:
        grad_chkpt_kwargs = {"use_reentrant": False}
        logger.debug("using gradient checkpointing with %s", grad_chkpt_kwargs)
    else:
        grad_chkpt_kwargs = None
        logger.debug("not using gradient checkpointing")

    logger.debug('using "%s" optimizer', optim)

    return TrainingArguments(
        output_dir="log",
        report_to=report_to,
        run_name=run_name,
        save_strategy="no",
        eval_strategy="epoch",
        logging_steps=500,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        use_cpu=not have_cuda,
        bf16_full_eval=have_cuda,
        bf16=have_cuda,
        optim=optim,
        gradient_checkpointing=grad_chkpt,
        gradient_checkpointing_kwargs=grad_chkpt_kwargs,
    )


def _get_peft_task(task: Task) -> TaskType:
    if task == "causal-lm":
        return TaskType.CAUSAL_LM

    if task == "seq2seq":
        return TaskType.SEQ_2_SEQ_LM

    raise NotImplementedError(f"PEFT task type for {task}")


def _log_params(model: PreTrainedModel, tracking: bool = False) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug("total parameters %d", total)
    logger.debug("trainable parameters %d", trainable)
    logger.debug("trainable to total ratio %.2f", trainable / total)

    if tracking:
        mlflow.log_metric("total_parameters", total)
        mlflow.log_metric("trainable_parameters", trainable)

def _log_hardware_metrics(
    trainer: Trainer,
    wall_time: float,
    tracking: bool = False,
) -> None:
    metrics: dict[str, float] = {"train_wall_time_s": wall_time}

    if torch.cuda.is_available():
        metrics["peak_gpu_memory_allocated_mb"] = (
            torch.cuda.max_memory_allocated() / 1024**2
        )
        metrics["peak_gpu_memory_reserved_mb"] = (
            torch.cuda.max_memory_reserved() / 1024**2
        )

    log_history = trainer.state.log_history
    train_logs = [e for e in log_history if "train_samples_per_second" in e]
    if train_logs:
        metrics["train_samples_per_second"] = train_logs[-1]["train_samples_per_second"]

    for k, v in metrics.items():
        logger.info("%s: %.4f", k, v)
        if tracking:
            mlflow.log_metric(k, v)
