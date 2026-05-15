from typing import cast

import mlflow
import torch
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)
from torch.utils.data import Dataset
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
)

from ptperf.callbacks import HWMetricsCallback
from ptperf.datasets import load_data
from ptperf.logging import logger
from ptperf.types import Method, Task


def fine_tune(
    model_path: str,
    task: Task,
    method: Method,
    run_name: str,
    num_virtual_tokens: int,
    max_steps: int,
    batch_size: int,
    grad_chkpt: bool = False,
    tracking: bool = False,
) -> None:
    logger.debug('load "%s" config', model_path)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = _load_tokenizer(model_path)

    logger.info("prepare model")
    model = _load_base_model(model_path, task)
    model = _prepare_model(model, task, method, num_virtual_tokens)
    _log_params(model, tracking)

    logger.info("prepare data")
    data = load_data(tokenizer, config, task, num_virtual_tokens)

    collator = _load_collator(tokenizer, task)
    args = _get_training_args(
        run_name=run_name,
        max_steps=max_steps,
        batch_size=batch_size,
        grad_chkpt=grad_chkpt,
        tracking=tracking,
    )

    callback = HWMetricsCallback(tracking=tracking)
    trainer = Trainer(
        args=args,
        model=model,
        callbacks=[callback],
        data_collator=collator,
        train_dataset=data["train"],
    )

    logger.info("start trainer")
    trainer.train()

    logger.info("start inference")
    callback.reset_cuda_stats()
    callback.phase = "inference"
    trainer.evaluate(cast(Dataset, data["train"]), metric_key_prefix="inference")


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

    model = cast(PreTrainedModel, model)
    if model.config.pad_token_id is None:
        logger.warning("model doesn't have a padding token, using eos")
        model.config.pad_token_id = model.config.eos_token_id

    return model


def _load_collator(tokenizer: PreTrainedTokenizerFast, task: Task) -> DataCollator:
    logger.debug("prepare data collator for %s", task)

    if task == "causal-lm":
        return DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

    if task == "seq2seq":
        return DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)


def _prepare_model(
    model: PreTrainedModel,
    task: Task,
    method: Method,
    num_virtual_tokens: int,
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

    if method == "prompt-tune":
        logger.info("prepare prompt tuning model")
        peft_config = PromptTuningConfig(
            task_type=task_type,
            num_virtual_tokens=num_virtual_tokens,
        )

        return get_peft_model(model, peft_config)

    if method == "p-tune":
        logger.info("prepare prompt tuning model")
        peft_config = PromptEncoderConfig(
            task_type=task_type,
            num_virtual_tokens=num_virtual_tokens,
        )

        return get_peft_model(model, peft_config)


def _get_training_args(
    run_name: str,
    max_steps: int,
    batch_size: int,
    grad_chkpt: bool,
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
        eval_strategy="no",
        logging_steps=max(1, max_steps // 10),
        num_train_epochs=0,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        use_cpu=not have_cuda,
        bf16_full_eval=have_cuda,
        bf16=have_cuda,
        optim=optim,
        gradient_checkpointing=grad_chkpt,
        gradient_checkpointing_kwargs=grad_chkpt_kwargs,
        full_determinism=True,
    )


def _get_peft_task(task: Task) -> TaskType:
    if task == "causal-lm":
        return TaskType.CAUSAL_LM

    if task == "seq2seq":
        return TaskType.SEQ_2_SEQ_LM


def _log_params(model: PreTrainedModel, tracking: bool = False) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total

    logger.debug("total parameters %d", total)
    logger.debug("trainable parameters %d", trainable)
    logger.debug("trainable to total ratio %.2f", ratio)

    if not tracking:
        return

    mlflow.log_metric("total_parameters", total)
    mlflow.log_metric("trainable_parameters", trainable)
    if torch.cuda.is_available():
        mlflow.log_metric("gpu_device_count", torch.cuda.device_count())
        mlflow.log_metric("gpu_current_device", torch.cuda.current_device())
