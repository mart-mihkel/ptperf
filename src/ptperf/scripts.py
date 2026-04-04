import os
from typing import cast

import torch
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

from ptperf.datasets import load_data
from ptperf.logging import logger
from ptperf.types import Task


def _load_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    logger.debug("load pretrained tokenizer for %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)

    if tokenizer.pad_token is None:
        logger.warning("tokenizer doesn't have a padding token, using eos")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def _load_model(model_path: str, task: Task) -> PreTrainedModel:
    logger.debug("load pretrained %s for %s", model_path, task)

    if task == "causal-lm":
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif task == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
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
        return DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

    if task == "seq2seq":
        return DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    raise NotImplementedError(f"Collator for task: {task}")


def _get_training_args(
    run_name: str,
    epochs: int,
    batch_size: int,
) -> TrainingArguments:
    have_cuda = torch.cuda.is_available()
    logdir = os.path.join("log", run_name)

    if have_cuda:
        logger.debug("have accelerator")
        logger.debug("using brain floating point 16")
        optim = "adamw_8bit"
    if not have_cuda:
        logger.warning("no accelerator")
        logger.debug("using full precision floating point")
        optim = "adamw_torch_fused"

    logger.debug("logdir %s", logdir)
    logger.debug("using %s optimizer", optim)

    return TrainingArguments(
        report_to="mlflow",
        run_name=run_name,
        output_dir=logdir,
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
    )


def fine_tune(
    model_path: str,
    task: Task,
    run_name: str,
    epochs: int,
    batch_size: int,
) -> None:
    logger.debug("load %s config", model_path)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = _load_tokenizer(model_path)
    data = load_data(tokenizer, config, task)
    model = _load_model(model_path, task)
    collator = _load_collator(tokenizer, task)
    args = _get_training_args(run_name, epochs, batch_size)

    trainer = Trainer(
        args=args,
        model=model,
        data_collator=collator,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
    )

    trainer.train()
    trainer.save_model()
