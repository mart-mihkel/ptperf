from typing import cast

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from ptperf.datasets import load_data
from ptperf.logging import logger
from ptperf.types import Task


def fine_tune(model_path: str, task: Task):
    logger.debug("load pretrained model config of %s", model_path)
    config = AutoConfig.from_pretrained(model_path)

    logger.debug("load pretrained tokenizer for %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)

    if tokenizer.pad_token is None:
        logger.warning("tokenizer doesn't have a padding token, using eos")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.debug("prepare %s dataset", task)
    data = load_data(task, tokenizer, config)
    assert "train" in data, "No train data"
    assert "validation" in data, "No eval data"

    logger.debug("load pretrained %s model", model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if model.config.pad_token_id is None:
        logger.warning("model doesn't have a padding token, using eos")
        model.config.pad_token_id = model.config.eos_token_id

    have_cuda = torch.cuda.is_available()
    optim = "adamw_8bit"

    if not have_cuda:
        optim = "adamw_torch_fused"
        logger.warning("no accelerator")

    args = TrainingArguments(
        output_dir="out/test",
        save_strategy="no",
        eval_strategy="epoch",
        logging_steps=500,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        use_cpu=not have_cuda,
        bf16_full_eval=have_cuda,
        bf16=have_cuda,
        optim=optim,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        args=args,
        model=model,
        data_collator=collator,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
    )

    trainer.train()
    trainer.save_model()
