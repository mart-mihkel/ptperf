from typing import TypedDict

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedConfig, PreTrainedTokenizerFast

from ptperf.logging import logger
from ptperf.types import Task


class SamSumExample(TypedDict):
    id: int
    dialogue: str
    summary: str


class WikiTextExample(TypedDict):
    text: str


def load_data(
    tokenizer: PreTrainedTokenizerFast,
    config: PreTrainedConfig,
    task: Task,
    split: Split | None = None,
) -> DatasetDict:
    logger.debug("prepare dataset for %s", task)

    if task == "causal-lm":
        return load_wikitext(tokenizer, config, split)

    if task == "seq2seq":
        return load_samsum(tokenizer, split)

    raise NotImplementedError(f"Dataset for task: {task}")


def load_samsum(
    tokenizer: PreTrainedTokenizerFast,
    split: Split | None = None,
) -> DatasetDict:
    logger.debug("load samsum")
    raw = load_dataset("knkarthick/samsum", split=split)

    cols = ["id", "dialogue", "summary"]
    fn_kwargs = {"tokenizer": tokenizer}
    data = raw.map(_tokenize_samsum, remove_columns=cols, fn_kwargs=fn_kwargs)

    return data


def _tokenize_samsum(
    example: SamSumExample,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    prompt = tokenizer(example["dialogue"], truncation=True)
    answer = tokenizer(example["summary"], truncation=True)
    prompt["labels"] = answer["input_ids"]
    return prompt


def load_wikitext(
    tokenizer: PreTrainedTokenizerFast,
    config: PreTrainedConfig,
    split: Split | None = None,
) -> DatasetDict:
    logger.debug("load wikitext")
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)

    logger.debug("filter empty sequences")
    raw = raw.filter(lambda example: len(example["text"].strip()) > 0)

    cols = ["text"]
    fn_kwargs = {"tokenizer": tokenizer, "config": config}
    data = raw.map(_tokenize_wikitext, remove_columns=cols, fn_kwargs=fn_kwargs)

    return data


def _tokenize_wikitext(
    example: WikiTextExample,
    tokenizer: PreTrainedTokenizerFast,
    config: PreTrainedConfig,
) -> BatchEncoding:
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=config.max_position_embeddings,
    )
