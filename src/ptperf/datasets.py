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
    num_virtual_tokens: int = 0,
    split: Split | None = None,
) -> DatasetDict:
    logger.debug("prepare dataset for %s", task)

    if task == "causal-lm":
        return load_wikitext(tokenizer, config, num_virtual_tokens, split)

    if task == "seq2seq":
        return load_samsum(tokenizer, split)


def load_samsum(
    tokenizer: PreTrainedTokenizerFast,
    split: Split | None = None,
) -> DatasetDict:
    logger.debug('load "knkarthick/samsum"')
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
    num_virtual_tokens: int,
    split: Split | None = None,
) -> DatasetDict:
    logger.debug('load "Salesforce/wikitext" subtask "wikitext-2-raw-v1"')
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)

    cols = ["text"]
    max_len = config.max_position_embeddings - num_virtual_tokens
    fn_kwargs = {"tokenizer": tokenizer, "max_len": max_len}
    return raw.map(_tokenize_wikitext, remove_columns=cols, fn_kwargs=fn_kwargs)


def _tokenize_wikitext(
    example: WikiTextExample,
    tokenizer: PreTrainedTokenizerFast,
    max_len: int,
) -> BatchEncoding:
    return tokenizer(example["text"], truncation=True, max_length=max_len)
