from itertools import chain
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

    raise NotImplementedError(f"Dataset for task: {task}")


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

    logger.debug("filter empty sequences")
    raw = raw.filter(lambda example: len(example["text"].strip()) > 0)

    block_size = min(512, config.max_position_embeddings - num_virtual_tokens)

    cols = ["text"]
    fn_kwargs = {"tokenizer": tokenizer}

    tokenized = raw.map(
        _tokenize_wikitext,
        remove_columns=cols,
        fn_kwargs=fn_kwargs,
        batched=True,
    )

    data = tokenized.map(
        _chunk_wikitext,
        batched=True,
        fn_kwargs={"block_size": block_size},
    )

    return data


def _tokenize_wikitext(
    examples: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    return tokenizer(examples["text"])


def _chunk_wikitext(
    examples: dict[str, list],
    block_size: int,
) -> dict[str, list]:
    # concatenate all token ids into one long sequence
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
    total_length = len(concatenated_examples[next(iter(examples.keys()))])
    # drop the last incomplete block
    total_length = (total_length // block_size) * block_size

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
