from typing import cast

from datasets.dataset_dict import DatasetDict
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

from ptperf.datasets.samsum import load_samsum
from ptperf.datasets.wikitext import load_wikitext
from ptperf.logging import logger
from ptperf.types import DatasetName


def _load_data(
    dataset: DatasetName,
    tokenizer: PreTrainedTokenizerFast,
) -> DatasetDict:
    if dataset == "wikitext":
        return load_wikitext(tokenizer)

    if dataset == "samsum":
        return load_samsum(tokenizer)

    raise NotImplementedError(f"Dataset '{dataset}'")


def fine_tune(model_path: str, dataset: DatasetName):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)

    data = _load_data(dataset, tokenizer)
    logger.debug(data)

    model = AutoModel.from_pretrained(model_path)
    logger.debug(model)
