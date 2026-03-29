from datasets.dataset_dict import DatasetDict
from transformers import PreTrainedTokenizerFast


def load_wikitext(tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
    raise NotImplementedError("wikitext")
