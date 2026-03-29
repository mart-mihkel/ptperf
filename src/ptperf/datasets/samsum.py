from datasets.dataset_dict import DatasetDict
from transformers import PreTrainedTokenizerFast


def load_samsum(tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
    raise NotImplementedError("samsum")
