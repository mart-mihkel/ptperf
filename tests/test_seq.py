from typing import cast

from datasets.dataset_dict import DatasetDict
from datasets.splits import Split
from pytest import fixture
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    T5Config,
    T5Model,
    T5Tokenizer,
)

from ptperf.datasets import load_data

_t5 = "hf-internal-testing/tiny-random-t5"


@fixture(scope="session")
def t5() -> T5Model:
    model = AutoModelForSeq2SeqLM.from_pretrained(_t5)
    return cast(T5Model, model)


@fixture(scope="session")
def t5_config() -> T5Config:
    config = AutoConfig.from_pretrained(_t5)
    return cast(T5Config, config)


@fixture(scope="session")
def t5_tokenizer() -> T5Tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(_t5)
    return cast(T5Tokenizer, tokenizer)


@fixture(scope="session")
def t5_samsum(
    t5_config: T5Config,
    t5_tokenizer: T5Tokenizer,
) -> DatasetDict:
    split = {
        "train": "train[:10]",
        "validation": "validation[:10]",
        "test": "test[:10]",
    }

    split = cast(Split, split)
    return load_data(t5_tokenizer, t5_config, "seq2seq", split)


def test_t5_samsum(t5_samsum: DatasetDict) -> None:
    assert set(t5_samsum.keys()) == {"train", "validation", "test"}
    for split in t5_samsum.values():
        assert set(split.column_names) == {"input_ids", "attention_mask", "labels"}
        assert split.num_rows > 0


def test_t5_samsum_forward_pass(
    t5: T5Model,
    t5_tokenizer: T5Tokenizer,
    t5_samsum: DatasetDict,
) -> None:
    examples = [t5_samsum["test"][i] for i in range(4)]
    collator = DataCollatorForSeq2Seq(t5_tokenizer)

    batch = collator(examples)
    out = t5(**batch)

    assert out.loss is not None
    assert out.logits is not None
