from typing import cast

from datasets.dataset_dict import DatasetDict
from datasets.splits import Split
from pytest import fixture
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
)

from ptperf.datasets import load_data

_gpt2 = "hf-internal-testing/tiny-random-gpt2"


@fixture(scope="session")
def gpt2() -> GPT2Model:
    model = AutoModelForCausalLM.from_pretrained(_gpt2)
    model.config.pad_token_id = model.config.eos_token_id
    return cast(GPT2Model, model)


@fixture(scope="session")
def gpt2_config() -> GPT2Config:
    config = AutoConfig.from_pretrained(_gpt2)
    return cast(GPT2Config, config)


@fixture(scope="session")
def gpt2_tokenizer() -> GPT2Tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(_gpt2)
    tokenizer = cast(GPT2Tokenizer, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@fixture(scope="session")
def gpt2_wikitext(
    gpt2_config: GPT2Config,
    gpt2_tokenizer: GPT2Tokenizer,
) -> DatasetDict:
    split = {
        "train": "train[:10]",
        "validation": "validation[:10]",
        "test": "test[:10]",
    }

    split = cast(Split, split)
    return load_data(gpt2_tokenizer, gpt2_config, "causal-lm", split)


def test_gpt2_wikitext(gpt2_wikitext: DatasetDict) -> None:
    assert set(gpt2_wikitext.keys()) == {"train", "validation", "test"}
    for split in gpt2_wikitext.values():
        assert set(split.column_names) == {"input_ids", "attention_mask"}
        assert split.num_rows > 0


def test_gpt2_wikitext_forward_pass(
    gpt2: GPT2Model,
    gpt2_tokenizer: GPT2Tokenizer,
    gpt2_wikitext: DatasetDict,
) -> None:
    examples = [gpt2_wikitext["test"][i] for i in range(4)]
    collator = DataCollatorForLanguageModeling(gpt2_tokenizer, mlm=False)

    batch = collator(examples)
    out = gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None
    assert out.logits.shape[0] == batch["input_ids"].shape[0]
