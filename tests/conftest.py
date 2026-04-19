from typing import cast

from datasets.dataset_dict import DatasetDict
from datasets.splits import Split
from peft import LoraModel, PeftModelForCausalLM, PeftModelForSeq2SeqLM
from pytest import fixture
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
    T5Config,
    T5Model,
    T5Tokenizer,
)

from ptperf.datasets import load_data
from ptperf.scripts import _prepare_model

_gpt2 = "hf-internal-testing/tiny-random-gpt2"
_t5 = "hf-internal-testing/tiny-random-t5"


@fixture(scope="session")
def gpt2() -> GPT2Model:
    model = AutoModelForCausalLM.from_pretrained(_gpt2)
    model.config.pad_token_id = model.config.eos_token_id
    return cast(GPT2Model, model)


@fixture(scope="session")
def gpt2_lora() -> LoraModel:
    model = AutoModelForCausalLM.from_pretrained(_gpt2)
    model.config.pad_token_id = model.config.eos_token_id
    model = _prepare_model(model, "causal-lm", "lora", 0)
    return cast(LoraModel, model)


@fixture(scope="session")
def gpt2_prefix() -> PeftModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(_gpt2)
    model.config.pad_token_id = model.config.eos_token_id
    model = _prepare_model(model, "causal-lm", "prefix-tune", 10)
    return cast(PeftModelForCausalLM, model)


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
    return load_data(gpt2_tokenizer, gpt2_config, "causal-lm", 10, split=split)


@fixture(scope="session")
def t5() -> T5Model:
    model = AutoModelForSeq2SeqLM.from_pretrained(_t5)
    return cast(T5Model, model)


@fixture(scope="session")
def t5_lora() -> LoraModel:
    model = AutoModelForSeq2SeqLM.from_pretrained(_t5)
    model = _prepare_model(model, "seq2seq", "lora", 0)
    return cast(LoraModel, model)


@fixture(scope="session")
def t5_prefix() -> PeftModelForSeq2SeqLM:
    model = AutoModelForSeq2SeqLM.from_pretrained(_t5)
    model = _prepare_model(model, "seq2seq", "prefix-tune", 10)
    return cast(PeftModelForSeq2SeqLM, model)


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
    return load_data(t5_tokenizer, t5_config, "seq2seq", 10, split=split)
