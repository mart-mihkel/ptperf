from datasets.dataset_dict import DatasetDict
from peft import LoraModel, PeftModelForCausalLM
from transformers import (
    GPT2Model,
    GPT2Tokenizer,
    Trainer,
)

from ptperf.modeling import _get_training_args, _load_collator


def test_gpt2_wikitext(gpt2_wikitext: DatasetDict) -> None:
    for split in gpt2_wikitext.values():
        assert set(split.column_names) == {"input_ids", "attention_mask"}
        assert split.num_rows > 0


def test_gpt2_wikitext_forward_pass(
    gpt2: GPT2Model,
    gpt2_tokenizer: GPT2Tokenizer,
    gpt2_wikitext: DatasetDict,
) -> None:
    examples = [gpt2_wikitext["train"][i] for i in range(4)]
    collator = _load_collator(gpt2_tokenizer, "causal-lm")

    batch = collator(examples)
    out = gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_lora_wikitext_forward_pass(
    gpt2_lora: LoraModel,
    gpt2_tokenizer: GPT2Tokenizer,
    gpt2_wikitext: DatasetDict,
) -> None:
    examples = [gpt2_wikitext["train"][i] for i in range(4)]
    collator = _load_collator(gpt2_tokenizer, "causal-lm")

    batch = collator(examples)
    out = gpt2_lora(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_prefix_wikitext_forward_pass(
    gpt2_prefix: PeftModelForCausalLM,
    gpt2_tokenizer: GPT2Tokenizer,
    gpt2_wikitext: DatasetDict,
) -> None:
    examples = [gpt2_wikitext["train"][i] for i in range(4)]
    collator = _load_collator(gpt2_tokenizer, "causal-lm")

    batch = collator(examples)
    out = gpt2_prefix(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_prompt_wikitext_forward_pass(
    gpt2_prompt: PeftModelForCausalLM,
    gpt2_tokenizer: GPT2Tokenizer,
    gpt2_wikitext: DatasetDict,
) -> None:
    examples = [gpt2_wikitext["train"][i] for i in range(4)]
    collator = _load_collator(gpt2_tokenizer, "causal-lm")

    batch = collator(examples)
    out = gpt2_prompt(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_p_tune_wikitext_forward_pass(
    gpt2_p_tune: PeftModelForCausalLM,
    gpt2_tokenizer: GPT2Tokenizer,
    gpt2_wikitext: DatasetDict,
) -> None:
    examples = [gpt2_wikitext["train"][i] for i in range(4)]
    collator = _load_collator(gpt2_tokenizer, "causal-lm")

    batch = collator(examples)
    out = gpt2_p_tune(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_training_loop(
    gpt2_lora: LoraModel,
    gpt2_wikitext: DatasetDict,
    gpt2_tokenizer: GPT2Tokenizer,
) -> None:
    collator = _load_collator(gpt2_tokenizer, "causal-lm")
    args = _get_training_args("test", 4, 2, False)
    trainer = Trainer(
        args=args,
        model=gpt2_lora,
        data_collator=collator,
        train_dataset=gpt2_wikitext["train"],
    )

    trainer.train()
