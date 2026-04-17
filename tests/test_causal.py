from unittest import skip

from datasets.dataset_dict import DatasetDict
from peft import LoraModel, PeftModelForCausalLM
from transformers import GPT2Model, default_data_collator


def test_gpt2_wikitext(gpt2_wikitext: DatasetDict) -> None:
    assert set(gpt2_wikitext.keys()) == {"train", "validation", "test"}
    for split in gpt2_wikitext.values():
        assert set(split.column_names) == {"input_ids", "attention_mask", "labels"}
        assert split.num_rows > 0


def test_gpt2_wikitext_forward_pass(
    gpt2: GPT2Model,
    gpt2_wikitext: DatasetDict,
) -> None:
    examples = [gpt2_wikitext["test"][:4]]

    batch = default_data_collator(examples)
    out = gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_lora_wikitext_forward_pass(
    gpt2_lora: LoraModel,
    gpt2_wikitext: DatasetDict,
) -> None:
    examples = [gpt2_wikitext["test"][:4]]

    batch = default_data_collator(examples)
    out = gpt2_lora(**batch)

    assert out.loss is not None
    assert out.logits is not None


@skip("broken")  # FIXME: Tensors must have same number of dimensions: got 2 and 3
def test_gpt2_prefix_wikitext_forward_pass(
    gpt2_prefix: PeftModelForCausalLM,
    gpt2_wikitext: DatasetDict,
) -> None:
    examples = [gpt2_wikitext["test"][:4]]

    batch = default_data_collator(examples)
    out = gpt2_prefix(**batch)

    assert out.loss is not None
    assert out.logits is not None
