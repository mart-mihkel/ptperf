from datasets.dataset_dict import DatasetDict
from peft import LoraModel, PeftModelForSeq2SeqLM
from transformers import (
    DataCollatorForSeq2Seq,
    T5Model,
    T5Tokenizer,
    Trainer,
)

from ptperf.modeling import _get_training_args, _load_collator


def test_t5_samsum(t5_samsum: DatasetDict) -> None:
    for split in t5_samsum.values():
        assert set(split.column_names) == {"input_ids", "attention_mask", "labels"}
        assert split.num_rows > 0


def test_t5_samsum_forward_pass(
    t5: T5Model,
    t5_tokenizer: T5Tokenizer,
    t5_samsum: DatasetDict,
) -> None:
    examples = [t5_samsum["train"][i] for i in range(4)]
    collator = DataCollatorForSeq2Seq(t5_tokenizer)

    batch = collator(examples)
    out = t5(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_t5_lora_samsum_forward_pass(
    t5_lora: LoraModel,
    t5_tokenizer: T5Tokenizer,
    t5_samsum: DatasetDict,
) -> None:
    examples = [t5_samsum["train"][i] for i in range(4)]
    collator = DataCollatorForSeq2Seq(t5_tokenizer)

    batch = collator(examples)
    out = t5_lora(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_t5_prefix_samsum_forward_pass(
    t5_prefix: PeftModelForSeq2SeqLM,
    t5_tokenizer: T5Tokenizer,
    t5_samsum: DatasetDict,
) -> None:
    examples = [t5_samsum["train"][i] for i in range(4)]
    collator = DataCollatorForSeq2Seq(t5_tokenizer)

    batch = collator(examples)
    out = t5_prefix(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_t5_prompt_samsum_forward_pass(
    t5_prompt: PeftModelForSeq2SeqLM,
    t5_tokenizer: T5Tokenizer,
    t5_samsum: DatasetDict,
) -> None:
    examples = [t5_samsum["train"][i] for i in range(4)]
    collator = DataCollatorForSeq2Seq(t5_tokenizer)

    batch = collator(examples)
    out = t5_prompt(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_t5_p_tune_samsum_forward_pass(
    t5_p_tune: PeftModelForSeq2SeqLM,
    t5_tokenizer: T5Tokenizer,
    t5_samsum: DatasetDict,
) -> None:
    examples = [t5_samsum["train"][i] for i in range(4)]
    collator = DataCollatorForSeq2Seq(t5_tokenizer)

    batch = collator(examples)
    out = t5_p_tune(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_t5_trainer_training_loop(
    t5_lora: LoraModel,
    t5_samsum: DatasetDict,
    t5_tokenizer: T5Tokenizer,
) -> None:
    collator = _load_collator(t5_tokenizer, "seq2seq")
    args = _get_training_args("test", 4, 2, False)
    trainer = Trainer(
        args=args,
        model=t5_lora,
        data_collator=collator,
        train_dataset=t5_samsum["train"],
    )

    trainer.train()
