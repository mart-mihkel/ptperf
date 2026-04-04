from datasets.dataset_dict import DatasetDict
from transformers import DataCollatorForSeq2Seq, T5Model, T5Tokenizer


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
