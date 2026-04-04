from unittest.mock import patch

from datasets.dataset_dict import DatasetDict

from ptperf.scripts import fine_tune, lora, prefix_tune

_gpt2 = "hf-internal-testing/tiny-random-gpt2"
_t5 = "hf-internal-testing/tiny-random-t5"


def test_causal_fine_tune_sanity(gpt2_wikitext: DatasetDict):
    with patch("ptperf.datasets.load_data", return_value=gpt2_wikitext):
        fine_tune(_gpt2, "causal-lm", "test", 0, 1)


def test_causal_lora_sanity(gpt2_wikitext: DatasetDict):
    with patch("ptperf.datasets.load_data", return_value=gpt2_wikitext):
        lora(_gpt2, "causal-lm", "test", 0, 1)


def test_causal_prefix_tune_sanity(gpt2_wikitext: DatasetDict):
    with patch("ptperf.datasets.load_data", return_value=gpt2_wikitext):
        prefix_tune(_gpt2, "causal-lm", "test", 0, 1)


def test_seq_fine_tune_sanity(t5_samsum: DatasetDict):
    with patch("ptperf.datasets.load_data", return_value=t5_samsum):
        fine_tune(_t5, "seq2seq", "test", 0, 1)


def test_seq_lora_sanity(t5_samsum: DatasetDict):
    with patch("ptperf.datasets.load_data", return_value=t5_samsum):
        lora(_t5, "seq2seq", "test", 0, 1)


def test_seq_prefix_tune_sanity(t5_samsum: DatasetDict):
    with patch("ptperf.datasets.load_data", return_value=t5_samsum):
        prefix_tune(_t5, "seq2seq", "test", 0, 1)
