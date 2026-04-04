from typing import Literal

type Task = Literal["causal-lm", "seq2seq"]
type Method = Literal["fine-tune", "lora", "prefix-tune"]
