from typer.testing import CliRunner

from ptperf.cli import app

runner = CliRunner()


def test_lora_missing_rank() -> None:
    result = runner.invoke(
        app,
        [
            "--model",
            "gpt2",
            "--task",
            "causal-lm",
            "--method",
            "lora",
            "--lora-alpha",
            "16",
            "--max-steps",
            "1",
        ],
    )

    assert result.exception is not None
    assert isinstance(result.exception, ValueError)
    assert "`lora_rank` nad `lora_alpha` must be set" in str(result.exception)


def test_lora_missing_alpha() -> None:
    result = runner.invoke(
        app,
        [
            "--model",
            "gpt2",
            "--task",
            "causal-lm",
            "--method",
            "lora",
            "--lora-rank",
            "16",
            "--max-steps",
            "1",
        ],
    )

    assert result.exception is not None
    assert isinstance(result.exception, ValueError)
    assert "`lora_rank` nad `lora_alpha` must be set" in str(result.exception)


def test_lora_neither() -> None:
    result = runner.invoke(
        app,
        [
            "--model",
            "gpt2",
            "--task",
            "causal-lm",
            "--method",
            "lora",
            "--max-steps",
            "1",
        ],
    )

    assert result.exception is not None
    assert isinstance(result.exception, ValueError)
    assert "`lora_rank` nad `lora_alpha` must be set" in str(result.exception)


def test_prefix_tune_missing_virtual_tokens() -> None:
    result = runner.invoke(
        app,
        [
            "--model",
            "gpt2",
            "--task",
            "causal-lm",
            "--method",
            "prefix-tune",
            "--max-steps",
            "1",
        ],
    )

    assert result.exception is not None
    assert isinstance(result.exception, ValueError)
    assert "`virtual_tokens` must be set" in str(result.exception)


def test_prompt_tune_missing_virtual_tokens() -> None:
    result = runner.invoke(
        app,
        [
            "--model",
            "gpt2",
            "--task",
            "causal-lm",
            "--method",
            "prompt-tune",
            "--max-steps",
            "1",
        ],
    )

    assert result.exception is not None
    assert isinstance(result.exception, ValueError)
    assert "`virtual_tokens` must be set" in str(result.exception)


def test_p_tune_missing_virtual_tokens() -> None:
    result = runner.invoke(
        app,
        [
            "--model",
            "gpt2",
            "--task",
            "causal-lm",
            "--method",
            "p-tune",
            "--max-steps",
            "1",
        ],
    )

    assert result.exception is not None
    assert isinstance(result.exception, ValueError)
    assert "`virtual_tokens` must be set" in str(result.exception)


def test_p_tune_missing_encoder_dim() -> None:
    result = runner.invoke(
        app,
        [
            "--model",
            "gpt2",
            "--task",
            "causal-lm",
            "--method",
            "p-tune",
            "--virtual-tokens",
            "8",
            "--max-steps",
            "1",
        ],
    )

    assert result.exception is not None
    assert isinstance(result.exception, ValueError)
    assert "`encoder_dim` must be set" in str(result.exception)
