# Prompt Based Fine-Tuning Hardware Performance

## Development

Use [uv](https://docs.astral.sh/uv/) for package management.

Setup a virtualenv with your corresponding accelerator:

```bash
uv sync --extra [cpu|cu130|rocm72]
```

## Usage

```bash
uv run cli --help
```
