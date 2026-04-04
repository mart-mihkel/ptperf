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

## Tracking

Experiment metrics are reported to `mlflow` and can be see by serving the ui

```bash
uv run mlflow ui
```

For tracking on a remote server set the `MLFLOW_TRACKING_URI` environment
variable.
