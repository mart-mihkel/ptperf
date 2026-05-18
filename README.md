# Prompt Based Fine-Tuning Hardware Performance

## Development

Use [uv](https://docs.astral.sh/uv/) for package management.

Setup a virtualenv for cpu or your corresponding accelerator. When using an
accelerator you should also have cuda-toolkit or rocm-toolkit available on the
system to compile `flash-attn`.

```bash
MAX_JOBS=[n-jobs] uv sync --compile-bytecode --extra [cpu|cu128|rocm72]
```

Use the `notebooks` extra for `jupyter` and plotting dependencies.

## Usage

All experiments are runnable trough the `cli` installed in the virtualenv.

```bash
cli --help
```

The setup for our runs is in [run.sh](./run/run.sh).

## Tracking

Experiment metrics are reported to `mlflow` and can be seen by serving the ui.

```bash
mlflow ui
```

Metrics can be exported to a csv with the `collect` script in the virtualenv.

```bash
collect --help
```
