"""Small launcher for running the project training pipeline.

This main script provides a lightweight entrypoint for quick experiments.
It delegates to `src.train.run_training()` and defaults to a disposable
`scratch_models` output folder when run without arguments.
"""

from src import train


def main():
    # Quick synthetic run for interactive or CI checks. Use the CLI in
    # `src/train.py` for full control when needed.
    train.run_training(use_synthetic=True, model_dir="scratch_models", save_models=False)


if __name__ == "__main__":
    main()
