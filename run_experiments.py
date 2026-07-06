from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ihepc_forecast.engine import ExperimentConfig, run_single_experiment
from ihepc_forecast.plots import (
    aggregate_prediction_frames,
    plot_forecast,
    plot_metric_bars,
    plot_model_comparison,
    plot_training_history,
)


DEFAULT_SEEDS = [2026, 2036, 2046, 2056, 2066]
warnings.filterwarnings("ignore", message=".*not compiled with flash attention.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible power forecasting experiments")
    parser.add_argument("--models", nargs="+", default=["lstm", "transformer", "stl_former"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[90, 365])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--train-csv", type=Path, default=ROOT / "data/processed/daily/train_daily.csv")
    parser.add_argument("--test-csv", type=Path, default=ROOT / "data/processed/daily/test_daily.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/experiments")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Optional global learning-rate override. By default the vanilla "
            "Transformer uses 1e-4 for the 365-day task, following common "
            "long-term forecasting settings; other cases use 1e-3."
        ),
    )
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help=(
            "Optional global dropout override. Defaults to 0.10 for the "
            "vanilla Transformer and 0.15 for the recurrent/improved models."
        ),
    )
    parser.add_argument("--validation-origins", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One-seed, two-epoch integration test")
    return parser.parse_args()


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, horizon), group in metrics.groupby(["model", "horizon"], sort=False):
        row = {"model": model, "horizon": horizon, "runs": len(group)}
        for metric in [
            "mse",
            "mae",
            "mse_all_imputed",
            "mae_all_imputed",
            "runtime_seconds",
            "peak_cuda_memory_mb",
        ]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        row["parameters"] = int(group["parameters"].iloc[0])
        row["best_epoch_mean"] = group["best_epoch"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.train_csv.exists() or not args.test_csv.exists():
        raise FileNotFoundError("Processed CSV files are missing. Run prepare_data.py first.")
    if args.smoke:
        args.seeds = [2026]
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.validation_origins = min(args.validation_origins, 4)
        args.d_model = min(args.d_model, 32)
        args.hidden_dim = min(args.hidden_dim, 32)
        args.batch_size = min(args.batch_size, 16)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []
    all_predictions: dict[tuple[str, int], list[pd.DataFrame]] = {}
    for horizon in args.horizons:
        if horizon not in {90, 365}:
            raise ValueError("Course tasks support horizons 90 and 365 only.")
        for model_name in args.models:
            if model_name not in {"lstm", "transformer", "stl_former"}:
                raise ValueError(f"Unsupported model: {model_name}")
            learning_rate = args.lr
            if learning_rate is None:
                learning_rate = (
                    1e-4 if model_name == "transformer" and horizon == 365 else 1e-3
                )
            dropout = args.dropout
            if dropout is None:
                dropout = 0.10 if model_name == "transformer" else 0.15
            config = ExperimentConfig(
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                validation_origins=args.validation_origins,
                learning_rate=learning_rate,
                d_model=args.d_model,
                hidden_dim=args.hidden_dim,
                dropout=dropout,
                num_workers=args.num_workers,
                use_amp=not args.no_amp,
            )
            for seed in args.seeds:
                metrics, predictions, history = run_single_experiment(
                    model_name=model_name,
                    horizon=horizon,
                    seed=seed,
                    train_csv=args.train_csv,
                    test_csv=args.test_csv,
                    output_dir=args.output_dir,
                    config=config,
                )
                all_metrics.append(metrics)
                all_predictions.setdefault((model_name, horizon), []).append(predictions)
                plot_training_history(
                    history,
                    args.output_dir / model_name / f"horizon_{horizon}" / f"seed_{seed}" / "training_curve.png",
                )
                pd.DataFrame(all_metrics).to_csv(args.output_dir / "metrics_runs.csv", index=False)

    metrics_frame = pd.DataFrame(all_metrics)
    summary = summarize(metrics_frame)
    summary.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    plot_metric_bars(summary, args.output_dir / "metrics_comparison.png")

    train_daily = pd.read_csv(args.train_csv, parse_dates=["date"])
    for horizon in args.horizons:
        comparison = {}
        for model_name in args.models:
            aggregate = aggregate_prediction_frames(all_predictions[(model_name, horizon)])
            aggregate.to_csv(
                args.output_dir / model_name / f"horizon_{horizon}" / "predictions_aggregate.csv",
                index=False,
            )
            comparison[model_name] = aggregate
            plot_forecast(
                train_daily["date"].iloc[-90:],
                train_daily["global_active_power"].to_numpy()[-90:],
                aggregate,
                model_name,
                horizon,
                args.output_dir / model_name / f"horizon_{horizon}" / "forecast_mean_std.png",
            )
        plot_model_comparison(
            comparison,
            horizon,
            args.output_dir / f"model_comparison_horizon_{horizon}.png",
        )
    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResults saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
