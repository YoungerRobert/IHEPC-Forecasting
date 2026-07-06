from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ihepc_forecast.engine import ExperimentConfig, run_single_experiment


DEFAULT_SEEDS = [2026, 2036, 2046, 2056, 2066]
VARIANTS = {
    "full": (True, True),
    "no_spectral": (True, False),
    "no_multiscale": (False, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STL-Former ablation study")
    parser.add_argument("--horizons", nargs="+", type=int, default=[90, 365])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--train-csv", type=Path, default=ROOT / "data/processed/daily/train_daily.csv")
    parser.add_argument("--test-csv", type=Path, default=ROOT / "data/processed/daily/test_daily.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/ablation_stl_former")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--validation-origins", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, horizon), group in metrics.groupby(["variant", "horizon"], sort=False):
        row = {"variant": variant, "model": variant, "horizon": horizon, "runs": len(group)}
        for metric in ["mse", "mae", "mse_all_imputed", "mae_all_imputed", "runtime_seconds", "peak_cuda_memory_mb"]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        row["parameters"] = int(group["parameters"].iloc[0])
        row["best_epoch_mean"] = group["best_epoch"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        validation_origins=args.validation_origins,
        learning_rate=1e-3,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )
    rows = []
    for horizon in args.horizons:
        for variant, (use_multiscale, use_spectral) in VARIANTS.items():
            for seed in args.seeds:
                metrics, _, _ = run_single_experiment(
                    model_name="stl_former",
                    horizon=horizon,
                    seed=seed,
                    train_csv=args.train_csv,
                    test_csv=args.test_csv,
                    output_dir=args.output_dir / variant,
                    config=config,
                    use_multiscale=use_multiscale,
                    use_spectral=use_spectral,
                )
                metrics["variant"] = variant
                rows.append(metrics)
                pd.DataFrame(rows).to_csv(args.output_dir / "ablation_runs.csv", index=False)
    summary = summarize(pd.DataFrame(rows))
    summary.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResults saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
