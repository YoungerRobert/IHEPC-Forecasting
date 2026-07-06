from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "stl_former": "STL-Former",
}
MODEL_COLORS = {
    "lstm": "#377eb8",
    "transformer": "#ff7f00",
    "stl_former": "#984ea3",
}


def plot_training_history(history: list[dict], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(frame["epoch"], frame["train_loss"], label="Train", linewidth=1.8)
    axis.plot(frame["epoch"], frame["validation_loss"], label="Validation", linewidth=1.8)
    best = frame.loc[frame["validation_loss"].idxmin()]
    axis.scatter([best["epoch"]], [best["validation_loss"]], color="#d62728", zorder=3)
    axis.set(xlabel="Epoch", ylabel="Masked MSE (standardized)", title="Training history")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def aggregate_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No prediction frames supplied")
    stacked = np.stack([frame["prediction"].to_numpy() for frame in frames])
    base = frames[0].copy()
    base["prediction_mean"] = stacked.mean(axis=0)
    base["prediction_std"] = stacked.std(axis=0, ddof=1) if len(frames) > 1 else 0.0
    return base


def plot_forecast(
    history_dates: pd.Series,
    history_values: np.ndarray,
    aggregate: pd.DataFrame,
    model_name: str,
    horizon: int,
    output: str | Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    forecast_dates = pd.to_datetime(aggregate["date"])
    mean = aggregate["prediction_mean"].to_numpy()
    std = aggregate["prediction_std"].to_numpy()
    truth = aggregate["ground_truth"].to_numpy()
    quality = aggregate["included_in_primary_metrics"].astype(bool).to_numpy()

    figure, axis = plt.subplots(figsize=(14, 5.5))
    axis.plot(pd.to_datetime(history_dates), history_values, color="#777777", label="History (90 days)")
    axis.plot(forecast_dates[quality], truth[quality], color="#2ca02c", linewidth=1.5, label="Ground Truth")
    if (~quality).any():
        axis.scatter(
            forecast_dates[~quality],
            truth[~quality],
            color="#aaaaaa",
            s=12,
            label="Imputed/low coverage target",
            zorder=2,
        )
    color = MODEL_COLORS.get(model_name, "#d62728")
    axis.plot(forecast_dates, mean, color=color, linewidth=1.6, label=MODEL_LABELS.get(model_name, model_name))
    axis.fill_between(forecast_dates, mean - std, mean + std, color=color, alpha=0.18, label="±1 std (seeds)")
    axis.axvline(forecast_dates.iloc[0], color="#333333", linestyle="--", linewidth=1)
    axis.set(
        title=f"{MODEL_LABELS.get(model_name, model_name)}: 90 → {horizon} days",
        xlabel="Date",
        ylabel="Daily total active power (sum of minute-level kW)",
    )
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, fontsize=9)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_model_comparison(
    aggregates: dict[str, pd.DataFrame], horizon: int, output: str | Path
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    first = next(iter(aggregates.values()))
    dates = pd.to_datetime(first["date"])
    quality = first["included_in_primary_metrics"].astype(bool).to_numpy()
    figure, axis = plt.subplots(figsize=(14, 5.5))
    axis.plot(dates[quality], first.loc[quality, "ground_truth"], color="#222222", linewidth=1.5, label="Ground Truth")
    for model_name, frame in aggregates.items():
        axis.plot(
            dates,
            frame["prediction_mean"],
            color=MODEL_COLORS.get(model_name),
            linewidth=1.25,
            label=MODEL_LABELS.get(model_name, model_name),
        )
    axis.set(
        title=f"Model comparison: 90 → {horizon} days",
        xlabel="Date",
        ylabel="Daily total active power",
    )
    axis.grid(alpha=0.22)
    axis.legend(ncol=4, fontsize=9)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_metric_bars(summary: pd.DataFrame, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    horizons = sorted(summary["horizon"].unique())
    figure, axes = plt.subplots(len(horizons), 2, figsize=(11, 4.2 * len(horizons)), squeeze=False)
    for row, horizon in enumerate(horizons):
        subset = summary[summary["horizon"] == horizon].copy()
        labels = [MODEL_LABELS.get(name, name) for name in subset["model"]]
        x = np.arange(len(subset))
        for col, metric in enumerate(["mse", "mae"]):
            axis = axes[row, col]
            axis.bar(
                x,
                subset[f"{metric}_mean"],
                yerr=subset[f"{metric}_std"].fillna(0),
                color=[MODEL_COLORS.get(name, "#777777") for name in subset["model"]],
                alpha=0.85,
                capsize=4,
            )
            axis.set_xticks(x, labels, rotation=12)
            axis.set_title(f"90 → {horizon}: {metric.upper()} (mean ± std)")
            axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
