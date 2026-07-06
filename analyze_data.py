from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible EDA assets")
    parser.add_argument("--daily", type=Path, default=ROOT / "data/processed/daily/daily_all.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/eda")
    return parser.parse_args()


def top_spectral_periods(values: np.ndarray, count: int = 8) -> list[dict]:
    centered = values - values.mean()
    spectrum = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(len(centered), d=1.0)
    valid = frequencies > 0
    periods = 1.0 / frequencies[valid]
    amplitudes = np.abs(spectrum[valid])
    # Avoid returning several adjacent bins for effectively the same peak.
    order = np.argsort(amplitudes)[::-1]
    selected = []
    for index in order:
        period = float(periods[index])
        if all(abs(period - item["period_days"]) > max(1.0, 0.05 * period) for item in selected):
            selected.append({"period_days": period, "amplitude": float(amplitudes[index])})
        if len(selected) >= count:
            break
    return selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.daily, parse_dates=["date"])
    target = frame["global_active_power"]
    split_date = frame["date"].iloc[-365]

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(frame["date"], target, linewidth=0.8, color="#1f77b4")
    axes[0].axvline(split_date, color="#d62728", linestyle="--", label="Test start")
    axes[0].set(title="Daily household active power", ylabel="Daily sum")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(frame["date"], frame["observed_ratio"], color="#555555", linewidth=0.8)
    axes[1].axhline(0.95, color="#d62728", linestyle="--", label="95% metric threshold")
    axes[1].set(xlabel="Date", ylabel="Observed ratio", ylim=(-0.03, 1.03))
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "time_series_and_coverage.png", dpi=220)
    plt.close(figure)

    numeric = frame.select_dtypes(include=[np.number]).drop(columns=["observed_ratio"])
    correlation = numeric.corr()
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation)), correlation.columns, rotation=65, ha="right", fontsize=8)
    axis.set_yticks(range(len(correlation)), correlation.columns, fontsize=8)
    for row in range(len(correlation)):
        for col in range(len(correlation)):
            axis.text(col, row, f"{correlation.iloc[row, col]:.2f}", ha="center", va="center", fontsize=6)
    figure.colorbar(image, ax=axis, shrink=0.8, label="Pearson correlation")
    axis.set_title("Daily feature correlation")
    figure.tight_layout()
    figure.savefig(args.output_dir / "feature_correlation.png", dpi=220)
    plt.close(figure)

    seasonal = frame.assign(
        weekday=frame["date"].dt.day_name(),
        month=frame["date"].dt.month,
    )
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_mean = seasonal.groupby("weekday")["global_active_power"].mean().reindex(weekday_order)
    month_mean = seasonal.groupby("month")["global_active_power"].mean().reindex(range(1, 13))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(range(7), weekday_mean, color="#4c78a8")
    axes[0].set_xticks(range(7), [day[:3] for day in weekday_order])
    axes[0].set(title="Mean by weekday", ylabel="Daily active-power sum")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].plot(range(1, 13), month_mean, marker="o", color="#f58518")
    axes[1].set_xticks(range(1, 13))
    axes[1].set(title="Mean by month", xlabel="Month")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output_dir / "calendar_patterns.png", dpi=220)
    plt.close(figure)

    autocorrelation = {str(lag): float(target.autocorr(lag)) for lag in [1, 7, 14, 30, 90, 365]}
    summary = {
        "daily_rows": int(len(frame)),
        "date_range": [str(frame["date"].min().date()), str(frame["date"].max().date())],
        "test_start": str(split_date.date()),
        "low_coverage_days_below_95_percent": int((frame["observed_ratio"] < 0.95).sum()),
        "fully_missing_days": int((frame["observed_ratio"] == 0).sum()),
        "target_descriptive_statistics": target.describe().to_dict(),
        "target_autocorrelation": autocorrelation,
        "dominant_spectral_periods": top_spectral_periods(target.to_numpy()),
        "target_feature_correlations": correlation["global_active_power"].sort_values(ascending=False).to_dict(),
    }
    with (args.output_dir / "eda_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"EDA saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
