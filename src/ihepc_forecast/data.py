from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ELECTRICAL_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]
SUM_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
]
MEAN_COLUMNS = ["voltage", "global_intensity"]
CALENDAR_COLUMNS = [
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]
DEFAULT_FEATURES = SUM_COLUMNS + MEAN_COLUMNS + CALENDAR_COLUMNS + ["observed_ratio"]
TARGET = "global_active_power"
PRIMARY_COVERAGE_THRESHOLD = 0.95


def _snake_case_columns(columns: Iterable[str]) -> list[str]:
    return [str(c).strip().lower().replace(" ", "_") for c in columns]


def _calendar_frame(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "doy_sin": np.sin(2 * np.pi * (doy - 1) / 365.25),
            "doy_cos": np.cos(2 * np.pi * (doy - 1) / 365.25),
        }
    )


def read_uci_minute_data(raw_path: str | Path) -> pd.DataFrame:
    """Read the original semicolon-separated UCI text file.

    Missing values are parsed but deliberately not filled here.  Keeping this
    step separate makes the missing-data policy explicit and testable.
    """
    raw_path = Path(raw_path)
    df = pd.read_csv(
        raw_path,
        sep=";",
        na_values=["?", "", "NA"],
        low_memory=False,
    )
    df.columns = _snake_case_columns(df.columns)
    required = {"date", "time", *ELECTRICAL_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )
    for column in ELECTRICAL_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[["datetime", *ELECTRICAL_COLUMNS]].sort_values("datetime")


def _missing_run_lengths(mask: np.ndarray) -> np.ndarray:
    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    lengths = np.zeros(len(mask), dtype=np.int32)
    for start, end in zip(starts, ends):
        lengths[start:end] = end - start
    return lengths


def impute_electrical_series(
    series: pd.Series,
    max_interpolation_minutes: int = 180,
    seasonal_days: int = 7,
) -> tuple[pd.Series, dict]:
    """Paper-guided hybrid imputation with no cross-day interpolation.

    Short gaps are linearly interpolated only when both endpoints are in the
    same calendar day. Longer gaps use the median of the same minute from the
    previous seven days. This combines literature guidance with the repository's
    pseudo-missing benchmark; all original missing positions remain traceable
    through ``observed_ratio``.
    """
    original = series.isna().to_numpy()
    run_lengths = _missing_run_lengths(original)
    short_mask = original & (run_lengths <= max_interpolation_minutes)

    # Interpolation is restricted within each calendar date so an input day can
    # never borrow a value from a future output day.
    interpolated = series.groupby(series.index.normalize()).transform(
        lambda day: day.interpolate(method="linear", limit_area="inside")
    )
    values = series.to_numpy(dtype=np.float64, copy=True)
    interpolation_values = interpolated.to_numpy(dtype=np.float64)
    usable_short = short_mask & np.isfinite(interpolation_values)
    values[usable_short] = interpolation_values[usable_short]

    seasonal_filled = 0
    for index in np.flatnonzero(~np.isfinite(values)):
        candidates = []
        for day in range(1, seasonal_days + 1):
            prior = index - day * 1440
            if prior >= 0 and np.isfinite(values[prior]):
                candidates.append(values[prior])
        if candidates:
            values[index] = float(np.median(candidates))
            seasonal_filled += 1

    filled = pd.Series(values, index=series.index, name=series.name).ffill().bfill()
    return filled, {
        "missing_before": int(original.sum()),
        "linear_within_day": int(usable_short.sum()),
        "seasonal_median": int(seasonal_filled),
        "fallback_fill": int(original.sum() - usable_short.sum() - seasonal_filled),
    }


def aggregate_daily(raw_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Convert the original minute data to an uninterrupted daily series.

    Short internal gaps are interpolated within a day; long gaps use a robust
    historical seasonal statistic. Boundary days are removed because the UCI
    recording starts and ends part-way through a day. ``observed_ratio`` is
    retained so training/evaluation can mask unreliable labels without breaking
    the 90/365-day output sequence.
    """
    minute = read_uci_minute_data(raw_path)
    if minute["datetime"].duplicated().any():
        minute = minute.drop_duplicates("datetime", keep="last")

    minute = minute.set_index("datetime")
    full_index = pd.date_range(minute.index.min(), minute.index.max(), freq="min")
    minute = minute.reindex(full_index)

    observed = minute[TARGET].notna().astype(np.float32)
    original_missing = int((1 - observed).sum())
    imputation_counts: dict[str, dict] = {}
    for column in ELECTRICAL_COLUMNS:
        minute[column], imputation_counts[column] = impute_electrical_series(
            minute[column]
        )

    if minute[ELECTRICAL_COLUMNS].isna().any().any():
        missing = minute[ELECTRICAL_COLUMNS].isna().sum()
        raise ValueError(f"Unfilled electrical values remain:\n{missing}")

    minute["sub_metering_remainder"] = (
        minute[TARGET] * 1000.0 / 60.0
        - minute["sub_metering_1"]
        - minute["sub_metering_2"]
        - minute["sub_metering_3"]
    )
    minute["observed"] = observed

    aggregation = {column: "sum" for column in SUM_COLUMNS}
    aggregation.update({column: "mean" for column in MEAN_COLUMNS})
    aggregation["observed"] = "sum"
    daily = minute.resample("D").agg(aggregation)
    daily["observed_ratio"] = daily.pop("observed") / 1440.0

    # Only the first and last dates are partial because recording started/ended
    # intraday.  Internal low-coverage dates remain, with a quality indicator.
    first_date = minute.index.min().normalize()
    last_date = minute.index.max().normalize()
    boundary_removed = []
    for date in (first_date, last_date):
        if date in daily.index and daily.at[date, "observed_ratio"] < 0.999:
            boundary_removed.append(str(date.date()))
            daily = daily.drop(index=date)

    daily = daily.reset_index(names="date")
    calendar = _calendar_frame(daily["date"])
    daily = pd.concat([daily.reset_index(drop=True), calendar], axis=1)

    metadata = {
        "raw_path": str(Path(raw_path).resolve()),
        "raw_start": str(minute.index.min()),
        "raw_end": str(minute.index.max()),
        "raw_rows_after_reindex": int(len(minute)),
        "missing_target_minutes_before_imputation": original_missing,
        "imputation_policy": (
            "within_day_linear_for_gaps_le_180min_then_previous_7d_same_minute_median"
        ),
        "imputation_counts": imputation_counts,
        "boundary_days_removed": boundary_removed,
        "daily_rows": int(len(daily)),
        "daily_start": str(daily["date"].min().date()),
        "daily_end": str(daily["date"].max().date()),
        "daily_target_definition": "sum of 1440 minute-level kW readings",
    }
    return daily, metadata


def prepare_train_test(
    raw_path: str | Path,
    output_dir: str | Path,
    split_date: str | None = None,
    test_days: int = 365,
    save_raw_minute_splits: bool = True,
) -> dict:
    """Aggregate once, then make a chronological train/test split.

    By default the last 365 *complete* days form the untouched test block.  A
    date can still be supplied for sensitivity analyses, but it is never
    inferred from or copied from another project.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = output_dir / "daily"
    minute_dir = output_dir / "minute"
    daily_dir.mkdir(parents=True, exist_ok=True)
    if save_raw_minute_splits:
        minute_dir.mkdir(parents=True, exist_ok=True)
    daily, metadata = aggregate_daily(raw_path)
    if split_date is None:
        if len(daily) <= test_days:
            raise ValueError("Daily series is shorter than the requested test block.")
        split = pd.Timestamp(daily.iloc[-test_days]["date"])
    else:
        split = pd.Timestamp(split_date)
    train = daily[daily["date"] < split].copy()
    test = daily[daily["date"] >= split].copy()
    if len(train) < 90 + 365:
        raise ValueError("Training range is too short for a 90 -> 365 sample.")
    if len(test) < 365:
        raise ValueError("Test range must contain at least 365 days.")

    daily.to_csv(daily_dir / "daily_all.csv", index=False)
    train.to_csv(daily_dir / "train_daily.csv", index=False)
    test.to_csv(daily_dir / "test_daily.csv", index=False)
    if save_raw_minute_splits:
        # Re-read the untouched measurements so these files remain auditable:
        # missing values stay missing and no imputed value is presented as raw.
        raw_minute = read_uci_minute_data(raw_path)
        raw_dates = raw_minute["datetime"].dt.normalize()
        first_complete_date = pd.Timestamp(daily["date"].min())
        last_complete_date = pd.Timestamp(daily["date"].max())
        train_minute = raw_minute[
            (raw_dates >= first_complete_date) & (raw_dates < split)
        ].copy()
        test_minute = raw_minute[
            (raw_dates >= split) & (raw_dates <= last_complete_date)
        ].copy()
        train_minute.to_csv(minute_dir / "train_minute_raw.csv", index=False)
        test_minute.to_csv(minute_dir / "test_minute_raw.csv", index=False)
        metadata.update(
            {
                "train_minute_raw_rows": int(len(train_minute)),
                "test_minute_raw_rows": int(len(test_minute)),
                "minute_split_policy": "original_values_with_missing_preserved",
                "minute_output_dir": str(minute_dir.resolve()),
            }
        )
    metadata["daily_output_dir"] = str(daily_dir.resolve())
    metadata.update(
        {
            "split_date": str(split.date()),
            "split_policy": (
                f"last_{test_days}_complete_days" if split_date is None else "explicit_date"
            ),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_date_range": [
                str(train["date"].min().date()),
                str(train["date"].max().date()),
            ],
            "test_date_range": [
                str(test["date"].min().date()),
                str(test["date"].max().date()),
            ],
        }
    )
    with (output_dir / "preprocess_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0).astype(np.float32)
        scale = values.std(axis=0).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_target(self, values: np.ndarray, target_index: int = 0) -> np.ndarray:
        return values * self.scale[target_index] + self.mean[target_index]

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


class WindowDataset(Dataset):
    """Samples with shape input_days x features -> horizon target days."""

    def __init__(
        self,
        features: np.ndarray,
        future_calendar: np.ndarray,
        target: np.ndarray,
        target_mask: np.ndarray,
        input_days: int,
        horizon: int,
        stride: int = 1,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.calendar = torch.as_tensor(future_calendar, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.target_mask = torch.as_tensor(target_mask, dtype=torch.float32)
        self.input_days = input_days
        self.horizon = horizon
        self.origins = list(
            range(input_days, len(features) - horizon + 1, max(1, stride))
        )
        if not self.origins:
            raise ValueError(
                f"No windows: rows={len(features)}, input={input_days}, horizon={horizon}"
            )

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int):
        origin = self.origins[index]
        return (
            self.features[origin - self.input_days : origin],
            self.calendar[origin : origin + self.horizon],
            self.target[origin : origin + self.horizon],
            self.target_mask[origin : origin + self.horizon],
        )


@dataclass
class DataBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_names: list[str]
    scaler: Standardizer
    train_features: np.ndarray
    train_calendar: np.ndarray
    train_target: np.ndarray
    train_target_mask: np.ndarray
    test_features: np.ndarray
    test_calendar: np.ndarray
    test_target: np.ndarray
    test_target_mask: np.ndarray

    @property
    def target_index(self) -> int:
        return self.feature_names.index(TARGET)

    def training_dataset(self, input_days: int, horizon: int, stride: int) -> WindowDataset:
        return WindowDataset(
            self.train_features,
            self.train_calendar,
            self.train_target,
            self.train_target_mask,
            input_days,
            horizon,
            stride,
        )

    def test_case(self, input_days: int, horizon: int):
        if len(self.test_df) < horizon:
            raise ValueError(f"Test set has fewer than {horizon} days")
        x = torch.as_tensor(self.train_features[-input_days:], dtype=torch.float32)
        future = torch.as_tensor(self.test_calendar[:horizon], dtype=torch.float32)
        truth = self.test_df[TARGET].to_numpy(np.float32)[:horizon]
        dates = pd.to_datetime(self.test_df["date"]).iloc[:horizon]
        quality = self.test_df["observed_ratio"].to_numpy(np.float32)[:horizon]
        return x.unsqueeze(0), future.unsqueeze(0), truth, dates, quality


def load_data_bundle(
    train_csv: str | Path,
    test_csv: str | Path,
    feature_names: list[str] | None = None,
    scaler_fit_end: int | None = None,
) -> DataBundle:
    train_df = pd.read_csv(train_csv, parse_dates=["date"])
    test_df = pd.read_csv(test_csv, parse_dates=["date"])
    feature_names = list(feature_names or DEFAULT_FEATURES)
    missing = set(feature_names).difference(train_df.columns)
    if missing:
        raise ValueError(f"Processed data is missing features: {sorted(missing)}")

    scaler_values = train_df[feature_names].to_numpy(np.float32)
    if scaler_fit_end is not None:
        if not 1 <= scaler_fit_end <= len(train_df):
            raise ValueError(f"Invalid scaler_fit_end={scaler_fit_end}")
        scaler_values = scaler_values[:scaler_fit_end]
    scaler = Standardizer.fit(scaler_values)
    train_features = scaler.transform(train_df[feature_names].to_numpy(np.float32))
    test_features = scaler.transform(test_df[feature_names].to_numpy(np.float32))
    target_index = feature_names.index(TARGET)
    train_target = train_features[:, target_index]
    test_target = test_features[:, target_index]
    train_target_mask = (
        train_df["observed_ratio"].to_numpy(np.float32) >= PRIMARY_COVERAGE_THRESHOLD
    ).astype(np.float32)
    test_target_mask = (
        test_df["observed_ratio"].to_numpy(np.float32) >= PRIMARY_COVERAGE_THRESHOLD
    ).astype(np.float32)
    return DataBundle(
        train_df=train_df,
        test_df=test_df,
        feature_names=feature_names,
        scaler=scaler,
        train_features=train_features,
        train_calendar=train_df[CALENDAR_COLUMNS].to_numpy(np.float32),
        train_target=train_target,
        train_target_mask=train_target_mask,
        test_features=test_features,
        test_calendar=test_df[CALENDAR_COLUMNS].to_numpy(np.float32),
        test_target=test_target,
        test_target_mask=test_target_mask,
    )
