from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import PRIMARY_COVERAGE_THRESHOLD, DataBundle, load_data_bundle
from .models import build_model, count_parameters


@dataclass
class ExperimentConfig:
    input_days: int = 90
    stride: int = 1
    batch_size: int = 32
    epochs: int = 60
    patience: int = 10
    validation_origins: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    d_model: int = 96
    hidden_dim: int = 96
    dropout: float = 0.15
    num_workers: int = 0
    use_amp: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    weighted = (prediction - target).square() * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def chronological_validation_indices(
    origins: list[int],
    n_days: int,
    horizon: int,
    validation_origins: int,
    validation_end_origin: int | None = None,
) -> tuple[list[int], list[int], int]:
    """Create label-disjoint train/validation blocks.

    Training outputs end before ``validation_start``. Validation outputs begin
    at or after that boundary and finish within the training CSV. Inputs may use
    the immediately preceding history, which is available at forecast time.
    """
    latest_origin = n_days - horizon
    validation_end = (
        latest_origin
        if validation_end_origin is None
        else min(latest_origin, validation_end_origin)
    )
    validation_start = validation_end - validation_origins + 1
    if validation_start <= 0:
        raise ValueError("Not enough daily rows for the requested validation block.")
    train_indices = [
        index
        for index, origin in enumerate(origins)
        if origin + horizon <= validation_start
    ]
    validation_indices = [
        index
        for index, origin in enumerate(origins)
        if validation_start <= origin <= validation_end
    ]
    if not train_indices or not validation_indices:
        raise ValueError(
            f"Invalid time split: train={len(train_indices)}, val={len(validation_indices)}"
        )
    return train_indices, validation_indices, validation_start


def _loader(
    dataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_weight = 0.0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for x, calendar, target, mask in loader:
            x = x.to(device, non_blocking=True)
            calendar = calendar.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=amp_enabled, dtype=torch.float16
            ):
                prediction = model(x, calendar)
                loss = masked_mse(prediction, target, mask)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            weight = float(mask.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * max(weight, 1.0)
            total_weight += max(weight, 1.0)
    return total_loss / max(total_weight, 1.0)


def tune_epoch_count(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[int, list[dict], dict]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            config.gradient_clip,
            amp_enabled,
            scaler,
        )
        validation_loss = _run_epoch(
            model,
            validation_loader,
            device,
            None,
            config.gradient_clip,
            amp_enabled,
            None,
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  epoch={epoch:03d} train={train_loss:.6f} "
                f"val={validation_loss:.6f} best={best_loss:.6f}"
            )
        if stale_epochs >= config.patience:
            break
    return best_epoch, history, best_state


def refit_full_training_set(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    config: ExperimentConfig,
    device: torch.device,
) -> list[float]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    losses = []
    for epoch in range(1, max(1, epochs) + 1):
        loss = _run_epoch(
            model,
            loader,
            device,
            optimizer,
            config.gradient_clip,
            amp_enabled,
            scaler,
        )
        losses.append(loss)
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(f"  refit epoch={epoch:03d}/{epochs:03d} loss={loss:.6f}")
    return losses


def _new_model(
    model_name: str,
    bundle: DataBundle,
    horizon: int,
    config: ExperimentConfig,
    use_multiscale: bool,
    use_spectral: bool,
) -> nn.Module:
    return build_model(
        model_name,
        input_dim=len(bundle.feature_names),
        input_days=config.input_days,
        horizon=horizon,
        target_index=bundle.target_index,
        d_model=config.d_model,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        use_multiscale=use_multiscale,
        use_spectral=use_spectral,
    )


def _metric_record(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict:
    primary = mask.astype(bool)
    if not primary.any():
        raise ValueError("No high-coverage test targets remain for evaluation.")
    return {
        "mse": float(mean_squared_error(truth[primary], prediction[primary])),
        "mae": float(mean_absolute_error(truth[primary], prediction[primary])),
        "mse_all_imputed": float(mean_squared_error(truth, prediction)),
        "mae_all_imputed": float(mean_absolute_error(truth, prediction)),
        "primary_days": int(primary.sum()),
        "forecast_days": int(len(truth)),
    }


def run_single_experiment(
    model_name: str,
    horizon: int,
    seed: int,
    train_csv: str | Path,
    test_csv: str | Path,
    output_dir: str | Path,
    config: ExperimentConfig,
    use_multiscale: bool = True,
    use_spectral: bool = True,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    start_time = time.perf_counter()
    output_dir = Path(output_dir)
    run_dir = output_dir / model_name / f"horizon_{horizon}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"\n[{model_name} | horizon={horizon} | seed={seed} | {device}]")

    # Find the blocked validation boundary before fitting the tuning scaler.
    preview = load_data_bundle(train_csv, test_csv)
    preview_dataset = preview.training_dataset(config.input_days, horizon, config.stride)
    train_dates = pd.to_datetime(preview.train_df["date"])
    test_start = pd.Timestamp(preview.test_df.iloc[0]["date"])
    season_matched_date = test_start - pd.DateOffset(years=1)
    season_distance = (train_dates - season_matched_date).abs()
    season_matched_origin = int(season_distance.to_numpy().argmin())
    train_indices, validation_indices, validation_start = chronological_validation_indices(
        preview_dataset.origins,
        len(preview.train_df),
        horizon,
        min(config.validation_origins, max(1, len(preview_dataset) // 4)),
        validation_end_origin=season_matched_origin,
    )
    tune_bundle = load_data_bundle(
        train_csv, test_csv, scaler_fit_end=validation_start
    )
    tune_dataset = tune_bundle.training_dataset(
        config.input_days, horizon, config.stride
    )
    train_loader = _loader(
        tune_dataset,
        train_indices,
        config.batch_size,
        True,
        config.num_workers,
    )
    validation_loader = _loader(
        tune_dataset,
        validation_indices,
        config.batch_size,
        False,
        config.num_workers,
    )
    set_seed(seed)
    tune_model = _new_model(
        model_name,
        tune_bundle,
        horizon,
        config,
        use_multiscale,
        use_spectral,
    )
    parameter_count = count_parameters(tune_model)
    print(
        f"  windows: train={len(train_indices)} val={len(validation_indices)} "
        f"parameters={parameter_count:,}"
    )
    best_epoch, tune_history, _ = tune_epoch_count(
        tune_model, train_loader, validation_loader, config, device
    )
    del tune_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Refit a fresh model on every training window using a train-only scaler.
    full_bundle = load_data_bundle(train_csv, test_csv)
    full_dataset = full_bundle.training_dataset(
        config.input_days, horizon, config.stride
    )
    full_loader = _loader(
        full_dataset,
        list(range(len(full_dataset))),
        config.batch_size,
        True,
        config.num_workers,
    )
    set_seed(seed)
    final_model = _new_model(
        model_name,
        full_bundle,
        horizon,
        config,
        use_multiscale,
        use_spectral,
    )
    refit_losses = refit_full_training_set(
        final_model, full_loader, best_epoch, config, device
    )

    x, calendar, truth, dates, observed_ratio = full_bundle.test_case(
        config.input_days, horizon
    )
    final_model.eval()
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type,
        enabled=config.use_amp and device.type == "cuda",
        dtype=torch.float16,
    ):
        normalized_prediction = final_model(
            x.to(device), calendar.to(device)
        ).squeeze(0)
    prediction = full_bundle.scaler.inverse_target(
        normalized_prediction.float().cpu().numpy(), full_bundle.target_index
    )
    quality_mask = observed_ratio >= PRIMARY_COVERAGE_THRESHOLD
    metrics = _metric_record(truth, prediction, quality_mask)
    persistence = np.full_like(truth, full_bundle.train_df.iloc[-1]["global_active_power"])
    persistence_metrics = _metric_record(truth, persistence, quality_mask)
    runtime = time.perf_counter() - start_time
    peak_cuda_memory_mb = (
        float(torch.cuda.max_memory_allocated() / 1024**2)
        if device.type == "cuda"
        else 0.0
    )
    metrics.update(
        {
            "model": model_name,
            "horizon": horizon,
            "seed": seed,
            "best_epoch": best_epoch,
            "learning_rate": config.learning_rate,
            "parameters": parameter_count,
            "train_windows": len(full_dataset),
            "validation_start_date": str(
                pd.Timestamp(full_bundle.train_df.iloc[validation_start]["date"]).date()
            ),
            "validation_policy": "season_matched_block_one_year_before_test_origin",
            "runtime_seconds": runtime,
            "peak_cuda_memory_mb": peak_cuda_memory_mb,
            "device": str(device),
            "persistence_mse": persistence_metrics["mse"],
            "persistence_mae": persistence_metrics["mae"],
        }
    )
    if hasattr(final_model, "local_strength_logit"):
        metrics["learned_local_strength"] = float(
            torch.sigmoid(final_model.local_strength_logit).detach().cpu()
        )
    if hasattr(final_model, "spectral_strength_logit"):
        metrics["learned_spectral_strength"] = float(
            torch.sigmoid(final_model.spectral_strength_logit).detach().cpu()
        )
    prediction_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).dt.strftime("%Y-%m-%d"),
            "ground_truth": truth,
            "prediction": prediction,
            "observed_ratio": observed_ratio,
            "included_in_primary_metrics": quality_mask,
        }
    )
    prediction_frame.to_csv(run_dir / "predictions.csv", index=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (run_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"tuning": tune_history, "refit_losses": refit_losses},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "model": model_name,
            "horizon": horizon,
            "seed": seed,
            "feature_names": full_bundle.feature_names,
            "scaler": full_bundle.scaler.to_dict(),
            "experiment_config": asdict(config),
            "best_epoch": best_epoch,
            "use_multiscale": use_multiscale,
            "use_spectral": use_spectral,
        },
        run_dir / "model.pt",
    )
    print(
        f"  test MSE={metrics['mse']:.3f} MAE={metrics['mae']:.3f} "
        f"days={metrics['primary_days']}/{horizon} runtime={runtime:.1f}s"
    )
    return metrics, prediction_frame, tune_history
