"""Run inference with a trained multimodal forecaster."""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import torch

from ..models.model import MultimodalPriceForecaster
from .config import TrainingConfig
from .data import Batch, create_dataloader
from .diffusion import DiffusionSchedule


def _move_batch(batch: Batch, device: torch.device) -> Batch:
    return {
        "timeseries_history": batch["timeseries_history"].to(device),
        "timeseries_future": batch["timeseries_future"].to(device),
        "text": batch["text"],
    }


def _load_model_from_checkpoint(path: str, device: torch.device) -> tuple[MultimodalPriceForecaster, TrainingConfig]:
    checkpoint = torch.load(path, map_location=device)
    config_data = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(config_data, dict):
        config = TrainingConfig.from_dict(config_data)
    else:
        config = TrainingConfig()
    model = MultimodalPriceForecaster(config.model)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config


def _sample_batch(
    model: MultimodalPriceForecaster,
    histories: torch.Tensor,
    texts: List[str],
    schedule: DiffusionSchedule,
    num_samples: int,
    generator_seed: Optional[int],
) -> torch.Tensor:
    device = histories.device
    batch_size = histories.size(0)
    forecast_steps = model.diffusion.config.forecast_steps
    price_channels = model.diffusion.config.price_channels
    samples = []

    for sample_idx in range(num_samples):
        generator = None
        if generator_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(generator_seed + sample_idx)
        current = torch.randn(
            (batch_size, forecast_steps, price_channels),
            device=device,
            generator=generator,
        )
        for step in reversed(range(schedule.num_steps)):
            timestep = torch.full((batch_size,), step, device=device, dtype=torch.long)
            timestep_cond = schedule.normalise_timesteps(timestep)
            noise_pred = model(histories, texts, current, timestep_cond)
            current = schedule.step(noise_pred, timestep, current, generator)
        samples.append(current)

    stacked = torch.stack(samples, dim=0)  # (num_samples, batch, steps, channels)
    return stacked.permute(1, 0, 2, 3)  # (batch, num_samples, steps, channels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint (.pt file).")
    parser.add_argument("--output", required=True, help="Destination .npz file for predictions.")
    parser.add_argument("--split", default="test", help="Dataset split to run inference on.")
    parser.add_argument("--batch-size", type=int, help="Override batch size from the checkpoint config.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers for inference.")
    parser.add_argument("--limit", type=int, help="Optional cap on number of examples processed.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of diffusion samples per example.")
    parser.add_argument("--seed", type=int, help="Base random seed for sampling.")
    parser.add_argument("--diffusion-steps", type=int, default=1000, help="Number of reverse diffusion steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.num_samples < 1:
        raise ValueError("--num-samples must be a positive integer")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    model, config = _load_model_from_checkpoint(args.checkpoint, device)
    batch_size = args.batch_size or config.batch_size

    dataloader = create_dataloader(
        split=args.split,
        window_config=config.window,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    schedule = DiffusionSchedule(num_steps=args.diffusion_steps).to(device)

    all_predictions: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_texts: List[str] = []
    seen = 0
    limit = args.limit

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch(batch, device)
            histories = batch["timeseries_history"]
            targets = batch["timeseries_future"]
            texts = batch["text"]

            if limit is not None and seen >= limit:
                break
            if limit is not None and seen + histories.size(0) > limit:
                slice_count = limit - seen
                histories = histories[:slice_count]
                targets = targets[:slice_count]
                texts = texts[:slice_count]

            predictions = _sample_batch(
                model,
                histories,
                texts,
                schedule,
                num_samples=args.num_samples,
                generator_seed=args.seed,
            )

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_texts.extend(texts)
            seen += histories.size(0)

            if limit is not None and seen >= limit:
                break

    if all_predictions:
        predictions_arr = torch.cat(all_predictions, dim=0).numpy()
        targets_arr = torch.cat(all_targets, dim=0).numpy()
    else:
        predictions_arr = np.empty((0, args.num_samples, model.diffusion.config.forecast_steps, model.diffusion.config.price_channels), dtype=np.float32)
        targets_arr = np.empty((0, model.diffusion.config.forecast_steps, model.diffusion.config.price_channels), dtype=np.float32)

    texts_arr = np.array(all_texts, dtype=object)
    np.savez_compressed(
        args.output,
        predictions=predictions_arr,
        targets=targets_arr,
        texts=texts_arr,
    )
    print(f"Saved predictions for {predictions_arr.shape[0]} examples to {args.output}")


if __name__ == "__main__":
    main()

