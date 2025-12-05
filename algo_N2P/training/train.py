"""Command-line training entrypoint for the multimodal forecaster."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from ..models.model import MultimodalPriceForecaster
from .config import TrainingConfig, load_training_config
from .data import Batch, create_dataloader
from .diffusion import DiffusionSchedule


def _move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return {
        "timeseries_history": batch["timeseries_history"].to(device),
        "timeseries_future": batch["timeseries_future"].to(device),
        "text": batch["text"],
    }


def _train_one_epoch(
    model: MultimodalPriceForecaster,
    dataloader,
    optimiser: AdamW,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    schedule: DiffusionSchedule,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
    log_interval: int,
    start_step: int,
) -> Dict[str, float]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    running_loss = 0.0
    running_steps = 0
    global_step = start_step

    for step, batch in enumerate(dataloader, start=1):
        batch = _move_batch_to_device(batch, device)
        optimiser.zero_grad(set_to_none=True)
        futures = batch["timeseries_future"]
        histories = batch["timeseries_history"]
        texts = batch["text"]

        timesteps = torch.randint(0, schedule.num_steps, (futures.size(0),), device=device)
        noise = torch.randn_like(futures)
        noisy_future = schedule.add_noise(futures, noise, timesteps)
        timestep_cond = schedule.normalise_timesteps(timesteps).to(device)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16 if use_amp else None):
            pred_noise = model(histories, texts, noisy_future, timestep_cond)
            loss = F.mse_loss(pred_noise, noise)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimiser)
            clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimiser)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * futures.size(0)
        running_steps += futures.size(0)
        global_step += 1
        if log_interval and step % log_interval == 0:
            current_loss = running_loss / max(running_steps, 1)
            print(f"Step {step:05d} | Global {global_step:06d} | Loss: {current_loss:.5f}")

    epoch_loss = running_loss / max(running_steps, 1)
    return {"loss": epoch_loss, "global_step": float(global_step)}


def _evaluate(
    model: MultimodalPriceForecaster,
    dataloader,
    schedule: DiffusionSchedule,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            futures = batch["timeseries_future"]
            histories = batch["timeseries_history"]
            texts = batch["text"]
            timesteps = torch.randint(0, schedule.num_steps, (futures.size(0),), device=device)
            noise = torch.randn_like(futures)
            noisy_future = schedule.add_noise(futures, noise, timesteps)
            timestep_cond = schedule.normalise_timesteps(timesteps).to(device)
            pred_noise = model(histories, texts, noisy_future, timestep_cond)
            loss = F.mse_loss(pred_noise, noise, reduction="sum")
            total_loss += loss.item()
            total_examples += futures.size(0)
    mse = total_loss / max(total_examples, 1)
    return {"noise_mse": mse, "noise_rmse": math.sqrt(mse)}


def _build_scheduler(
    optimiser: AdamW,
    warmup_ratio: float,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    if total_steps <= 0:
        return None
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


def _save_checkpoint(state: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Saved checkpoint to {path}")


def _load_checkpoint(
    path: str,
    model: MultimodalPriceForecaster,
    optimiser: Optional[AdamW] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: Optional[str] = None,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimiser is not None and "optim_state" in checkpoint:
        optimiser.load_state_dict(checkpoint["optim_state"])
    scheduler_state = checkpoint.get("scheduler_state")
    if scheduler is not None and scheduler_state:
        scheduler.load_state_dict(scheduler_state)
    print(f"Loaded checkpoint from {path}")
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, help="Path to training YAML configuration.")
    parser.add_argument("--output-dir", type=str, default="runs/multimodal", help="Directory to store checkpoints.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, help="Override batch size from config.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument("--log-interval", type=int, default=25, help="Steps between logging updates.")
    parser.add_argument("--eval-split", type=str, default="validation", help="Dataset split for evaluation.")
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation during training.")
    parser.add_argument("--checkpoint", type=str, help="Optional checkpoint for resuming training or evaluation.")
    parser.add_argument("--validate-only", action="store_true", help="Skip training and run evaluation only.")
    parser.add_argument("--diffusion-steps", type=int, default=1000, help="Number of diffusion steps for training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_state: Optional[Dict[str, object]] = None
    if args.checkpoint:
        checkpoint_state = torch.load(args.checkpoint, map_location="cpu")

    if args.config:
        config = load_training_config(args.config)
    elif checkpoint_state and isinstance(checkpoint_state.get("config"), dict):
        config = TrainingConfig.from_dict(checkpoint_state["config"])  # type: ignore[arg-type]
    else:
        config = TrainingConfig()
    if args.batch_size:
        config.batch_size = args.batch_size

    train_loader = None
    if not args.validate_only:
        train_loader = create_dataloader(
            split="train",
            window_config=config.window,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    eval_loader = None
    if not args.no_eval:
        eval_loader = create_dataloader(
            split=args.eval_split,
            window_config=config.window,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    model = MultimodalPriceForecaster(config.model)
    model.to(device)

    diffusion_schedule = DiffusionSchedule(num_steps=args.diffusion_steps).to(device)

    optimiser = AdamW(
        model.parameters(),
        lr=config.optimiser.learning_rate,
        betas=config.optimiser.betas,
        weight_decay=config.optimiser.weight_decay,
    )
    total_steps = 0
    if train_loader is not None:
        total_steps = len(train_loader) * args.epochs
    scheduler = _build_scheduler(optimiser, config.optimiser.warmup_ratio, total_steps)

    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    if args.checkpoint:
        checkpoint = _load_checkpoint(args.checkpoint, model, optimiser if not args.validate_only else None, scheduler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(checkpoint.get("best_metric", best_metric))

    if args.validate_only:
        if eval_loader is None:
            raise ValueError("Evaluation split must be available for validation-only mode.")
        metrics = _evaluate(model, eval_loader, diffusion_schedule, device)
        print({"validation": metrics})
        return

    if train_loader is None:
        raise ValueError("Training dataloader could not be constructed.")

    use_amp = config.mixed_precision and device.type == "cuda"

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optimiser,
            scheduler,
            diffusion_schedule,
            device,
            config.grad_clip,
            use_amp,
            args.log_interval,
            global_step,
        )
        global_step = int(train_metrics["global_step"])
        print(f"Training loss: {train_metrics['loss']:.6f}")

        if eval_loader is not None:
            metrics = _evaluate(model, eval_loader, diffusion_schedule, device)
            print({"validation": metrics})
            current_metric = metrics["noise_rmse"]
            is_best = current_metric < best_metric
            if is_best:
                best_metric = current_metric
            state = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "config": asdict(config),
                "best_metric": best_metric,
            }
            checkpoint_path = os.path.join(args.output_dir, "checkpoint_last.pt")
            _save_checkpoint(state, checkpoint_path)
            if is_best:
                _save_checkpoint(state, os.path.join(args.output_dir, "checkpoint_best.pt"))
        else:
            state = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "config": asdict(config),
                "best_metric": best_metric,
            }
            _save_checkpoint(state, os.path.join(args.output_dir, "checkpoint_last.pt"))


if __name__ == "__main__":
    main()

