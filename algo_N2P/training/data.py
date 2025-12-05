"""Data loading helpers for training and evaluation."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import StockNewsDataset, WindowConfig


Batch = Dict[str, torch.Tensor | List[str]]


def collate_examples(examples: Iterable[Dict[str, np.ndarray | str]]) -> Batch:
    """Collate dataset items into batched tensors."""

    histories = torch.from_numpy(np.stack([ex["timeseries_history"] for ex in examples])).float()
    futures = torch.from_numpy(np.stack([ex["timeseries_future"] for ex in examples])).float()
    texts = [str(ex["text"]) for ex in examples]
    return {
        "timeseries_history": histories,
        "timeseries_future": futures,
        "text": texts,
    }


def create_dataloader(
    split: str,
    window_config: WindowConfig,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Instantiate a :class:`~torch.utils.data.DataLoader` for a dataset split."""

    dataset = StockNewsDataset(split=split, window_config=replace(window_config))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_examples,
    )


__all__ = ["Batch", "collate_examples", "create_dataloader"]

