"""Dataset utilities for multimodal stock forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset


@dataclass
class WindowConfig:
    """Configuration for slicing price windows."""

    history_length: int = 30
    forecast_horizon: int = 30
    price_columns: Tuple[str, ...] = ("open", "high", "low", "close", "volume")


def normalise_prices(prices: np.ndarray) -> np.ndarray:
    """Convert raw prices to log returns for improved stationarity."""

    log_prices = np.log(np.clip(prices, a_min=1e-6, a_max=None))
    return np.diff(log_prices, axis=0, prepend=log_prices[:, :1, :])


class StockNewsDataset(Dataset):
    """Wrap the Hugging Face dataset to emit multimodal training examples."""

    def __init__(
        self,
        split: str,
        window_config: WindowConfig | None = None,
        text_fields: Tuple[str, str] = ("Title", "Content"),
        dataset_name: str = "oliverwang15/us_stock_news_with_price",
    ) -> None:
        super().__init__()
        self.window_config = window_config or WindowConfig()
        self.dataset = load_dataset(dataset_name, split=split)
        self.text_fields = text_fields

    def __len__(self) -> int:  # noqa: D401 - delegated behaviour
        return len(self.dataset)

    def _build_text(self, record: Dict[str, str]) -> str:
        return " \n\n".join(str(record[field]) for field in self.text_fields if record.get(field))

    def _window_prices(self, record: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.window_config
        history = np.stack([record[col][: cfg.history_length] for col in cfg.price_columns], axis=-1)
        future = np.stack(
            [record[col][cfg.history_length : cfg.history_length + cfg.forecast_horizon] for col in cfg.price_columns],
            axis=-1,
        )
        return history, future

    def __getitem__(self, index: int) -> Dict[str, np.ndarray | str]:  # noqa: D401 - torch Dataset contract
        record = self.dataset[index]
        history, future = self._window_prices(record)
        history = normalise_prices(history)
        future = normalise_prices(future)
        text = self._build_text(record)
        return {
            "timeseries_history": history.astype(np.float32),
            "timeseries_future": future.astype(np.float32),
            "text": text,
        }


__all__ = ["StockNewsDataset", "WindowConfig", "normalise_prices"]
