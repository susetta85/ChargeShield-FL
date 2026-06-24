# src/adapters/elaadnl_dataset.py
import csv
from pathlib import Path
from typing import Any

from src.core.base_dataset import AbstractDataset


class ElaadNLDataset(AbstractDataset):
    """
    Adapter for the ElaadNL public EV charging dataset.
    Knows about CSV format, but nothing about FL or protocols.
    """

    FEATURE_NAMES = [
        "session_id",
        "start_time",
        "end_time",
        "total_energy_kwh",
        "max_power_kw",
        "charging_mode",
        "soc_percent",
        "temperature_c",
        "anomaly_label",
        "transaction_it",
        "node_id",
        "cluster_id",
        "error_code",
    ]

    def __init__(self):
        self._data: list[dict[str, Any]] = []

    def load(self, path: str) -> None:
        """Load ElaadNL CSV from disk."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {path}")

        self._data = []
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._data.append(self._parse_row(row))

    def _parse_row(self, row: dict[str, str]) -> dict[str, Any]:
        """Convert raw CSV row to typed dict."""
        return {
            "session_id": row.get("session_id", ""),
            "start_time": row.get("start_time", ""),
            "end_time": row.get("end_time", ""),
            "total_energy_kwh": float(row.get("total_energy_kwh", 0)),
            "max_power_kw": float(row.get("max_power_kw", 0)),
            "charging_mode": row.get("charging_mode", "AC"),
            "soc_percent": float(row.get("soc_percent", 0)),
            "temperature_c": float(row.get("temperature_c", 0)),
            "anomaly_label": int(row.get("anomaly_label", 0)),
            "transaction_id": row.get("transaction_id", ""),
            "node_id": row.get("node_id", ""),
            "cluster_id": row.get("cluster_id", ""),
            "error_code": row.get("error_code") or None,
        }

    def get_sample(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self._data):
            raise IndexError(f"Index {index} out of range (dataset size: {len(self._data)})")
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def get_feature_names(self) -> list[str]:
        return self.FEATURE_NAMES
