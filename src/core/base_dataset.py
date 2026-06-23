# src/core/base_dataset.py
from abc import ABC, abstractmethod
from typing import Any


class AbstractDataset(ABC):
    """
    Contract for any dataset used in FL training.
    Dataset knows nothing about FL or protocols.
    """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load dataset from disk."""
        ...

    @abstractmethod
    def get_sample(self, index: int) -> dict[str, Any]:
        """Return a single sample by index."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples."""
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return the list of feature column names."""
        ...
