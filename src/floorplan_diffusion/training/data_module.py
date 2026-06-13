"""PyTorch Lightning DataModule for ResPlan floorplan data.

Wraps :class:`~floorplan_diffusion.data.dataset.ResPlanDataset` and provides
DataLoaders for the deterministic train/eval splits (the split itself lives in
the dataset, seeded over raw pickle indices, so evaluation scripts that load
``set_name="eval"`` see exactly the data held out here).
"""

from __future__ import annotations

import os
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..data.dataset import ResPlanDataset


class ResPlanDataModule(pl.LightningDataModule):
    """Lightning DataModule for ResPlan data.

    Args:
        pickle_path: Path to the ResPlan ``.pkl`` file.
        cache_dir: Optional directory for ``.npz`` tensor caches.
        batch_size: Batch size for both train and val loaders.
        num_workers: DataLoader worker count.
        val_fraction: Fraction of the dataset used for validation (default 0.1).
    """

    def __init__(
        self,
        pickle_path: str | os.PathLike[str],
        cache_dir: str | os.PathLike[str] | None = None,
        batch_size: int = 32,
        num_workers: int = 2,
        val_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self.pickle_path = pickle_path
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction

        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        """Instantiate the deterministic train and eval splits."""
        self.train_dataset = ResPlanDataset(
            pickle_path=self.pickle_path,
            cache_dir=self.cache_dir,
            set_name="train",
            val_fraction=self.val_fraction,
        )
        # Separate instance: disjoint by the seeded raw-index split, and free
        # of the rotation augmentation that set_name="train" enables.
        self.val_dataset = ResPlanDataset(
            pickle_path=self.pickle_path,
            cache_dir=self.cache_dir,
            set_name="eval",
            val_fraction=self.val_fraction,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
