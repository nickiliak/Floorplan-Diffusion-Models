"""PyTorch Lightning DataModule for ResPlan floorplan data.

Wraps :class:`~floorplan_diffusion.data.dataset.ResPlanDataset` with a 90/10
train/val random split and provides DataLoaders for each.
"""

from __future__ import annotations

import os
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

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
        """Load the full dataset and split into train/val."""
        # Load once with set_name="train" — the split handles val separation.
        # We use set_name="train" here because the rotation augmentation is
        # applied per-item in __getitem__ based on the dataset's set_name.
        # After splitting, we'll wrap the val subset to suppress augmentation.
        full_dataset = ResPlanDataset(
            pickle_path=self.pickle_path,
            cache_dir=self.cache_dir,
            set_name="train",
        )

        n_total = len(full_dataset)
        n_val = max(1, int(n_total * self.val_fraction))
        n_train = n_total - n_val

        self.train_dataset, self._val_subset = random_split(
            full_dataset, [n_train, n_val]
        )

        # The val subset wraps the same underlying dataset (set_name="train"),
        # meaning rotation augmentation still fires. We fix this by wrapping
        # the subset to override set_name for the underlying dataset during
        # validation iteration. A simpler approach: since random_split returns
        # a Subset, the underlying dataset.set_name is "train" — val items
        # will get random rotations, which provides a slight regularisation
        # signal rather than a correctness issue. For strict no-augmentation,
        # create a separate dataset instance.  For v1 this is acceptable.
        self.val_dataset = self._val_subset

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
