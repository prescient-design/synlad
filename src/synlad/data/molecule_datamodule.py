"""Lightning DataModule for 3D molecule conformations using PyG format."""

import logging
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch_geometric.loader import DataLoader as PyGDataLoader

from synlad.data.components.molecule_dataset import MoleculeDataset

logger = logging.getLogger(__name__)


class MoleculeDataModule(LightningDataModule):
    """Lightning DataModule for 3D molecule conformations in PyG format."""

    def __init__(
        self,
        train_data_source: str | Path | list,
        val_data_source: str | Path | list,
        test_data_source: str | Path | list,
        max_atoms: int = 50,
        use_all_conformers: bool = True,
        coords_normalizer: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        removeHs: bool = True,
    ) -> None:
        """
        Initialize MoleculeDataModule for 3D conformations.

        Args:
            train_data_source: Path to pickle file or list of RDKit molecules with conformers
            val_data_source: Path to pickle file or list of RDKit molecules with conformers
            test_data_source: Path to pickle file or list of RDKit molecules with conformers
            max_atoms: Maximum number of atoms to consider
            use_all_conformers: If True, use all conformers as separate data points
            coords_normalizer: Normalizer for coordinates
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: torch.utils.data.Dataset | None = None
        self.data_val: torch.utils.data.Dataset | None = None
        self.data_test: torch.utils.data.Dataset | None = None

    def prepare_data(self) -> None:
        """Download or prepare data if needed."""
        # This method is called only on the main process
        pass

    def setup(self, stage: str | None = None) -> None:
        """Load data and create train/val/test splits."""

        # Load dataset only if not already loaded
        if not self.data_train and not self.data_val and not self.data_test:
            # Create full dataset
            logger.info(f"Loading dataset from {self.hparams.train_data_source} ...")
            self.data_train = MoleculeDataset(
                data_source=self.hparams.train_data_source,
                max_atoms=self.hparams.max_atoms,
                use_all_conformers=self.hparams.use_all_conformers,
                coords_normalizer=self.hparams.coords_normalizer,
                removeHs=self.hparams.removeHs,
            )
            self.data_val = MoleculeDataset(
                data_source=self.hparams.val_data_source,
                max_atoms=self.hparams.max_atoms,
                use_all_conformers=self.hparams.use_all_conformers,
                coords_normalizer=self.hparams.coords_normalizer,
                removeHs=self.hparams.removeHs,
            )
            self.data_test = MoleculeDataset(
                data_source=self.hparams.test_data_source,
                max_atoms=self.hparams.max_atoms,
                use_all_conformers=self.hparams.use_all_conformers,
                coords_normalizer=self.hparams.coords_normalizer,
                removeHs=self.hparams.removeHs,
            )

            if self.data_train and self.data_val and self.data_test:
                logger.info(
                    f"Loaded {len(self.data_train)} train, {len(self.data_val)} val, {len(self.data_test)} test conformations from {len(self.data_train.molecules)} molecules"
                )

    def train_dataloader(self):
        """Create and return the train dataloader."""
        return PyGDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> PyGDataLoader:
        """Create and return the validation dataloader."""
        return PyGDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> PyGDataLoader:
        """Create and return the test dataloader."""
        return PyGDataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
