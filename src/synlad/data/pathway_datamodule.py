"""Lightning DataModule for combined 3D molecules and synthesis pathways."""

import logging
from pathlib import Path

import numpy.random as np_random
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from synlad.data.components import synthesis_molecular_graphs as molecular_graphs
from synlad.data.components.pathway_collator import PathwayCollateFunction
from synlad.data.components.pathway_dataset import PathwayDataset
from synlad.tokenization.synthesis_serialization import (
    EarliestFirst,
    MultiplierWeighter,
    RxnTypeWeighter,
)
from synlad.tokenization.synthesis_vocab import SPECIAL_TOKENS_LIBRARY, TokenLibrary

logger = logging.getLogger(__name__)


class PathwayDataModule(LightningDataModule):
    """Lightning DataModule for combined 3D molecules and synthesis pathways.
    The molecule dataset samples conformers at runtime, so each molecule
    appears only once in the dataset but can have multiple conformers sampled during training.
    """

    def __init__(
        self,
        train_pathway_file: str | Path | None = None,
        val_pathway_file: str | Path | None = None,
        test_pathway_file: str | Path | None = None,
        # Molecule processing parameters
        max_atoms: int = 50,
        use_all_conformers: bool = True,
        coords_normalizer: float = 1.0,
        removeHs: bool = True,
        # DataLoader parameters
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        # Collate function parameters
        nested_tensors: bool = True,
        max_seq_len: int = 500,
        enable_conditioning: bool = True,
        prefix_free_generation: bool = True,
        earliest_bbs_first: bool = True,
        # Token weighting parameters
        token_weightings: dict[str, float] | None = None,
        # Pharmacophore parameters
        do_pharmacophores: bool = False,
        # Optional paths for downstream modules (e.g., LDM module)
        bincount_path: str | Path | None = None,
        train_smiles_path: str | Path | None = None,
    ) -> None:
        """
        Initialize PathwayDataModule.

        Args:
            train_pathway_file: Path to training pathway file with molecules
            val_pathway_file: Path to validation pathway file with molecules
            test_pathway_file: Path to test pathway file with molecules

            # Common parameters
            max_atoms: Maximum number of atoms to consider
            use_all_conformers: If True, sample from all conformers at runtime
            coords_normalizer: Normalizer for coordinates
            removeHs: Whether to remove hydrogen atoms
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            seed: Random seed for reproducibility

            # Collate function parameters
            nested_tensors: Whether to use nested tensors
            max_seq_len: Maximum sequence length
            enable_conditioning: Whether to enable conditioning
            prefix_free_generation: Whether to use prefix-free generation
            earliest_bbs_first: Whether to use earliest building blocks first
            token_weightings: Dictionary of token weightings for reaction type weighting
            do_pharmacophores: Whether to use pharmacophores
            bincount_path: Optional path to bincount file (used by downstream modules)
            train_smiles_path: Optional path to training SMILES file (used by downstream modules)
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: PathwayDataset | None = None
        self.data_val: PathwayDataset | None = None
        self.data_test: PathwayDataset | None = None

        self.collate_fn: PathwayCollateFunction | None = None

        self.batch_size_per_device = batch_size
        self.bincount_path = bincount_path
        self.train_smiles_path = train_smiles_path

    def prepare_data(self) -> None:
        """Download or prepare data if needed."""
        # This method is called only on the main process
        # Validate file paths exist if provided
        for file_path in [
            self.hparams.train_pathway_file,
            self.hparams.val_pathway_file,
            self.hparams.test_pathway_file,
        ]:
            if file_path is not None and not Path(file_path).exists():
                raise FileNotFoundError(f"Pathway file not found: {file_path}")

    def _load_split(self, pathway_file: str | Path | None) -> PathwayDataset:
        return PathwayDataset(
            pathways=pathway_file,
            max_atoms=self.hparams.max_atoms,
            use_all_conformers=self.hparams.use_all_conformers,
            coords_normalizer=self.hparams.coords_normalizer,
            removeHs=self.hparams.removeHs,
            random_seed=self.hparams.seed,
            do_pharmacophores=self.hparams.do_pharmacophores,
        )

    def setup(self, stage: str | None = None) -> None:
        """Load only the splits required for `stage`.

        Lightning calls this with `"fit"`, `"validate"`, `"test"`, `"predict"`, or `None`
        (manual). Loading is idempotent — splits already loaded are not reloaded.
        """
        need_train = stage in ("fit", None)
        need_val = stage in ("fit", "validate", None)
        need_test = stage in ("test", "predict", None)

        if need_train and self.data_train is None:
            self.data_train = self._load_split(self.hparams.train_pathway_file)
        if need_val and self.data_val is None:
            self.data_val = self._load_split(self.hparams.val_pathway_file)
        if need_test and self.data_test is None:
            self.data_test = self._load_split(self.hparams.test_pathway_file)

        def _count_conformers(dataset: PathwayDataset | None) -> int | None:
            if dataset is None:
                return None
            return sum(
                mol.GetNumConformers()
                for mol in dataset.molecule_dataset.molecules
                if mol is not None
            )

        loaded_summary = ", ".join(
            f"{len(d)} {name}"
            for name, d in (
                ("train", self.data_train),
                ("val", self.data_val),
                ("test", self.data_test),
            )
            if d is not None
        )
        if loaded_summary:
            logger.info(f"Loaded {loaded_summary} combined samples")
            conformer_summary = ", ".join(
                f"{_count_conformers(d)} {name}"
                for name, d in (
                    ("train", self.data_train),
                    ("val", self.data_val),
                    ("test", self.data_test),
                )
                if d is not None
            )
            logger.info(f"  - Available conformers: {conformer_summary}")

        # Setup collate function from whichever split is loaded — building_block_vocab
        # is identical across splits (sourced from the pathway pickle).
        vocab_source = self.data_train or self.data_val or self.data_test
        if vocab_source is not None and self.collate_fn is None:
            token_weightings = self.hparams.token_weightings
            if token_weightings is None:
                token_weightings = {"<REV_RXN>": 0.0}
            prompt_to_action_probabilizer = RxnTypeWeighter(token_weightings)
            bb_graphs = [
                molecular_graphs.from_smiles(smi) for smi in vocab_source.building_block_vocab
            ]
            bb_token_lib = TokenLibrary(
                vocab_source.building_block_vocab,
                start_idx=SPECIAL_TOKENS_LIBRARY.end_idx + 1,
                molecule_tokens=True,
            )

            if self.hparams.earliest_bbs_first:
                prompt_to_action_probabilizer = MultiplierWeighter(
                    [prompt_to_action_probabilizer, EarliestFirst(bb_token_lib)]
                )

            self.collate_fn = PathwayCollateFunction(
                rng=np_random.RandomState(self.hparams.seed),
                prompt_to_action_probabilizer=lambda _: prompt_to_action_probabilizer,
                bb_graphs=bb_graphs,
                bb_tkn_library=bb_token_lib,
                nested_tensors=self.hparams.nested_tensors,
                max_seq_len=self.hparams.max_seq_len,
                enable_conditioning=self.hparams.enable_conditioning,
                prefix_free_generation=self.hparams.prefix_free_generation,
            )

    def train_dataloader(self):
        """Create and return the train dataloader."""
        if self.data_train is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.get("num_workers", 0),
            pin_memory=self.hparams.get("pin_memory", False),
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        """Create and return the validation dataloader."""
        if self.data_val is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.get("num_workers", 0),
            pin_memory=self.hparams.get("pin_memory", False),
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        """Create and return the test dataloader."""
        if self.data_test is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.get("num_workers", 0),
            pin_memory=self.hparams.get("pin_memory", False),
            shuffle=False,
            collate_fn=self.collate_fn,
        )
