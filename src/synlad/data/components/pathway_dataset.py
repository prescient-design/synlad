"""Combined dataset of 3D molecules and synthesis pathways."""

import logging
import pickle
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
from torch.utils.data import Dataset

from synlad.data.components.molecule_dataset import MoleculeDataset
from synlad.data.components.synthesis_dataset import RxnNetTask, RxnNetTaskDataset

logger = logging.getLogger(__name__)


def extract_molecules_from_rxn_tasks(pathways: list[RxnNetTask]) -> list[Chem.Mol]:
    """
    Extract molecules from pathway data where conformers are stored in molecule_conformations.

    Args:
        pathways: List of RxnNetTask objects with molecule_conformations field

    Returns:
        List of RDKit molecules with conformers
    """
    molecules = []
    for pathway in pathways:
        if (
            hasattr(pathway, "molecule_conformations")
            and pathway.molecule_conformations is not None
        ):
            molecules.append(pathway.molecule_conformations)
        else:
            raise ValueError(f"Pathway {pathway} has no molecule_conformations field")
    return molecules


def extract_pharmacophores_from_rxn_tasks(
    pathways: list[RxnNetTask],
) -> list[dict[int, dict[str, torch.Tensor]]]:
    """
    Extract pharmacophores from pathway data where pharmacophores are stored in pharmacophores_dict.
    Args:
        pathways: List of RxnNetTask objects with pharmacophores_dict field

    Returns:
        List of pharmacophores
    """
    pharmacophores = []
    for pathway in pathways:
        if hasattr(pathway, "pharmacophores_dict") and pathway.pharmacophores_dict is not None:
            pharmacophores.append(pathway.pharmacophores_dict)
        else:
            return None
    return pharmacophores


def load_pathway_data(pathway_file: str | Path) -> tuple[list[RxnNetTask], list[str]]:
    """
    Load pathway data from a pickle file.

    Args:
        pathway_file: Path to pickle file containing pathway data

    Returns:
        Tuple of (List of RxnNetTask objects, building block vocabulary)
    """
    try:
        with open(pathway_file, "rb") as f:
            data = pickle.load(f)
    except (OSError, FileNotFoundError, pickle.UnpicklingError) as e:
        raise ValueError(f"Failed to load pathway data from {pathway_file}: {e}") from e

    if not isinstance(data, dict) or "dataset" not in data or "building_block_vocab" not in data:
        raise ValueError(
            "Invalid pathway data format. Expected dict with 'dataset' and 'building_block_vocab' keys"
        )

    return data["dataset"], data["building_block_vocab"]


class PathwayDataset(Dataset):
    """Combined dataset of 3D molecules and synthesis pathways.

    This dataset combines MoleculeDataset and RxnNetTaskDataset to provide
    both 3D molecular conformations and synthesis pathway information for
    the same molecules, ensuring consistent indexing across both datasets.

    Supports two usage patterns:
    1. Separate molecules and pathways files/lists
    2. Molecules extracted from pathways beforehand
    """

    def __init__(
        self,
        pathways: str | Path | list[RxnNetTask],
        max_atoms: int = 50,
        use_all_conformers: bool = True,
        coords_normalizer: float = 1.0,
        removeHs: bool = True,
        random_seed: int | None = None,
        do_pharmacophores: bool = False,
    ):
        """
        Initialize PathwayDataset.

        Args:
            pathways: Path to pickle file or list of RxnNetTask objects for synthesis pathways
            max_atoms: Maximum number of atoms to consider
            use_all_conformers: If True, sample from all conformers at runtime
            coords_normalizer: Normalizer for coordinates
            removeHs: Whether to remove hydrogen atoms
            random_seed: Random seed for conformer sampling (None for random)
        """
        # Load pathway data based on input type
        if isinstance(pathways, (str, Path)):
            self.pathways, self.building_block_vocab = load_pathway_data(pathways)
        elif isinstance(pathways, list):
            self.pathways = pathways
            self.building_block_vocab = []  # Empty vocab when pathways provided directly
        else:
            raise ValueError(f"Unsupported pathways type: {type(pathways)}")

        self.molecules = extract_molecules_from_rxn_tasks(self.pathways)
        logger.info(f"Extracted {len(self.molecules)} molecules from pathways")
        self.pharmacophores = extract_pharmacophores_from_rxn_tasks(self.pathways)
        self.do_pharmacophores = do_pharmacophores
        if self.do_pharmacophores:
            assert self.pharmacophores is not None, "Pharmacophores not found in pathways"
            logger.info(f"Extracted {len(self.pharmacophores)} pharmacophores from pathways")
            assert len(self.molecules) == len(self.pharmacophores), (
                f"Molecules and pharmacophores must have the same length. Got {len(self.molecules)} molecules and {len(self.pharmacophores)} pharmacophores"
            )

        # Create the molecule dataset
        self.molecule_dataset = MoleculeDataset(
            data_source=self.molecules,
            max_atoms=max_atoms,
            use_all_conformers=use_all_conformers,
            coords_normalizer=coords_normalizer,
            removeHs=removeHs,
            random_seed=random_seed,
            pharmacophores=self.pharmacophores,
        )

        # Create the pathway dataset
        if not isinstance(self.pathways, list):
            raise ValueError("Pathways must be a list of RxnNetTask objects at this point")
        self.pathway_dataset = RxnNetTaskDataset(self.pathways)

        assert len(self.molecule_dataset) == len(self.pathway_dataset), (
            f"Molecule and pathway datasets must have the same length. Got {len(self.molecule_dataset)} molecules and {len(self.pathway_dataset)} pathways"
        )

    def __len__(self):
        """Return the length of the molecule dataset."""
        return len(self.molecule_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a combined data point with both molecular and pathway information.

        Args:
            idx: Index of the data point

        Returns:
            Dictionary containing both molecular and pathway data
        """
        # Get molecular data
        mol_data, ph4_data = self.molecule_dataset[idx]

        # Get pathway data - use mol_idx to find corresponding pathway
        mol_idx = mol_data.mol_idx
        pathway_data = None

        # Find the pathway for this molecule
        if mol_idx < len(self.pathway_dataset):
            pathway_data = self.pathway_dataset[mol_idx]

        if self.do_pharmacophores:
            return {
                "molecule": mol_data,
                "pharmacophore": ph4_data,
                "pathway": pathway_data,
                "mol_idx": mol_idx,
            }
        else:
            return {
                "molecule": mol_data,
                "pathway": pathway_data,
                "mol_idx": mol_idx,
            }

    def get_molecule_dataset(self) -> MoleculeDataset:
        """Get the underlying molecule dataset."""
        return self.molecule_dataset

    def get_pathway_dataset(self) -> RxnNetTaskDataset:
        """Get the underlying pathway dataset."""
        return self.pathway_dataset
