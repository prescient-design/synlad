"""3D Molecule Dataset for handling conformations from pickle files."""

import logging
import pickle
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

from synlad.utils.constants import atomic_number_to_atom_type_index, ph4_type_to_index

logger = logging.getLogger(__name__)


def custom_transform(coords: torch.Tensor, coords_normalizer: float = 1.0) -> torch.Tensor:
    """Process coordinates by removing hydrogens and normalizing."""
    if torch.isnan(coords).any():
        raise ValueError(f"Found NaN values in coordinates: {coords}")

    if torch.isinf(coords).any():
        raise ValueError(f"Found infinite values in coordinates: {coords}")

    # Check for extremely large values
    if coords.abs().max() > 1000.0:
        raise ValueError(
            f"Found extremely large coordinate values (max: {coords.abs().max():.2f}). "
            f"This may indicate corrupted data or incorrect units."
        )

    # Handle single-atom molecules (avoid division by zero in mean)
    if coords.shape[0] == 1:
        return coords / coords_normalizer

    # Standard mean centering and normalization
    coords_mean = coords.mean(dim=0)
    coords_centered = coords - coords_mean

    coords_normalized = coords_centered / coords_normalizer

    # Final validation
    if torch.isnan(coords_normalized).any():
        raise ValueError(
            f"NaN values produced after normalization. Coords before normalization: {coords_centered}"
        )

    if torch.isinf(coords_normalized).any():
        raise ValueError(
            f"Infinite values produced after normalization. Coords before normalization: {coords_centered}"
        )

    return coords_normalized, coords_mean


class MoleculeDataset(Dataset):
    """Dataset for loading 3D molecule conformations from pickle files.

    Each molecule can have multiple conformers, and a conformer is randomly sampled
    at __getitem__ time during training.
    """

    def __init__(
        self,
        data_source: str | Path | list[Chem.Mol],
        max_atoms: int = 50,
        use_all_conformers: bool = True,
        coords_normalizer: float = 1.0,
        removeHs: bool = True,
        random_seed: int | None = None,
        pharmacophores: list[dict[int, dict[str, torch.Tensor]]] | None = None,
    ):
        """
        Initialize MoleculeDataset.

        Args:
            data_source: Path to pickle file or list of RDKit molecules with conformers
            max_atoms: Maximum number of atoms to consider
            use_all_conformers: If True, sample from all conformers. If False, always use first conformer.
            coords_normalizer: Normalizer for coordinates
            removeHs: Whether to remove hydrogen atoms
            random_seed: Random seed for conformer sampling (None for random)
            pharmacophores: List of pharmacophores for each molecule
        """
        self.max_atoms = max_atoms
        self.use_all_conformers = use_all_conformers
        self.coords_normalizer = coords_normalizer
        self.removeHs = removeHs
        self.pharmacophores = pharmacophores

        # Set up random number generator for conformer sampling
        self.rng = random.Random(random_seed)

        # Load molecules
        if isinstance(data_source, (str, Path)):
            self.molecules = self._load_molecules_from_pickle(data_source)
        elif isinstance(data_source, list):
            self.molecules = data_source
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")

        # Process molecules and filter valid ones
        self.valid_molecules, valid_idcs = self._process_and_filter_molecules()
        if self.pharmacophores is not None:
            self.valid_pharmacophores = [self.pharmacophores[i] for i in valid_idcs]

        logger.info(
            f"Loaded {len(self.valid_molecules)} valid molecules from {len(self.molecules)} total molecules"
        )
        if self.use_all_conformers:
            total_conformers = sum(mol_data["num_conformers"] for mol_data in self.valid_molecules)
            logger.info(f"Total conformers available: {total_conformers}")
        else:
            logger.info("Using only first conformer per molecule")

    def _load_molecules_from_pickle(self, file_path: str | Path) -> list[Chem.Mol]:
        """Load molecules from pickle file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")

        if file_path.suffix.lower() not in [".pkl", ".pickle"]:
            raise ValueError(f"Expected .pkl file, got: {file_path.suffix}")

        with open(file_path, "rb") as f:
            molecules = pickle.load(f)

        if not isinstance(molecules, list):
            raise ValueError("Pickle file should contain a list of RDKit molecules")

        # Validate that all items are RDKit molecules
        for i, mol in enumerate(molecules):
            if not isinstance(mol, Chem.Mol):
                raise ValueError(f"Item {i} in pickle file is not an RDKit molecule: {type(mol)}")

        return molecules

    def _process_and_filter_molecules(self) -> list[dict[str, Any]]:
        """Process molecules and filter valid ones, storing metadata for conformer sampling."""
        valid_molecules = []
        valid_idcs = []

        for mol_idx, mol in enumerate(self.molecules):
            try:
                if mol is None:
                    logger.warning(f"Skipping None molecule at index {mol_idx}")
                    continue

                # Filter by number of atoms
                if mol.GetNumHeavyAtoms() > self.max_atoms:
                    logger.warning(
                        f"Skipping molecule {mol_idx}: too many heavy atoms ({mol.GetNumHeavyAtoms()} > {self.max_atoms})"
                    )
                    continue

                # Check if molecule has conformers
                num_conformers = mol.GetNumConformers()
                if num_conformers == 0:
                    logger.warning(f"Skipping molecule {mol_idx}: no conformers found")
                    continue

                # Store molecule data for later conformer sampling
                mol_data = {
                    "mol": mol,
                    "mol_idx": mol_idx,
                    "num_conformers": num_conformers,
                    "smiles": None,  # Will be computed when needed
                }

                # Pre-compute SMILES (this doesn't depend on conformer)
                smiles = Chem.MolToSmiles(mol)
                mol_data["smiles"] = smiles

                valid_idcs.append(mol_idx)
                valid_molecules.append(mol_data)

            except Exception as e:
                logger.error(f"Error processing molecule {mol_idx}: {e}")
                continue

        return valid_molecules, valid_idcs

    def __len__(self) -> int:
        """Return the number of molecules (not conformers)."""
        return len(self.valid_molecules)

    def __getitem__(self, idx: int) -> Data:
        """Get a single molecule with a randomly sampled conformer."""
        mol_data = self.valid_molecules[idx]
        mol = mol_data["mol"]
        mol_idx = mol_data["mol_idx"]
        num_conformers = mol_data["num_conformers"]
        smiles = mol_data["smiles"]
        if self.pharmacophores is not None:
            pharmacophore = self.valid_pharmacophores[idx]
        else:
            pharmacophore = None

        # Sample conformer ID
        if self.use_all_conformers and num_conformers > 1:
            conformer_id = self.rng.randint(0, num_conformers - 1)
            if pharmacophore is not None and conformer_id not in pharmacophore:
                conformer_id = 0
        else:
            conformer_id = 0  # Always use first conformer

        # Process molecule (remove Hs if needed)
        if self.removeHs:
            mol = deepcopy(mol)
            mol = Chem.RemoveAllHs(mol)

        try:
            # Get atomic information
            atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            atom_type_indices = [
                atomic_number_to_atom_type_index(atomic_num) for atomic_num in atomic_numbers
            ]
            atom_types = torch.LongTensor(atom_type_indices)

            # Get coordinates for the selected conformer
            conformer = mol.GetConformer(conformer_id)
            coords = torch.Tensor(conformer.GetPositions())
            coords, coords_mean = custom_transform(coords, self.coords_normalizer)

            num_atoms = atom_types.shape[0]
            num_nodes = torch.LongTensor([num_atoms])

            # Add pharmacophore data if available
            if pharmacophore is not None:
                pharmacophore_conf = pharmacophore[conformer_id]
                ph4_coords = pharmacophore_conf["coords"]
                ph4_channels = [
                    ph4_type_to_index(int(ph4_channel))
                    for ph4_channel in pharmacophore_conf["channels"]
                ]
                ph4_channels = torch.LongTensor(ph4_channels)
                num_ph4_channels = ph4_channels.shape[0]
                ph4_coords = ph4_coords - coords_mean
                ph4_coords = ph4_coords / self.coords_normalizer

            # Create PyTorch Geometric Data object
            data = Data(
                atom_types=atom_types,
                pos=coords,
                num_atoms=torch.LongTensor([num_atoms]),
                num_nodes=num_nodes,  # special attribute used for PyG batching
                token_idx=torch.arange(num_atoms),
            )

            # Add metadata
            data.smiles = smiles
            data.mol_idx = mol_idx
            data.conformer_id = conformer_id
            data.num_conformers = num_conformers

            if pharmacophore is not None:
                data_ph4 = Data(
                    ph4_coords=ph4_coords,
                    ph4_channels=ph4_channels,
                    num_ph4_channels=torch.LongTensor([num_ph4_channels]),
                    num_nodes=torch.LongTensor([num_ph4_channels]),
                    token_idx=torch.arange(num_ph4_channels),
                )
                data_ph4.smiles = smiles
                data_ph4.mol_idx = mol_idx
                data_ph4.conformer_id = conformer_id
                return data, data_ph4

            return data

        except Exception as e:
            logger.error(f"Error processing conformer {conformer_id} of molecule {mol_idx}: {e}")
            # Fallback to first conformer if random conformer fails
            if conformer_id != 0:
                try:
                    conformer = mol.GetConformer(0)
                    coords = torch.Tensor(conformer.GetPositions())
                    coords = custom_transform(coords, self.coords_normalizer)

                    data = Data(
                        atom_types=atom_types,
                        pos=coords,
                        num_atoms=torch.LongTensor([num_atoms]),
                        num_nodes=num_nodes,
                        token_idx=torch.arange(num_atoms),
                    )

                    data.smiles = smiles
                    data.mol_idx = mol_idx
                    data.conformer_id = 0
                    data.num_conformers = num_conformers

                    if pharmacophore is not None:
                        data_ph4 = Data(
                            ph4_coords=ph4_coords,
                            ph4_channels=ph4_channels,
                            num_ph4_channels=torch.LongTensor([num_ph4_channels]),
                            num_nodes=torch.LongTensor([num_ph4_channels]),
                            token_idx=torch.arange(num_ph4_channels),
                        )
                        data_ph4.smiles = smiles
                        data_ph4.mol_idx = mol_idx
                        data_ph4.conformer_id = 0
                        return data, data_ph4

                    return data
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to process any conformer for molecule {mol_idx}: {e2}"
                    ) from e2
            else:
                raise RuntimeError(f"Failed to process molecule {mol_idx}: {e}") from e

    def get_molecule_info(self, idx: int) -> dict[str, Any]:
        """Get information about a specific molecule without sampling."""
        mol_data = self.valid_molecules[idx]
        return {
            "mol_idx": mol_data["mol_idx"],
            "smiles": mol_data["smiles"],
            "num_conformers": mol_data["num_conformers"],
            "num_atoms": mol_data["mol"].GetNumAtoms(),
            "num_heavy_atoms": mol_data["mol"].GetNumHeavyAtoms(),
        }

    def set_random_seed(self, seed: int):
        """Set random seed for conformer sampling."""
        self.rng = random.Random(seed)

    def __repr__(self) -> str:
        total_conformers = sum(mol_data["num_conformers"] for mol_data in self.valid_molecules)
        return f"MoleculeDataset(num_molecules={len(self.valid_molecules)}, total_conformers={total_conformers})"
