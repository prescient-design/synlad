"""Molecular graph featurization (SMILES to PyG Data) with Morgan fingerprints."""

import functools
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric import data as pyg_data
from torch_geometric.utils import smiles as pyg_smiles

RDLogger.DisableLog("rdApp.*")

ATOMIC_NUMBERS_USED = [
    1,
    2,
    3,
    5,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    34,
    35,
    37,
    38,
    39,
    40,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    63,
    70,
    74,
    75,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    85,
    100,
]

# Create Morgan fingerprint generator
_MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048
)  # currently hardcoded in...


@functools.lru_cache(maxsize=50_000)
def _compute_fingerprint_cached(smiles: str) -> tuple:
    """Cached fingerprint computation. Returns tuple for hashability."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Add a warning here too
        warnings.warn(
            f"Invalid SMILES provided to _compute_fingerprint_cached: '{smiles}', featurized to empty graph.",
            stacklevel=2,
        )
        mol = Chem.MolFromSmiles("")
    fp = _MORGAN_FP_GENERATOR.GetFingerprintAsNumPy(mol)
    return tuple(fp)


def from_smiles(
    smiles: str,
    mol_modififier: Callable[[Chem.Mol], Chem.Mol] = None,
    from_rdmol: Callable[[Chem.Mol], pyg_data.Data] = None,
    logger: logging.Logger = None,
) -> pyg_data.Data:
    """Modified version of pytorch_geometric.utils.smiles.from_smiles that allows for more complex featurization.

    Args:
        smiles: The SMILES string.
        mol_modififier: A function that modifies the RDKit molecule. For instance if you want to add hydrogens, sanitize,
            or kekulize the molecule.
        from_rdmol: A function that converts the RDKit molecule to a PyTorch Geometric data object.
        logger: A logger to log warnings.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles("")
        if logger is not None:
            logger.warning(f"Invalid SMILES: {smiles}, featurized to empty graph.")
    if mol_modififier is not None:
        mol = mol_modififier(mol)

    if from_rdmol is None:
        from_rdmol = pyg_smiles.from_rdmol
    data = from_rdmol(mol)
    data.smiles = smiles

    # Add fingerprint
    fp_tuple = _compute_fingerprint_cached(smiles)
    data.fingerprint = torch.tensor(fp_tuple, dtype=torch.float32)

    return data


class Featurizer(Protocol):
    def __call__(self, value: Any) -> np.ndarray:
        """Convert input value to feature representation.

        Args:
            value: The input value to featurize (e.g., atomic number, bond type, boolean)

        Returns:
            np.ndarray: Feature array representation of the input value
        """
        ...


class OneHotFeaturizer(Featurizer):
    def __init__(self, options: list, fail_on_unknown: bool = False):
        self.fail_on_unknown = fail_on_unknown
        self.options_to_index = {option: index for index, option in enumerate(options)}
        self.num = len(options)

    def __call__(self, picked_option) -> int:
        out = np.zeros(self.num, dtype=np.float32)
        try:
            indx = self.options_to_index[picked_option]
        except KeyError:
            if self.fail_on_unknown:
                raise ValueError(f"Unknown option: {picked_option}") from None
            else:
                return out
        out[indx] = 1
        return out


class BoolFeaturizer(Featurizer):
    def __call__(self, picked_option: bool) -> int:
        return np.array([int(picked_option)], dtype=np.float32)


class IndexFeaturizer(Featurizer):
    def __init__(self, options: list):
        self.options_to_index = {option: index for index, option in enumerate(options)}
        self.num = len(options)

    def __call__(self, picked_option) -> int:
        return np.array([self.options_to_index[picked_option]], dtype=np.float32)


@dataclass
class MolFeaturizer:
    """
    Variant of `from_rdmol` from PyTorch Geometric that allows for more complex and _customizable_ featurization.
    """

    # Atom featurizers
    atomic_num: Featurizer = field(
        default_factory=lambda: IndexFeaturizer(pyg_smiles.x_map["atomic_num"])
    )
    chirality: Featurizer = field(
        default_factory=lambda: IndexFeaturizer(pyg_smiles.x_map["chirality"])
    )
    degree: Featurizer = field(default_factory=lambda: IndexFeaturizer(pyg_smiles.x_map["degree"]))
    formal_charge: Featurizer = field(
        default_factory=lambda: IndexFeaturizer(pyg_smiles.x_map["formal_charge"])
    )
    num_hs: Featurizer = field(default_factory=lambda: IndexFeaturizer(pyg_smiles.x_map["num_hs"]))
    num_radical_electrons: Featurizer = field(
        default_factory=lambda: IndexFeaturizer(pyg_smiles.x_map["num_radical_electrons"])
    )
    hybridization: Featurizer = field(
        default_factory=lambda: IndexFeaturizer(pyg_smiles.x_map["hybridization"])
    )
    is_aromatic: Featurizer = field(default_factory=lambda: BoolFeaturizer())
    is_in_ring: Featurizer = field(default_factory=lambda: BoolFeaturizer())

    # Edge featurizers
    bond_type: Featurizer = field(
        default_factory=lambda: IndexFeaturizer(pyg_smiles.e_map["bond_type"])
    )
    stereo: Featurizer = field(default_factory=lambda: IndexFeaturizer(pyg_smiles.e_map["stereo"]))
    is_conjugated: Featurizer = field(default_factory=lambda: BoolFeaturizer())

    def __call__(self, mol: Chem.Mol) -> pyg_data.Data:
        """Convert an RDKit molecule to a PyTorch Geometric Data object."""
        assert isinstance(mol, Chem.Mol)

        # Featurize atoms
        xs = []
        for atom in mol.GetAtoms():
            row: list[int] = []
            row.append(self.atomic_num(atom.GetAtomicNum()))
            row.append(self.chirality(str(atom.GetChiralTag())))
            row.append(self.degree(atom.GetTotalDegree()))
            row.append(self.formal_charge(atom.GetFormalCharge()))
            row.append(self.num_hs(atom.GetTotalNumHs()))
            row.append(self.num_radical_electrons(atom.GetNumRadicalElectrons()))
            row.append(self.hybridization(str(atom.GetHybridization())))
            row.append(self.is_aromatic(atom.GetIsAromatic()))
            row.append(self.is_in_ring(atom.IsInRing()))
            xs.append(np.concatenate(row))

        x = torch.tensor(np.stack(xs), dtype=torch.long)

        # Featurize bonds
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            e = []
            e.append(self.bond_type(str(bond.GetBondType())))
            e.append(self.stereo(str(bond.GetStereo())))
            e.append(self.is_conjugated(bond.GetIsConjugated()))
            e = np.concatenate(e)

            edge_indices += [[i, j], [j, i]]
            edge_attrs += [e, e]

        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(np.stack(edge_attrs), dtype=torch.long)

        if edge_index.numel() > 0:  # Sort indices.
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        return pyg_data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def from_data_list(data_list: list[pyg_data.Data]) -> pyg_data.Batch:
    """Custom version of Batch.from_data_list that explicitly batches fingerprints.

    Note: PyTorch Geometric could handle fingerprint batching automatically if we registered
    the attribute with the correct cat_dim (cat_dim=0 or None? for graph-level attributes). However,
    we're being explicit here to ensure fingerprints are stacked correctly and have a clear
    plural name (fingerprints vs fingerprint) in the batched output.

    Args:
        data_list: List of PyG Data objects with fingerprint attribute

    Returns:
        Batch object with fingerprints stacked as batch.fingerprints [num_graphs, fingerprint_dim]
    """
    # Exclude fingerprint from automatic batching since we handle it explicitly
    batch = pyg_data.Batch.from_data_list(data_list, exclude_keys=["fingerprint"])
    # Stack all fingerprints into a single tensor [num_graphs, fingerprint_dim]
    batch.fingerprints = torch.stack([g.fingerprint for g in data_list])
    assert batch.fingerprints.shape[0] == batch.num_graphs, (
        f"Fingerprint shape {batch.fingerprints.shape} does not match number of graphs {batch.num_graphs}"
    )
    return batch


def to_data_list(batch: pyg_data.Batch) -> list[pyg_data.Data]:
    """Custom version of Batch.to_data_list that properly splits fingerprints back to individual graphs.

    PyTorch Geometric's default to_data_list() doesn't know about our custom 'fingerprints' attribute
    (which we excluded from automatic batching), so we need to manually split it back to individual
    graphs as 'fingerprint' attributes.

    Args:
        batch: Batch object with fingerprints attribute [num_graphs, fingerprint_dim]

    Returns:
        List of Data objects, each with its fingerprint attribute restored
    """
    # Get the standard unbatched graphs
    data_list = batch.to_data_list()

    # this should not happen below as we add the "fingerprints" in afterwards, but check:
    if "fingerprints" in data_list[0].__dict__["_store"]:
        raise ValueError("Fingerprints should not be present in the data list before this point.")

    # If batch has fingerprints, split them back to individual graphs
    if hasattr(batch, "fingerprints"):
        assert len(data_list) == batch.fingerprints.shape[0], (
            f"Number of graphs {len(data_list)} doesn't match fingerprints shape {batch.fingerprints.shape}"
        )
        for i, data in enumerate(data_list):
            data.fingerprint = batch.fingerprints[i]

    return data_list
