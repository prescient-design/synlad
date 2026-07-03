"""Combined collator for molecular and pathway data."""

from collections.abc import Callable
from typing import Any

import numpy.random as np_random
import torch
import torch_geometric.data as pyg_data

from synlad.data.components.synthesis_dataset import RxnNetTaskCollateFunc
from synlad.tokenization.synthesis_vocab import TokenLibrary


class PathwayCollateFunction:
    """Collate function for combined molecular and pathway data."""

    def __init__(
        self,
        rng: np_random.RandomState,
        prompt_to_action_probabilizer: Callable[[list[str]], float],
        bb_graphs: list[pyg_data.Data],
        bb_tkn_library: TokenLibrary,
        nested_tensors: bool = True,
        max_seq_len: int = 500,
        enable_conditioning: bool = True,
        prefix_free_generation: bool = True,
    ):
        """
        Initialize the combined collate function.

        Args:
            rng: Random state for reproducible sampling
            prompt_to_action_probabilizer: Function to convert prompts to action probabilities
            bb_graphs: List of building block graphs
            bb_tkn_library: Token library for building blocks
            nested_tensors: Whether to use nested tensors
            max_seq_len: Maximum sequence length
            enable_conditioning: Whether to enable conditioning (must be True for current implementation)
            prefix_free_generation: Whether to use prefix-free generation (must be True for current implementation)
        """
        # Validate required parameters
        if not enable_conditioning:
            raise ValueError("enable_conditioning must be True for current implementation")
        if not prefix_free_generation:
            raise ValueError("prefix_free_generation must be True for current implementation")

        if not isinstance(bb_graphs, list):
            raise ValueError("bb_graphs must be a list")

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.bb_tkn_library = bb_tkn_library

        self.pathway_collate_func = RxnNetTaskCollateFunc(
            rng=rng,
            prompt_to_action_probabilizer=prompt_to_action_probabilizer,
            bb_graphs=bb_graphs,
            bb_tkn_library=bb_tkn_library,
            nested_tensors=nested_tensors,
            max_seq_len=max_seq_len,
            enable_conditioning=enable_conditioning,
            prefix_free_generation=prefix_free_generation,
        )

    def __call__(self, batch_items: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a batch of combined molecular and pathway data.

        Args:
            batch_items: List of dictionaries containing 'molecule', 'pathway', and 'mol_idx' keys

        Returns:
            Dictionary containing:
            - molecules: PyTorch Geometric batch of molecular data
            - pathways: RxnNetTaskBatch of pathway data (or None if all pathways are None)
            - pathway_indices: Tensor mapping batch indices to pathway batch indices (-1 for missing)
            - mol_indices: Tensor of original molecule indices
            - batch_size: Number of items in batch
        """
        # Validate input
        if not batch_items:
            raise ValueError("batch_items cannot be empty")

        # Separate molecules and pathways
        molecules = []
        pathways = []
        pharmacophores = []
        mol_indices = []
        pathway_indices = []

        for i, item in enumerate(batch_items):
            if not isinstance(item, dict) or "molecule" not in item:
                raise ValueError(f"Invalid item at index {i}: missing 'molecule' key")

            molecules.append(item["molecule"])
            mol_indices.append(item.get("mol_idx", i))
            pharmacophore = item.get("pharmacophore", None)
            if pharmacophore is not None:
                pharmacophores.append(pharmacophore)
            pathway = item.get("pathway")
            pathway_indices.append(len(pathways))  # Index in pathway batch
            pathways.append(pathway)

        # Collate molecular data using PyTorch Geometric
        mol_batch = pyg_data.Batch.from_data_list(molecules)
        if len(pharmacophores) > 0:
            ph4_batch = pyg_data.Batch.from_data_list(pharmacophores)
        else:
            ph4_batch = None

        # Collate pathway data
        pathway_batch = self.pathway_collate_func(pathways)

        return {
            "molecules": mol_batch,
            "pharmacophores": ph4_batch,
            "pathways": pathway_batch,
            "pathway_indices": torch.tensor(pathway_indices, dtype=torch.long),
            "mol_indices": torch.tensor(mol_indices, dtype=torch.long),
            "batch_size": len(batch_items),
        }
