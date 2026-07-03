"""Adapted from https://github.com/facebookresearch/all-atom-diffusion-transformer/blob/main/src/eval/molecule_reconstruction.py

Copyright (c) Meta Platforms, Inc. and affiliates.
This code is released under the CC-BY-NC License -- see https://github.com/facebookresearch/all-atom-diffusion-transformer for further details.
"""

import logging
import os
import tempfile

import numpy as np
import torch
import wandb
from openbabel import openbabel
from pymatgen.analysis.molecule_matcher import MoleculeMatcher
from pymatgen.core import Molecule
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from tqdm import tqdm

from synlad.utils.constants import atom_type_index_to_element_symbol

logger = logging.getLogger(__name__)

RDLogger.DisableLog("rdApp.*")

openbabel.obErrorLog.StopLogging()

# 2D structure rendering keeps explicit hydrogens because Draw.MolToImage uses
# the atoms it sees; SMILES generation strips them because hydrogens are
# implicit in canonical SMILES. Different code paths, different requirements.
_RENDER_REMOVE_HS = False


class MoleculeReconstructionEvaluator:
    """Evaluator for molecule reconstruction tasks. Can be used within a Lightning module,
    appending predictions and ground truths during training and computing metrics at the end of an
    epoch, or can be used as a standalone object to evaluate predictions on a dataset.

    Args:
        tolerance: MoleculeMatcher tolerance for whether two molecules are the same.
        removeHs: Whether to drop hydrogens when re-loading saved PDB files into RDKit
            for SMILES generation.

    Attributes:
        pred_arrays_list: Predictions appended via :meth:`append_pred_array`.
        gt_arrays_list: Ground truths appended via :meth:`append_gt_array`.
        pred_mol_list: Pymatgen ``Molecule`` predictions, populated by :meth:`get_metrics`.
        gt_mol_list: Pymatgen ``Molecule`` ground truths, populated by :meth:`get_metrics`.
        pred_smiles_list: SMILES strings of the predictions, populated by :meth:`get_metrics`.
        pred_rdkit_list: RDKit ``Mol`` objects of the predictions (``None`` for ones that
            failed to parse), populated by :meth:`get_metrics`.
    """

    MAX_TABLE_EXAMPLES = 5

    def __init__(self, tolerance: float = 0.01, removeHs: bool = True) -> None:
        self.matcher = MoleculeMatcher(tolerance=tolerance)
        self.pred_arrays_list: list[dict[str, np.ndarray]] = []
        self.gt_arrays_list: list[dict[str, np.ndarray]] = []
        self.pred_mol_list: list[Molecule] = []
        self.gt_mol_list: list[Molecule] = []
        self.pred_smiles_list: list[str] = []
        self.pred_rdkit_list: list[Chem.Mol | None] = []
        self.removeHs = removeHs

    def append_pred_array(self, pred: dict[str, np.ndarray]) -> None:
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def append_gt_array(self, gt: dict[str, np.ndarray]) -> None:
        """Append a ground truth to the evaluator."""
        self.gt_arrays_list.append(gt)

    def clear(self) -> None:
        """Clear the stored predictions and ground truths, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.gt_arrays_list = []
        self.pred_mol_list = []
        self.gt_mol_list = []
        self.pred_smiles_list = []
        self.pred_rdkit_list = []

    def _arrays_to_molecules(self, save: bool = False, save_dir: str = "") -> None:
        """Convert stored predictions and ground truths to ``Molecule`` objects."""
        self.pred_mol_list = []
        for array_dict in tqdm(self.pred_arrays_list, desc="    Pred to Molecule"):
            self.pred_mol_list.append(
                array_dict_to_molecule(
                    array_dict, save=save, save_dir_name=os.path.join(save_dir, "pred")
                )
            )

        self.gt_mol_list = []
        for array_dict in tqdm(self.gt_arrays_list, desc="    G.T. to Molecule"):
            self.gt_mol_list.append(
                array_dict_to_molecule(
                    array_dict, save=save, save_dir_name=os.path.join(save_dir, "gt")
                )
            )

        self._generate_smiles_from_molecules(save=save, save_dir=save_dir)

    def _generate_smiles_from_molecules(self, save: bool, save_dir: str) -> None:
        """Populate ``pred_smiles_list`` and ``pred_rdkit_list`` from the predicted molecules.

        When ``save`` is True we re-load the PDB files already written under
        ``save_dir/pred`` to take advantage of RDKit's bond perception. When ``save``
        is False we round-trip each molecule through a temporary PDB file so that
        downstream consumers (e.g. the synthesis evaluator) always get SMILES,
        regardless of whether visualizations are being saved this epoch.
        """
        self.pred_smiles_list = []
        self.pred_rdkit_list = []

        for mol in self.pred_mol_list:
            sample_idx = mol.properties["sample_idx"]
            smiles, rd_mol = "", None
            try:
                if save:
                    pdb_path = os.path.join(save_dir, "pred", f"molecule_{sample_idx}.pdb")
                    rd_mol = Chem.MolFromPDBFile(pdb_path, removeHs=self.removeHs)
                else:
                    rd_mol = _pymatgen_to_rdkit_via_tempfile(mol, removeHs=self.removeHs)
                if rd_mol is not None:
                    smiles = Chem.MolToSmiles(rd_mol)
            except Exception as exc:
                # RDKit / pymatgen / OpenBabel can raise a wide variety of exception
                # types from C++ bindings; treat any failure as "no SMILES for this
                # sample" rather than crashing the whole evaluation.
                logger.warning("SMILES generation failed for sample %s: %s", sample_idx, exc)
                rd_mol = None
                smiles = ""
            self.pred_smiles_list.append(smiles)
            self.pred_rdkit_list.append(rd_mol)

    def _get_metrics(self, pred: Molecule, gt: Molecule) -> float:
        try:
            rms_dist = self.matcher.get_rmsd(pred, gt)
            return float("inf") if rms_dist == np.inf else rms_dist
        except Exception as exc:
            # MoleculeMatcher delegates to OpenBabel C++ which can raise a wide
            # variety of exception types; treat any failure as "no match".
            logger.debug("RMSD computation failed: %s", exc)
            return float("inf")

    def get_metrics(
        self,
        current_epoch: int = 0,
        save: bool = False,
        save_dir: str = "",
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the match rate and avg. RMS distance between predictions and ground truths.

        Note: ``self.rms_dists`` can be used to access RMSD per sample but is not returned.

        Args:
            current_epoch: Current training epoch (used for the progress-bar label).
            save: Whether to save predicted and ground-truth molecules as PDB files.
            save_dir: Directory under which ``pred/`` and ``gt/`` subdirectories will be created.
            device: Device for the returned tensors. Defaults to CUDA if available, else CPU.

        Returns:
            Dictionary of metrics, including ``match_rate`` and ``rms_dist``.
        """
        if len(self.pred_arrays_list) != len(self.gt_arrays_list):
            raise ValueError(
                "Number of predictions and ground truths must match "
                f"(got {len(self.pred_arrays_list)} and {len(self.gt_arrays_list)})."
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._arrays_to_molecules(save, save_dir)

        rms_dists: list[float] = []
        for i in tqdm(
            range(len(self.pred_mol_list)), desc=f"Epoch {current_epoch}, reconstruction eval"
        ):
            rms_dists.append(self._get_metrics(self.pred_mol_list[i], self.gt_mol_list[i]))
        self.rms_dists = torch.tensor(rms_dists, device=device)
        match_rate = (~torch.isinf(self.rms_dists)).long()
        if match_rate.sum() == 0:
            # No valid predictions --> return large RMSD for logging purposes
            return {
                "match_rate": match_rate,
                "rms_dist": torch.tensor([10.0] * len(match_rate), device=device),
            }
        return {
            "match_rate": match_rate,
            "rms_dist": self.rms_dists[~torch.isinf(self.rms_dists)],
        }

    def _check_sample_idx_alignment(self, idx: int) -> int:
        """Return ``sample_idx`` at position ``idx``, raising if pred and gt disagree."""
        sample_idx = self.gt_arrays_list[idx]["sample_idx"]
        if sample_idx != self.pred_arrays_list[idx]["sample_idx"]:
            raise ValueError(
                f"Prediction and ground-truth sample_idx mismatch at position {idx}: "
                f"pred={self.pred_arrays_list[idx]['sample_idx']}, gt={sample_idx}."
            )
        return sample_idx

    def _render_2d(self, pdb_path: str) -> wandb.Image | None:
        """Render a PDB file as a 2D RDKit image for wandb, returning None on failure."""
        try:
            mol = Chem.MolFromPDBFile(pdb_path, removeHs=_RENDER_REMOVE_HS)
            if mol is None:
                return None
            return wandb.Image(Draw.MolToImage(mol))
        except Exception as exc:
            # RDKit can raise a wide variety of exception types from its C++
            # bindings; visualization failures must never crash evaluation.
            logger.debug("2D render failed for %s: %s", pdb_path, exc)
            return None

    def get_wandb_table(
        self,
        current_epoch: int = 0,
        save_dir: str = "",
    ) -> wandb.Table:
        """Create a wandb.Table object with the results of the evaluation.

        Args:
            current_epoch: Current training epoch.
            save_dir: Directory where molecule files are saved.
        """
        pred_table = wandb.Table(
            columns=[
                "Epoch",
                "Sample idx",
                "Num atoms",
                "RMSD",
                "Match?",
                "True atom types",
                "Pred atom types",
                "True 2D",
                "Pred 2D",
                "True 3D",
                "Pred 3D",
            ]
        )

        count = 0
        for idx in range(len(self.pred_mol_list)):
            if count >= self.MAX_TABLE_EXAMPLES:
                break

            sample_idx = self._check_sample_idx_alignment(idx)
            num_atoms = len(self.gt_mol_list[idx].atomic_numbers)
            rmsd = self.rms_dists[idx]
            match = rmsd != float("inf")
            true_atom_types = " ".join(str(int(t)) for t in self.gt_mol_list[idx].atomic_numbers)
            pred_atom_types = " ".join(str(int(t)) for t in self.pred_mol_list[idx].atomic_numbers)

            gt_pdb = os.path.join(save_dir, "gt", f"molecule_{sample_idx}.pdb")
            pred_pdb = os.path.join(save_dir, "pred", f"molecule_{sample_idx}.pdb")
            true_2d = self._render_2d(gt_pdb)
            pred_2d = self._render_2d(pred_pdb)
            true_3d = wandb.Molecule(gt_pdb)
            pred_3d = wandb.Molecule(pred_pdb)

            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                rmsd,
                match,
                true_atom_types,
                pred_atom_types,
                true_2d,
                pred_2d,
                true_3d,
                pred_3d,
            )
            count += 1
        return pred_table

    def get_combined_wandb_table(
        self,
        synthesis_examples: list[dict],
        current_epoch: int = 0,
        save_dir: str = "",
        max_examples: int = 5,
    ) -> wandb.Table:
        """Create a combined wandb.Table object with both molecular reconstruction and synthesis data.

        Args:
            synthesis_examples: List of synthesis examples from SynthesisEvaluator.
            current_epoch: Current training epoch.
            save_dir: Directory where molecule files are saved.
            max_examples: Maximum number of examples to include.
        """
        synthesis_by_sample_idx = {example["sample_idx"]: example for example in synthesis_examples}

        pred_table = wandb.Table(
            columns=[
                "Epoch",
                "Sample idx",
                "Num atoms",
                "RMSD",
                "Match?",
                "True atom types",
                "Pred atom types",
                "True 2D",
                "Pred 2D",
                "True 3D",
                "Pred 3D",
                "Synthesis task",
                "Synthesis target",
                "Synthesis prediction",
                "Synthesis losses",
                "Synthesis probs_target",
                "Synthesis probs_prediction",
            ]
        )

        count = 0
        for idx in range(len(self.pred_mol_list)):
            if count >= max_examples:
                break

            sample_idx = self._check_sample_idx_alignment(idx)
            if sample_idx not in synthesis_by_sample_idx:
                continue

            num_atoms = len(self.gt_mol_list[idx].atomic_numbers)
            rmsd = self.rms_dists[idx]
            match = rmsd != float("inf")
            true_atom_types = " ".join(str(int(t)) for t in self.gt_mol_list[idx].atomic_numbers)
            pred_atom_types = " ".join(str(int(t)) for t in self.pred_mol_list[idx].atomic_numbers)

            gt_pdb = os.path.join(save_dir, "gt", f"molecule_{sample_idx}.pdb")
            pred_pdb = os.path.join(save_dir, "pred", f"molecule_{sample_idx}.pdb")
            true_2d = self._render_2d(gt_pdb)
            pred_2d = self._render_2d(pred_pdb)
            true_3d = wandb.Molecule(gt_pdb)
            pred_3d = wandb.Molecule(pred_pdb)

            synthesis_example = synthesis_by_sample_idx[sample_idx]
            synthesis_target_str = " | ".join(map(str, synthesis_example["target"]))
            synthesis_prediction_str = " | ".join(map(str, synthesis_example["prediction"]))
            synthesis_losses_str = " | ".join(
                f"{x:.4f}" if not np.isnan(x) else "N/A" for x in synthesis_example["losses"]
            )
            synthesis_probs_target_str = " | ".join(
                f"{x:.4f}" for x in synthesis_example["probs_target"]
            )
            synthesis_probs_prediction_str = " | ".join(
                f"{x:.4f}" for x in synthesis_example["probs_prediction"]
            )

            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                rmsd,
                match,
                true_atom_types,
                pred_atom_types,
                true_2d,
                pred_2d,
                true_3d,
                pred_3d,
                synthesis_example["task"],
                synthesis_target_str,
                synthesis_prediction_str,
                synthesis_losses_str,
                synthesis_probs_target_str,
                synthesis_probs_prediction_str,
            )
            count += 1

        return pred_table


def array_dict_to_molecule(
    x: dict[str, np.ndarray],
    save: bool = False,
    save_dir_name: str = "",
) -> Molecule:
    """Convert a dictionary of numpy arrays to a ``Molecule`` compatible with ``MoleculeMatcher``.

    Args:
        x: Dictionary of numpy arrays with keys:
            - ``atom_types``: Atom type indices (0-8).
            - ``pos``: 3D coordinates of atoms.
            - ``sample_idx``: Index of the sample in the dataset.
        save: Whether to save the molecule as a PDB file.
        save_dir_name: Directory to save the PDB file (used only when ``save`` is True).

    Returns:
        Pymatgen ``Molecule``, optionally also written to disk.
    """
    atom_type_indices = x["atom_types"].astype(int).tolist()
    element_symbols = [atom_type_index_to_element_symbol(idx) for idx in atom_type_indices]

    mol = Molecule(
        species=element_symbols, coords=x["pos"], properties={"sample_idx": x["sample_idx"]}
    )
    if save:
        os.makedirs(save_dir_name, exist_ok=True)
        mol.to(os.path.join(save_dir_name, f"molecule_{x['sample_idx']}.pdb"), fmt="pdb")
    return mol


def _pymatgen_to_rdkit_via_tempfile(mol: Molecule, removeHs: bool) -> Chem.Mol | None:
    """Round-trip a pymatgen ``Molecule`` through a temporary PDB to get an RDKit ``Mol``.

    RDKit's bond perception works from 3D coordinates via PDB; this lets us obtain a
    SMILES string for predictions even when we don't want to persist PDB files.
    """
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
            tmp_path = tf.name
        mol.to(tmp_path, fmt="pdb")
        return Chem.MolFromPDBFile(tmp_path, removeHs=removeHs)
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)
