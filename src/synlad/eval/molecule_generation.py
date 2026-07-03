"""Adapted from https://github.com/facebookresearch/all-atom-diffusion-transformer/blob/main/src/eval/molecule_generation.py

Copyright (c) Meta Platforms, Inc. and affiliates.
This code is released under the CC-BY-NC License -- see https://github.com/facebookresearch/all-atom-diffusion-transformer for further details.
"""

import logging
import os
import random
import tempfile
from collections import defaultdict
from functools import partial
from importlib.resources import files
from typing import Any

import numpy as np
import torch
import wandb
import yaml
from openbabel import openbabel
from posebusters import PoseBusters
from pymatgen.core import Molecule
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, Draw, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from synlad.utils.constants import atom_type_index_to_element_symbol
from synlad.utils.metrics_utils import get_rocs_score
from synlad.utils.parallel import joblib_map

RDLogger.DisableLog("rdApp.*")

openbabel.obErrorLog.StopLogging()

logging.getLogger("posebusters").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# PoseBusters configuration shipped with the package; loaded once at import time.
_POSEBUSTERS_CFG_PATH = files("synlad").joinpath("data_assets/posebusters_cfg.yaml")
with _POSEBUSTERS_CFG_PATH.open("r", encoding="utf-8") as _f:
    _POSEBUSTERS_CFG = yaml.safe_load(_f)


class MoleculeGenerationEvaluator:
    """Evaluator for molecule generation tasks.

    Can be used within a Lightning module by appending sampled structures and computing metrics at
    the end of an epoch.
    """

    def __init__(self, dataset_smiles_list, removeHs=True, device="cpu"):
        self.dataset_smiles_list = dataset_smiles_list
        self.removeHs = removeHs
        self.buster = PoseBusters(config=_POSEBUSTERS_CFG)
        self.pred_arrays_list = []
        self.pred_mol_list = []
        self.pred_rdkit_list = []
        self.gt_arrays_list = []
        self.gt_mol_list = []
        self.gt_rdkit_list = []
        self.device = device

    def append_pred_array(self, pred: dict):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def append_gt_array(self, gt: dict):
        """Append a ground truth to the evaluator. Only used for conditioning."""
        self.gt_arrays_list.append(gt)

    def clear(self):
        """Clear the stored predictions, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.pred_mol_list = []
        self.pred_rdkit_list = []
        self.gt_arrays_list = []
        self.gt_mol_list = []
        self.gt_rdkit_list = []

    def _group_molecules_by_conditioning(
        self, n_samples_per_conditioning: int
    ) -> dict[int, list[Chem.Mol]]:
        """Group generated molecules by their conditioning source (pharmacophore index).

        Args:
            n_samples_per_conditioning: Number of samples generated per conditioning input.

        Returns:
            Dictionary mapping conditioning_idx -> list of RDKit molecules.
        """
        grouped_molecules = defaultdict(list)

        for idx, rdkit_mol in enumerate(self.pred_rdkit_list):
            # Calculate which conditioning this molecule belongs to
            conditioning_idx = idx // n_samples_per_conditioning
            grouped_molecules[conditioning_idx].append(rdkit_mol)

        return dict(grouped_molecules)

    def compute_tanimoto_diversity_within_group(
        self, smis: list[str], fingerprint_type: str = "RDKit"
    ) -> float:
        """Compute Tanimoto diversity within the unique molecules in a group of molecules.

        Args:
            smis: List of SMILES strings from the same conditioning.
            fingerprint_type: Type of fingerprint to use (``"ECFP4"`` or ``"RDKit"``).

        Returns:
            Average Tanimoto diversity (``1 - similarity``).
        """
        if len(smis) < 2:
            return 0.0

        unique_smis = set(smis)
        if len(unique_smis) < 2:
            return 0.0

        if fingerprint_type == "ECFP4":
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        elif fingerprint_type == "RDKit":
            fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
        else:
            raise ValueError(f"Invalid fingerprint type: {fingerprint_type}")

        fingerprints = [fpgen.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in unique_smis]

        # Mean of upper-triangular pairwise similarities via row-wise BulkTanimotoSimilarity.
        n = len(fingerprints)
        total = 0.0
        for i in range(n - 1):
            total += float(
                np.sum(DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i + 1 :]))
            )
        mean_similarity = total / (n * (n - 1) / 2)

        # Diversity = 1 - average similarity
        return 1.0 - mean_similarity

    def compute_scaffold_diversity_within_group(
        self, molecules: list[Chem.Mol]
    ) -> tuple[int, float]:
        """Compute scaffold diversity within a group of molecules.

        Args:
            molecules: List of RDKit molecules from the same conditioning.

        Returns:
            Tuple of ``(unique_scaffolds_count, scaffold_diversity_ratio)``.
        """
        if not molecules:
            return 0, 0.0

        scaffolds = set()
        valid_molecules = 0

        for mol in molecules:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is not None:
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds.add(scaffold_smiles)
                valid_molecules += 1

        unique_scaffolds = len(scaffolds)
        scaffold_diversity = unique_scaffolds / max(valid_molecules, 1)

        return unique_scaffolds, scaffold_diversity

    def compute_rocs_score_within_group(
        self, molecules: list[Chem.Mol], gt_molecule: Chem.Mol, do_rocs: bool = False
    ) -> tuple[float, float, float]:
        """Compute mean ROCS score across molecules in a group.

        Args:
            molecules: Candidate RDKit molecules from a single conditioning input.
            gt_molecule: Reference molecule to overlay against.
            do_rocs: If True, computes shape + colour + combo.

        Returns:
            Tuple of ``(mean_combo, mean_shape, mean_colour)``. When ``do_rocs`` is
            False, combo and shape are reported as ``0.0``.
        """
        if not molecules:
            return 0.0, 0.0, 0.0

        unique_mols_dict = {Chem.MolToSmiles(mol): mol for mol in molecules if mol is not None}
        unique_molecules = list(unique_mols_dict.values())

        random.shuffle(unique_molecules)
        max_molecules = min(100, len(unique_molecules))
        molecules_to_score = unique_molecules[:max_molecules]

        combo_scores, shape_scores, colour_scores = [], [], []

        for molecule in molecules_to_score:
            if do_rocs:
                combo_score, shape_score, colour_score = get_rocs_score(molecule, gt_molecule)
                combo_scores.append(combo_score)
                shape_scores.append(shape_score)
                colour_scores.append(colour_score)

        mean_combo = np.mean(combo_scores) if combo_scores else 0.0
        mean_shape = np.mean(shape_scores) if shape_scores else 0.0
        mean_colour = np.mean(colour_scores) if colour_scores else 0.0

        return mean_combo, mean_shape, mean_colour

    def compute_per_conditioning_metrics(
        self,
        n_samples_per_conditioning: int,
        fingerprint_type: str = "ECFP4",
        do_rocs: bool = False,
    ) -> dict[str, Any]:
        """Compute diversity metrics for molecules generated from each conditioning input.

        Args:
            n_samples_per_conditioning: Number of samples generated per conditioning input.
            fingerprint_type: Fingerprint to use for Tanimoto diversity (``"ECFP4"`` or ``"RDKit"``).
            do_rocs: If True, also compute mean ROCS shape + combo scores per group.

        Returns:
            Dictionary with ``"per_conditioning_metrics"`` (per-group breakdown) and
            ``"summary_stats"`` (means / stds across groups).
        """
        if not self.pred_rdkit_list:
            return {}

        # Group molecules by conditioning source
        grouped_molecules = self._group_molecules_by_conditioning(n_samples_per_conditioning)

        diversity_results = {"per_conditioning_metrics": {}, "summary_stats": {}}

        tanimoto_diversities = []
        scaffold_diversities = []
        uniqueness_ratio_per_conditioning = []
        rocs_colour_scores = []
        rocs_shape_scores = []
        rocs_combo_scores = []

        logger.info(f"3D: Computing metrics for {len(grouped_molecules)} conditioning inputs...")

        for conditioning_idx, molecules in grouped_molecules.items():
            # self.gt_rdkit_list contains the ground truth molecules for the conditioning input, with repeats for n_samples_per_conditioning
            gt_molecule = self.gt_rdkit_list[conditioning_idx * n_samples_per_conditioning]
            group_metrics = {}

            # Basic stats
            group_metrics["num_molecules"] = len(molecules)

            # Unique molecules
            unique_smiles = set()
            all_smiles = []
            for item in molecules:  # molecules are tuples of (rdkit_mol, smiles, 2d_image, valid)
                smiles = item[1]
                all_smiles.append(smiles)
                if smiles != "":  # only add valid molecules
                    unique_smiles.add(smiles)

            group_metrics["unique_molecules"] = len(unique_smiles)
            group_metrics["uniqueness_ratio"] = len(unique_smiles) / max(len(all_smiles), 1)
            uniqueness_ratio_per_conditioning.append(group_metrics["uniqueness_ratio"])

            # Tanimoto diversity
            smis = [item[1] for item in molecules if item[1] != ""]
            tanimoto_div = self.compute_tanimoto_diversity_within_group(smis, fingerprint_type)
            group_metrics["tanimoto_diversity"] = tanimoto_div
            tanimoto_diversities.append(tanimoto_div)

            # Scaffold diversity
            rd_mols = [item[0] for item in molecules if item[0] is not None]
            unique_scaffolds, scaffold_div = self.compute_scaffold_diversity_within_group(rd_mols)
            group_metrics["unique_scaffolds"] = unique_scaffolds
            group_metrics["scaffold_diversity"] = scaffold_div
            scaffold_diversities.append(scaffold_div)

            # ROCS score
            if gt_molecule is not None and do_rocs:
                rocs_div = self.compute_rocs_score_within_group(
                    rd_mols, gt_molecule, do_rocs=do_rocs
                )
                rocs_colour_score = rocs_div[2]
                group_metrics["rocs_colour_score"] = rocs_colour_score
                rocs_colour_scores.append(rocs_colour_score)
                rocs_shape_score = rocs_div[1]
                group_metrics["rocs_shape_score"] = rocs_shape_score
                rocs_shape_scores.append(rocs_shape_score)
                rocs_combo_score = rocs_div[0]
                group_metrics["rocs_combo_score"] = rocs_combo_score
                rocs_combo_scores.append(rocs_combo_score)

            diversity_results["per_conditioning_metrics"][conditioning_idx] = group_metrics

        # Compute summary statistics across all conditioning inputs
        if tanimoto_diversities:
            diversity_results["summary_stats"] = {
                "mean_tanimoto_diversity": float(np.mean(tanimoto_diversities)),
                "std_tanimoto_diversity": float(np.std(tanimoto_diversities)),
                "mean_scaffold_diversity": float(np.mean(scaffold_diversities)),
                "std_scaffold_diversity": float(np.std(scaffold_diversities)),
                "mean_uniqueness_ratio_per_conditioning": float(
                    np.mean(uniqueness_ratio_per_conditioning)
                ),
                "std_uniqueness_ratio_per_conditioning": float(
                    np.std(uniqueness_ratio_per_conditioning)
                ),
                "mean_rocs_combo_score": float(np.mean(rocs_combo_scores))
                if rocs_combo_scores
                else 0.0,
                "std_rocs_combo_score": float(np.std(rocs_combo_scores))
                if rocs_combo_scores
                else 0.0,
                "mean_rocs_shape_score": float(np.mean(rocs_shape_scores))
                if rocs_shape_scores
                else 0.0,
                "std_rocs_shape_score": float(np.std(rocs_shape_scores))
                if rocs_shape_scores
                else 0.0,
                "mean_rocs_colour_score": float(np.mean(rocs_colour_scores))
                if rocs_colour_scores
                else 0.0,
                "std_rocs_colour_score": float(np.std(rocs_colour_scores))
                if rocs_colour_scores
                else 0.0,
                "num_conditioning_inputs": len(grouped_molecules),
                "total_molecules_generated": sum(len(mols) for mols in grouped_molecules.values()),
            }

        return diversity_results

    def _arrays_to_molecules(self, save: bool = False, save_dir: str = ""):
        """Convert stored predictions and ground truths to Molecule objects for evaluation."""
        self.pred_mol_list = joblib_map(
            partial(
                array_dict_to_molecule,
                save=save,
                save_dir_name=f"{save_dir}/pred",
            ),
            self.pred_arrays_list,
            n_jobs=16,
            inner_max_num_threads=1,
            desc="    Pred to Molecule",
            total=len(self.pred_arrays_list),
        )
        if self.gt_arrays_list:
            self.gt_mol_list = joblib_map(
                partial(
                    array_dict_to_molecule,
                    save=save,
                    save_dir_name=f"{save_dir}/gt",
                ),
                self.gt_arrays_list,
                n_jobs=16,
                inner_max_num_threads=1,
                desc="    G.T. to Molecule",
                total=len(self.gt_arrays_list),
            )

    def get_metrics(
        self,
        save: bool = True,
        save_dir: str = "",
        do_pharmacophores: bool = False,
        do_rocs: bool = False,
        do_pb: bool = True,
        compute_diversity: bool = False,
        n_samples_per_conditioning: int = None,
        diversity_fingerprint_type: str = "RDKit",
    ):
        assert len(self.pred_arrays_list) > 0, "No predictions to evaluate."

        # Convert predictions to Molecule objects
        self._arrays_to_molecules(save, save_dir)

        valid_molecules = []
        valid_smiles = []
        if do_pharmacophores:
            gt_molecules = []
        for idx in range(len(self.pred_mol_list)):
            sample_idx = self.pred_mol_list[idx].properties["sample_idx"]
            pharmacophore_id = idx // n_samples_per_conditioning if do_pharmacophores else None
            if not save:
                try:
                    m = pymatgen_molecule_to_rdkit(self.pred_mol_list[idx], removeHs=self.removeHs)
                    pred_smiles = Chem.MolToSmiles(m, isomericSmiles=True)
                    pred_2d = None

                    m_frags = Chem.rdmolops.GetMolFrags(m, asMols=True)
                    largest_frag = max(m_frags, default=m, key=lambda frag: frag.GetNumAtoms())
                    pred_smiles = Chem.MolToSmiles(largest_frag, isomericSmiles=True)
                    valid = True
                    valid_molecules.append(m)
                    if do_pharmacophores:
                        m_gt = pymatgen_molecule_to_rdkit(
                            self.gt_mol_list[idx], removeHs=self.removeHs
                        )
                        gt_molecules.append(m_gt)
                    else:
                        m_gt = None
                    valid_smiles.append(pred_smiles)
                except Exception:
                    m = None
                    pred_smiles = ""
                    pred_2d = None
                    valid = False
                    if do_pharmacophores:
                        m_gt = pymatgen_molecule_to_rdkit(
                            self.gt_mol_list[idx], removeHs=self.removeHs
                        )
                    else:
                        m_gt = None
            else:
                if do_pharmacophores:
                    m_gt = Chem.MolFromPDBFile(
                        os.path.join(f"{save_dir}/gt", f"molecule_{sample_idx}.pdb"),
                        removeHs=self.removeHs,
                    )
                    gt_sdf_path = os.path.join(
                        save_dir, "gt", f"molecule_ph_{pharmacophore_id}.sdf"
                    )
                    if m_gt is not None and not os.path.exists(gt_sdf_path):
                        with Chem.SDWriter(gt_sdf_path) as w:
                            w.write(m_gt)
                    gt_molecules.append(m_gt)
                else:
                    m_gt = None

                try:
                    m = Chem.MolFromPDBFile(
                        os.path.join(f"{save_dir}/pred", f"molecule_{sample_idx}.pdb"),
                        removeHs=self.removeHs,
                    )
                    pred_smiles = Chem.MolToSmiles(m, isomericSmiles=True)
                    pred_2d = wandb.Image(Draw.MolToImage(m))

                    # simple fragment-based validity check
                    m_frags = Chem.rdmolops.GetMolFrags(m, asMols=True)
                    largest_frag = max(m_frags, default=m, key=lambda frag: frag.GetNumAtoms())
                    pred_smiles = Chem.MolToSmiles(largest_frag, isomericSmiles=True)
                    valid = True
                    valid_molecules.append(m)
                    if do_pharmacophores:
                        with Chem.SDWriter(
                            os.path.join(
                                save_dir, "pred", f"molecule_{sample_idx}_ph_{pharmacophore_id}.sdf"
                            )
                        ) as w:
                            w.write(m)
                    valid_smiles.append(pred_smiles)

                except Exception:
                    m = None
                    pred_smiles = ""
                    pred_2d = None
                    valid = False

            # Update list (used for wandb table)
            if do_pharmacophores:
                self.pred_rdkit_list.append((m, pred_smiles, pred_2d, valid, pharmacophore_id))
            else:
                self.pred_rdkit_list.append((m, pred_smiles, pred_2d, valid))
            self.gt_rdkit_list.append(m_gt)

        # Define expected PoseBusters metrics
        pb_metrics = {
            "mol_pred_loaded",
            "sanitization",
            "inchi_convertible",
            "all_atoms_connected",
            "bond_lengths",
            "bond_angles",
            "internal_steric_clash",
            "aromatic_ring_flatness",
            "double_bond_flatness",
            "internal_energy",
        }

        # Compute validity metrics
        if len(valid_smiles) > 0:
            unique_smiles = set(valid_smiles)
            novel_smiles = unique_smiles.difference(set(self.dataset_smiles_list))
            validity_metrics_dict = {
                "valid_rate": torch.tensor(
                    len(valid_smiles) / len(self.pred_rdkit_list), device=self.device
                ),
                "unique_rate": torch.tensor(
                    len(unique_smiles) / len(valid_smiles), device=self.device
                ),
                "novel_rate": torch.tensor(
                    len(novel_smiles) / len(valid_smiles), device=self.device
                ),
            }

            # Get PoseBusters metrics and filter to only expected ones
            pb_metrics_dict = {metric: 0.0 for metric in pb_metrics}
            posebusters_sum = 0
            if do_pb:
                all_pb_metrics = self.buster.bust(valid_molecules, None, None)
                all_pb_metrics_dict = all_pb_metrics.mean().to_dict()
                pb_metrics_dict.update(
                    {k: v for k, v in all_pb_metrics_dict.items() if k in pb_metrics}
                )
                for _, row in all_pb_metrics.iterrows():
                    posebusters_sum += 0 if row.isin([False]).any() else 1

            pb_metrics_dict["posebusters_sum"] = torch.tensor(
                posebusters_sum / len(valid_molecules), device=self.device
            )

            if do_pharmacophores:
                rocs_scores = []
                # Limit ROCS computation to max 100 molecules for faster validation
                max_rocs_molecules = min(100, len(valid_molecules))
                molecules_to_score = list(
                    zip(
                        valid_molecules[:max_rocs_molecules],
                        gt_molecules[:max_rocs_molecules],
                        strict=False,
                    )
                )

                if do_rocs:
                    for pred_mol, gt_mol in molecules_to_score:
                        if (
                            gt_mol is not None
                        ):  # due to the way we construct the rdkit mol. pred_mol cannot be None
                            rocs_scores.append(get_rocs_score(pred_mol, gt_mol))
                    rocs_scores = np.array(rocs_scores)

                if do_rocs:
                    # Full ROCS scores: [combo, shape, colour]
                    rocs_metrics_dict = {
                        "rocs_combo_score": torch.tensor(
                            rocs_scores[:, 0].mean(), device=self.device
                        ),
                        "rocs_shape_score": torch.tensor(
                            rocs_scores[:, 1].mean(), device=self.device
                        ),
                        "rocs_colour_score": torch.tensor(
                            rocs_scores[:, 2].mean(), device=self.device
                        ),
                    }
                else:
                    rocs_metrics_dict = {
                        "rocs_combo_score": torch.tensor(0.0, device=self.device),
                        "rocs_shape_score": torch.tensor(0.0, device=self.device),
                        "rocs_colour_score": torch.tensor(0.0, device=self.device),
                    }
        else:
            validity_metrics_dict = {
                "valid_rate": torch.tensor(0.0, device=self.device),
                "unique_rate": torch.tensor(0.0, device=self.device),
                "novel_rate": torch.tensor(0.0, device=self.device),
            }
            pb_metrics_dict = {metric: 0.0 for metric in pb_metrics}
            pb_metrics_dict["posebusters_sum"] = 0.0
            if do_pharmacophores:
                rocs_metrics_dict = {
                    "rocs_combo_score": torch.tensor(0.0, device=self.device),
                    "rocs_shape_score": torch.tensor(0.0, device=self.device),
                    "rocs_colour_score": torch.tensor(0.0, device=self.device),
                }
        # Compute per-conditioning diversity metrics if enabled
        diversity_metrics_dict = {}
        if compute_diversity and n_samples_per_conditioning is not None:
            logger.info("Computing per-conditioning diversity metrics...")
            diversity_results = self.compute_per_conditioning_metrics(
                n_samples_per_conditioning, diversity_fingerprint_type, do_rocs
            )

            if diversity_results and "summary_stats" in diversity_results:
                summary_stats = diversity_results["summary_stats"]

                # Add diversity metrics to the metrics dictionary
                diversity_metrics_dict = {
                    "diversity_mean_tanimoto": torch.tensor(
                        summary_stats.get("mean_tanimoto_diversity", 0.0), device=self.device
                    ),
                    "diversity_mean_scaffold": torch.tensor(
                        summary_stats.get("mean_scaffold_diversity", 0.0), device=self.device
                    ),
                    "diversity_mean_uniqueness_ratio_per_conditioning": torch.tensor(
                        summary_stats.get("mean_uniqueness_ratio_per_conditioning", 0.0),
                        device=self.device,
                    ),
                    "diversity_mean_rocs_combo": torch.tensor(
                        summary_stats.get("mean_rocs_combo_score", 0.0), device=self.device
                    ),
                    "diversity_mean_rocs_shape": torch.tensor(
                        summary_stats.get("mean_rocs_shape_score", 0.0), device=self.device
                    ),
                    "diversity_mean_rocs_colour": torch.tensor(
                        summary_stats.get("mean_rocs_colour_score", 0.0), device=self.device
                    ),
                    "diversity_num_conditioning_inputs": torch.tensor(
                        summary_stats.get("num_conditioning_inputs", 0), device=self.device
                    ),
                }

        if do_pharmacophores:
            metrics_dict = {
                **validity_metrics_dict,
                **pb_metrics_dict,
                **rocs_metrics_dict,
                **diversity_metrics_dict,
            }
        else:
            metrics_dict = {**validity_metrics_dict, **pb_metrics_dict, **diversity_metrics_dict}
        return metrics_dict

    def get_wandb_table(self, current_epoch: int = 0, save_dir: str = ""):
        # Log molecule structures and metrics to wandb
        pred_table = wandb.Table(
            columns=[
                "Global step",
                "Sample idx",
                "Num atoms",
                "Valid?",
                "Pred atom types",
                "Pred Smiles",
                "Pred 2D",
                "Pred 3D",
            ]
        )

        for idx in range(len(self.pred_mol_list)):
            sample_idx = self.pred_mol_list[idx].properties["sample_idx"]

            num_atoms = len(self.pred_mol_list[idx].atomic_numbers)

            pred_atom_types = " ".join(
                [str(int(t)) for t in self.pred_mol_list[idx].atomic_numbers]
            )

            pred_smiles = self.pred_rdkit_list[idx][1]

            pred_2d = self.pred_rdkit_list[idx][2]

            valid = self.pred_rdkit_list[idx][3]

            pred_3d = wandb.Molecule(os.path.join(save_dir, "pred", f"molecule_{sample_idx}.pdb"))

            # Update table
            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                valid,
                pred_atom_types,
                pred_smiles,
                pred_2d,
                pred_3d,
            )

        return pred_table


def array_dict_to_molecule(
    x: dict[str, np.ndarray],
    save: bool = False,
    save_dir_name: str = "",
) -> Molecule:
    """Convert a dictionary of numpy arrays to a pymatgen ``Molecule``.

    Args:
        x: Dictionary of numpy arrays with keys ``"atom_types"`` (atomic numbers),
            ``"pos"`` (3D coordinates), and ``"sample_idx"`` (sample index).
        save: Whether to save the molecule as a PDB file.
        save_dir_name: Directory to save the PDB file (used only when ``save`` is True).

    Returns:
        Pymatgen ``Molecule``, optionally also written to disk as a PDB file.
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


def pymatgen_molecule_to_rdkit(mol: Molecule, removeHs: bool = True):
    """Convert a pymatgen Molecule to an RDKit Mol.

    This uses a temporary PDB file to leverage RDKit's bond perception from 3D coordinates.

    Args:
        mol: Pymatgen Molecule instance.
        removeHs: Whether to remove hydrogens when loading into RDKit.

    Returns:
        rdkit.Chem.Mol or None if conversion fails.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
            tmp_path = tf.name
        mol.to(tmp_path, fmt="pdb")
        rd_mol = Chem.MolFromPDBFile(tmp_path, removeHs=removeHs)
        if rd_mol is not None:
            try:
                Chem.SanitizeMol(rd_mol, catchErrors=True)
            except Exception:
                pass
        return rd_mol
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
