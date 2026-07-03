import os
import warnings
from functools import lru_cache, partial
from typing import Any

import numpy as np
import torch
import wandb
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

# Suppress RDKit warnings
from rdkit.rdBase import DisableLog

from synlad.utils.parallel import joblib_map

DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)

from synlad.utils.metrics_utils import rdmol_to_oemol, score_smiles_to_query  # noqa: E402


@lru_cache(maxsize=4096)
def calculate_tanimoto_similarity(smi1: str, smi2: str) -> float | None:
    """Calculate Tanimoto similarity between two SMILES strings using RDKit topological fingerprints."""
    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)

        if mol1 is None or mol2 is None:
            return None

        fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator()
        fp1 = fp_gen.GetFingerprint(mol1)
        fp2 = fp_gen.GetFingerprint(mol2)

        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return None


def _extract_product_molecules_from_sequence(sequence: list[str]) -> list[str]:
    """
    Extract product molecules from a synthesis sequence.

    Args:
        sequence: List of tokens representing a synthesis pathway

    Returns:
        List of SMILES strings representing product molecules
    """
    products = []

    # Look for molecules that appear after reaction tokens
    # This is a simplified extraction - may need refinement based on sequence format
    for i, token in enumerate(sequence):
        if token in ["<FWD_RXN>", "<REV_RXN>"] and i + 1 < len(sequence):
            # Next token after reaction might be product
            next_token = sequence[i + 1]
            # Check if it's a SMILES string (basic validation)
            if _is_valid_smiles(next_token):
                products.append(next_token)
        elif token not in [
            "<BB>",
            "<FWD_RXN>",
            "<REV_RXN>",
            "<EOS>",
            "<PAD>",
            "<FWD_RXN_RUN>",
            "<REV_RXN_RUN>",
        ] and _is_valid_smiles(token):
            # Direct molecule tokens
            products.append(token)

    return products


def _is_valid_smiles(smiles: str) -> bool:
    """Check if a string is a valid SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


class SynthesisGenerationEvaluator:
    """
    Evaluator for synthesis pathway generation tasks.
    Compares synthesis decoder outputs with molecule decoder outputs.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the synthesis generation evaluator.

        Args:
            device: Device for tensor operations
        """
        self.device = device

        # Storage for raw predictions (following MoleculeGenerationEvaluator pattern)
        self.pred_data_list = []  # List of prediction dictionaries
        self.processed_results = []  # Processed results for wandb table

    def append_pred_data(self, pred_data: dict[str, Any]):
        """
        Append prediction data to the evaluator.

        Args:
            pred_data: Dictionary containing:
                - synthesis_predictions: List of synthesis decoder outputs
                - target_molecule: Target molecule from molecule decoder
                - sample_idx: Sample index
        """
        self.pred_data_list.append(pred_data)

    def clear(self):
        """Clear the stored predictions, to be used at the end of an epoch."""
        self.pred_data_list = []
        self.processed_results = []

    def update_target_molecules(self, pred_rdkit_list):
        """
        Update target molecules from molecule evaluator's processed results.

        Args:
            pred_rdkit_list: list of rdkit molecules predicted by molecule decoder
        """

        # Extract SMILES from molecule evaluator's results
        # pred_rdkit_list contains tuples of (m, pred_smiles, pred_2d, valid)
        target_smiles = []

        # figure out whether do_pharmacophores is True
        if len(pred_rdkit_list[0]) == 5:
            do_pharmacophores = True
            target_pharmacophore_ids = []
        else:
            do_pharmacophores = False

        for rdkit_result in pred_rdkit_list:
            pred_smiles = rdkit_result[1] if len(rdkit_result) > 1 else ""
            valid = rdkit_result[3] if len(rdkit_result) > 3 else False

            # Only use valid SMILES as targets
            if valid and pred_smiles:
                target_smiles.append(pred_smiles)
            else:
                target_smiles.append(None)

            if do_pharmacophores:
                pharmacophore_id = rdkit_result[4]
                target_pharmacophore_ids.append(pharmacophore_id)

        # Update stored prediction data with target molecules
        for i, pred_data in enumerate(self.pred_data_list):
            if i < len(target_smiles):
                pred_data["target_molecule"] = target_smiles[i]
                if do_pharmacophores:
                    pred_data["target_pharmacophore_id"] = target_pharmacophore_ids[i]
                else:
                    pred_data["target_pharmacophore_id"] = None

    def _process_predictions(self, save_dir: str = "", compute_rocs_for_syn: bool = False):
        """Process all stored predictions to compute metrics (similar to _arrays_to_molecules)."""
        self.processed_results = joblib_map(
            partial(
                process_synthesis_prediction,
                save_dir=save_dir,
                compute_rocs_for_syn=compute_rocs_for_syn,
            ),
            self.pred_data_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc="    Processing Synthesis",
            total=len(self.pred_data_list),
        )

    def get_metrics(
        self, save_dir: str = "", compute_rocs_for_syn: bool = False
    ) -> dict[str, float]:
        """
        Compute final metrics from accumulated data.

        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        assert len(self.pred_data_list) > 0, "No predictions to evaluate."

        # Process all predictions
        self._process_predictions(save_dir=save_dir, compute_rocs_for_syn=compute_rocs_for_syn)

        # Initialize counters
        total_samples = len(self.processed_results)
        valid_syntheses = 0
        top1_matches = 0
        overall_matches = 0
        total_score = 0.0
        score_counts = 0
        max_similarities_to_target = 0.0
        similarity_counts = 0
        total_combo_score = 0.0
        combo_counts = 0
        total_shape_score = 0.0
        shape_counts = 0
        total_colour_score = 0.0
        colour_counts = 0

        # Aggregate metrics from processed results
        for result in self.processed_results:
            if result["valid_syntheses"] > 0:
                valid_syntheses += 1

            if result["top1_match"]:
                top1_matches += 1

            if result["any_match"]:
                overall_matches += 1

            if result["avg_score"] is not None:
                total_score += result["avg_score"]
                score_counts += 1

            if result["max_similarity_to_target"] is not None:
                max_similarities_to_target += result["max_similarity_to_target"]
                similarity_counts += 1

            if result["max_combo_score"] is not None:
                total_combo_score += result["max_combo_score"]
                combo_counts += 1
            if result["max_shape_score"] is not None:
                total_shape_score += result["max_shape_score"]
                shape_counts += 1
            if result["max_colour_score"] is not None:
                total_colour_score += result["max_colour_score"]
                colour_counts += 1

        # Compute final metrics
        metrics = {
            "synthesis_valid_rate": torch.tensor(
                valid_syntheses / total_samples, device=self.device
            ),
            "top1_match_rate": torch.tensor(top1_matches / total_samples, device=self.device),
            "synthesis_molecule_match_rate": torch.tensor(
                overall_matches / total_samples, device=self.device
            ),
            "avg_synthesis_score": torch.tensor(
                total_score / max(score_counts, 1), device=self.device
            ),
            "avg_max_similarity_to_target": torch.tensor(
                max_similarities_to_target / max(similarity_counts, 1), device=self.device
            ),
            "avg_max_combo_score": torch.tensor(
                total_combo_score / max(combo_counts, 1), device=self.device
            ),
            "avg_max_shape_score": torch.tensor(
                total_shape_score / max(shape_counts, 1), device=self.device
            ),
            "avg_max_colour_score": torch.tensor(
                total_colour_score / max(colour_counts, 1), device=self.device
            ),
            "total_samples": total_samples,
        }

        return metrics

    def get_wandb_table(self, current_epoch: int = 0):
        """
        Create wandb table for synthesis predictions.

        Args:
            current_epoch: Current training epoch

        Returns:
            wandb.Table: Table with synthesis predictions and metrics
        """
        # Create table with synthesis-specific columns
        pred_table = wandb.Table(
            columns=[
                "Global step",
                "Sample idx",
                "Target molecule",
                "Top synthesis sequences",
                "Avg score",
                "Valid syntheses",
                "Top1 match",
                "Any match",
            ]
        )

        # Add data to table
        for _idx, result in enumerate(self.processed_results[:20]):
            sample_idx = result["sample_idx"]
            target_molecule = result["target_molecule"] or "N/A"

            # Format top synthesis sequences
            top_sequences = []
            for k, seq in enumerate(result["sequences"]):
                if seq:
                    top_sequences.append(f"Rank {k + 1}: {seq}")

            sequences_str = "\n".join(top_sequences) if top_sequences else "No valid sequences"
            avg_score_str = (
                f"{result['avg_score']:.3f}" if result["avg_score"] is not None else "N/A"
            )

            pred_table.add_data(
                current_epoch,
                sample_idx,
                target_molecule,
                sequences_str,
                avg_score_str,
                result["valid_syntheses"],
                result["top1_match"],
                result["any_match"],
            )

        return pred_table


def process_synthesis_prediction(
    pred_data: dict[str, Any], save_dir: str = "", compute_rocs_for_syn: bool = False
) -> dict[str, Any]:
    """
    Process a single synthesis prediction (parallel processing function).

    Args:
        pred_data: Dictionary containing synthesis predictions and target molecule

    Returns:
        Dict with processed results for metrics computation
    """
    synthesis_predictions = pred_data.get("synthesis_predictions", [])
    target_molecule = pred_data.get("target_molecule")
    target_pharmacophore_id = pred_data.get("target_pharmacophore_id", None)
    sample_idx = pred_data.get("sample_idx", 0)

    # Initialize result structure
    result = {
        "sample_idx": sample_idx,
        "target_molecule": target_molecule,
        "target_pharmacophore_id": target_pharmacophore_id,
        "sequences": [],
        "valid_syntheses": 0,
        "top1_match": False,
        "any_match": False,
        "avg_score": None,
        "max_combo_score": None,
        "max_shape_score": None,
        "max_colour_score": None,
        "max_similarity_to_target": None,
    }

    matches_found = []
    valid_scores = []
    similarity_scores = []
    final_products = []

    # Process each synthesis prediction
    for _k, pred_dict in enumerate(synthesis_predictions):
        sequence = pred_dict.get("sequence", [])
        score = pred_dict.get("score", 0.0)

        # Store sequence and score
        sequence_str = " → ".join(sequence) if sequence else ""
        result["sequences"].append(sequence_str)
        valid_scores.append(score)

        # Extract product molecules from synthesis sequence
        product_molecules = _extract_product_molecules_from_sequence(sequence)

        if product_molecules:
            result["valid_syntheses"] += 1

            # Check if any product matches the target molecule
            match_found = False
            if target_molecule:
                for product_mol in product_molecules:
                    similarity = calculate_tanimoto_similarity(target_molecule, product_mol)
                    similarity_scores.append(similarity)
                    if similarity > 0.95:  # High similarity threshold for "match"
                        match_found = True
                        break
                final_products.append(product_mol)

            matches_found.append(match_found)
        else:
            matches_found.append(False)

        combo_scores = []
        shape_scores = []
        colour_scores = []
        if target_pharmacophore_id is not None and final_products and compute_rocs_for_syn:
            # compute conformer for the final product molecule
            # compute ROCS score between the final product molecule and the pharmacophore molecule (from gt/*_ph_*.sdf)
            try:
                gt_sdf_path = os.path.join(
                    save_dir, "gt", f"molecule_ph_{target_pharmacophore_id}.sdf"
                )
                gt_mol = Chem.SDMolSupplier(gt_sdf_path)[0]
                ref_oemol = rdmol_to_oemol(gt_mol)
                combo_scores, shape_scores, colour_scores, _ = score_smiles_to_query(
                    final_products, ref_oemol, metric_name="tanimoto"
                )
                result["max_combo_score"] = (
                    np.max(combo_scores) if combo_scores and len(combo_scores) > 0 else None
                )
                result["max_shape_score"] = (
                    np.max(shape_scores) if shape_scores and len(shape_scores) > 0 else None
                )
                result["max_colour_score"] = (
                    np.max(colour_scores) if colour_scores and len(colour_scores) > 0 else None
                )
            except Exception:
                continue

    # Set match flags
    if matches_found:
        result["top1_match"] = matches_found[0]
        result["any_match"] = any(matches_found)

    # Set average score
    if valid_scores:
        result["avg_score"] = sum(valid_scores) / len(valid_scores)

    result["max_similarity_to_target"] = (
        np.max(similarity_scores) if similarity_scores and len(similarity_scores) > 0 else None
    )

    return result
