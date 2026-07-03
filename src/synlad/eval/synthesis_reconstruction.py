"""Accumulating evaluator and helper utilities for synthesis-pathway predictions."""

from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

from synlad.data.components import synthesis_dataset as synthesis_dataset
from synlad.tokenization import synthesis_serialization as serialization
from synlad.tokenization import synthesis_vocab as vocab


class _BasicMetric:
    """Per-rank running mean over (sum, count) pairs.

    Deliberately not a ``torchmetrics.MeanMetric``: the consuming module feeds the
    scalar produced by :meth:`get_value` into an outer ``MeanMetric`` that handles
    DDP synchronisation via ``self.log(..., sync_dist=True)``. Keeping this local
    accumulator as plain Python avoids registering an extra ``nn.Module`` and the
    associated device-placement / state-dict bookkeeping.
    """

    def __init__(self) -> None:
        self.total_value: float = 0.0
        self.total_count: int = 0

    def add_running_value(self, value: float, count: int) -> None:
        self.total_value += value
        self.total_count += count

    def get_value(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.total_value / self.total_count

    def reset(self) -> None:
        self.total_value = 0.0
        self.total_count = 0


def _validate_building_block(mol_smi: str, current_idx: int, mol_first_indices: dict) -> bool:
    """Check if a building block selection is valid.

    A building block is correct if:
    - It exists in ground truth (is a key in mol_first_indices)
    - It hasn't been selected yet.
    """
    return mol_smi in mol_first_indices and current_idx <= mol_first_indices[mol_smi]


def _validate_reactant_or_product(
    mol_smi: str,
    current_idx: int,
    mol_first_indices: dict,
    rxn_first_indices: dict,
    current_reaction_picks: set,
    is_selecting_reactants: bool,
    reaction_start_indx: int,
) -> bool:
    """Check if a reactant (in forward) or product (in backward) selection is valid.

    A selection is correct if:
    - It has been seen as a molecule (mol_smi in mol_first_indices and current_idx > mol_first_indices[mol_smi])
    - It hasn't been picked in the current reaction yet (not in current_reaction_picks)
    - It is a reactant/product in at least one ground truth reaction that hasn't been picked yet
    """
    # 1. Must have been seen as a molecule already
    if mol_smi not in mol_first_indices or current_idx <= mol_first_indices[mol_smi]:
        return False

    # 2. Must not have been picked in current reaction yet
    if mol_smi in current_reaction_picks:
        return False

    # 3. Must be in an unpicked reaction
    for (reactants_set, product), rxn_idx in rxn_first_indices.items():
        if rxn_idx >= reaction_start_indx:
            # ^ note that the index is checked against the reaction start index as we may actually be in the midst of
            # actually filling in this reaction!

            # This reaction hasn't been picked yet (at least to completion!)
            if is_selecting_reactants:
                # Check if mol_smi is a reactant
                if mol_smi in reactants_set:
                    return True
            else:
                # Check if mol_smi is the product (for reverse reactions)
                if mol_smi == product:
                    return True

    return False


def _predict_invariant_full_accuracies(
    predictions: torch.Tensor, batch: synthesis_dataset.RxnNetTaskBatch
) -> dict[int, tuple[int, int]]:
    """Predict the invariant full accuracies for the arguments given to particular actions. Does for forward, backward
    and retrosynthesis reactions. Meant to be used during teacher forcing evaluation -- so always assumes ground truth
    up to a given molecule is given.

    Note: Even though invariant accuracies try to resolve many of the issues with invariances around the exact
        conditional accuracies (e.g., penalizing building blocks picked out of order), they still mark incorrect valid
        routes that don't match the ground truth.

    Implementation note: this works via a loop and so is slow to run.

    Args:
        predictions (torch.Tensor): The predictions to evaluate. In index form.
        batch (dataset.RxnNetTaskBatch): The ground truth batch.
    Returns:
        The number of correct and total tokens for each action type. keys in dict indicate the action type as an int.
        (as that is what we were doing for the other exact (i.e., not invariant) accuracies).
    """
    # Get token indices for different action types
    BB_TOKEN_IDX = vocab.SPECIAL_TOKENS_LIBRARY.idx_frm_token(vocab.SPECIAL_TOKENS.BB.value)
    FWD_RXN_TOKEN_IDX = vocab.SPECIAL_TOKENS_LIBRARY.idx_frm_token(
        vocab.SPECIAL_TOKENS.FWD_RXN.value
    )
    REV_RXN_TOKEN_IDX = vocab.SPECIAL_TOKENS_LIBRARY.idx_frm_token(
        vocab.SPECIAL_TOKENS.REV_RXN.value
    )

    # Accumulator across all sequences in batch -- stores number of choices of that type correct and total number of
    # of choices.
    total_stats = {
        BB_TOKEN_IDX: [0, 0],
        FWD_RXN_TOKEN_IDX: [0, 0],
        REV_RXN_TOKEN_IDX: [0, 0],
    }

    for i, row in enumerate(predictions):
        prediction_row_as_list = row.detach().cpu().numpy().tolist()
        prediction_row_as_tkn_strs = [
            batch.tkn_lib_collection.token_frm_idx(tkn) for tkn in prediction_row_as_list
        ]

        # We want to see the whole sequence (including first token) so look at `input_sequences` rather than `output_sequences`.
        ground_truth_row_as_list = batch.input_sequences[i].detach().cpu().numpy().tolist()
        ground_truth_row_as_tkn_strs = [
            batch.tkn_lib_collection.token_frm_idx(tkn) for tkn in ground_truth_row_as_list
        ]

        # & we will align the prediction and ground truth rows -- so loop below works. this is because prediction
        # currently contains the next token for each ground truth token. see comment above.
        prediction_row_as_tkn_strs = [
            ground_truth_row_as_tkn_strs[0],
            *prediction_row_as_tkn_strs[:-1],
        ]

        # Get first occurrence indices for molecules and reactions
        mol_first_indices, rxn_first_indices = (
            serialization.Deserializer.get_molecule_reaction_first_indices(
                ground_truth_row_as_tkn_strs, batch.tkn_lib_collection
            )
        )

        # State tracking
        current_action = None  # Current action token we're processing arguments for
        current_reaction_picks = set()  # Molecules picked in current reaction
        run_delimiter_count = 0  # Track position in reaction (before/after RUN delimiters)
        reaction_start_indx = 0

        # Process token by token
        for token_idx, (pred_token_str, gt_token_str) in enumerate(
            zip(prediction_row_as_tkn_strs, ground_truth_row_as_tkn_strs, strict=False)
        ):
            # Update state based on ground truth token
            if gt_token_str in vocab.SPECIAL_TOKENS:
                gt_token_enum = vocab.SPECIAL_TOKENS(gt_token_str)

                if gt_token_enum in {
                    vocab.SPECIAL_TOKENS.BB,
                    vocab.SPECIAL_TOKENS.FWD_RXN,
                    vocab.SPECIAL_TOKENS.REV_RXN,
                }:
                    current_action = gt_token_str
                    current_reaction_picks.clear()
                    run_delimiter_count = 0
                    reaction_start_indx = token_idx
                elif gt_token_enum in {
                    vocab.SPECIAL_TOKENS.FWD_RXN_RUN,
                    vocab.SPECIAL_TOKENS.REV_RXN_RUN,
                }:
                    run_delimiter_count += 1
                else:  # EOS, PAD, prefix etc.  not an action we will be evaluating for though.
                    current_action = None

            else:
                if current_action == vocab.SPECIAL_TOKENS.BB.value:
                    # Validate building block selection
                    is_correct = _validate_building_block(
                        pred_token_str, token_idx, mol_first_indices
                    )
                    total_stats[BB_TOKEN_IDX][0] += int(is_correct)
                    total_stats[BB_TOKEN_IDX][1] += 1

                elif current_action == vocab.SPECIAL_TOKENS.FWD_RXN.value:
                    # For forward reactions, we pick reactants before first RUN delimiter
                    if run_delimiter_count == 0:
                        is_correct = _validate_reactant_or_product(
                            pred_token_str,
                            token_idx,
                            mol_first_indices,
                            rxn_first_indices,
                            current_reaction_picks,
                            is_selecting_reactants=True,
                            reaction_start_indx=reaction_start_indx,
                        )
                        total_stats[FWD_RXN_TOKEN_IDX][0] += int(is_correct)
                        total_stats[FWD_RXN_TOKEN_IDX][1] += 1
                        current_reaction_picks.add(gt_token_str)  # gt is now in the reaction picks.

                elif current_action == vocab.SPECIAL_TOKENS.REV_RXN.value:
                    # For reverse reactions, we pick products before first RUN delimiter
                    if run_delimiter_count == 0:
                        is_correct = _validate_reactant_or_product(
                            pred_token_str,
                            token_idx,
                            mol_first_indices,
                            rxn_first_indices,
                            current_reaction_picks,
                            is_selecting_reactants=False,
                            reaction_start_indx=reaction_start_indx,
                        )
                        total_stats[REV_RXN_TOKEN_IDX][0] += int(is_correct)
                        total_stats[REV_RXN_TOKEN_IDX][1] += 1
                        current_reaction_picks.add(gt_token_str)  # gt is now in the reaction picks.

    return total_stats


@lru_cache(maxsize=4096)
def _calculate_tanimoto_similarity(smi1: str, smi2: str) -> float:
    """Calculate Tanimoto similarity between two SMILES strings using Morgan fingerprints."""

    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)

        if mol1 is None or mol2 is None:
            return 0.0

        # Use Morgan fingerprints with radius 2
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        fp1 = fp_gen.GetFingerprint(mol1)
        fp2 = fp_gen.GetFingerprint(mol2)

        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def _evaluate_predictions(
    batch: synthesis_dataset.RxnNetTaskBatch,
    picked_predictions: torch.Tensor,
    return_all_accuracies: bool = False,
    return_padded: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-token / per-sequence accuracy for a batch of synthesis predictions.

    Args:
        batch: The batch to evaluate.
        picked_predictions: The predicted token indices, shape ``[B, S*]`` (may be nested).
        return_all_accuracies: If True, also return the per-token correctness mask.
        return_padded: If True, return padded tensors even for nested batches.
    """
    if batch.input_sequences.is_nested:
        # Convert nested tensors to padded form
        all_preds_together = torch.nested.to_padded_tensor(picked_predictions, padding=0.0)
        all_output_sequences = torch.nested.to_padded_tensor(batch.output_sequences, padding=0.0)
        valid_locs = torch.nested.to_padded_tensor(batch.output_from_model_masks, padding=False)
    else:
        all_preds_together = picked_predictions
        all_output_sequences = batch.output_sequences
        valid_locs = torch.logical_and(batch.input_nonpad_masks, batch.output_from_model_masks)

    # Compute accuracy per sequence
    num_tokens_per_sequence = valid_locs.sum(dim=1)  # [B]
    num_correct = all_preds_together == all_output_sequences  # [B, S']
    num_correct[torch.logical_not(valid_locs)] = 0  # [B, S']
    num_correct_per_sequence = num_correct.sum(dim=1)  # [B]
    accuracy_per_sequence = num_correct_per_sequence / num_tokens_per_sequence
    out0 = accuracy_per_sequence

    if return_all_accuracies:
        if batch.input_sequences.is_nested and not return_padded:
            seq_lens = batch.input_sequences.offsets().diff()
            out1 = torch.nested.narrow(num_correct, dim=1, length=seq_lens, layout=torch.jagged)
        else:
            out1 = num_correct

    if return_all_accuracies:
        return out0, out1
    else:
        return out0


class SynthesisEvaluator:
    """
    Evaluator for synthesis pathway prediction tasks.
    Accumulates predictions and targets across batches and computes metrics at the end.
    """

    def __init__(self, provide_examples: bool = True, full_accuracies: bool = False):
        """
        Initialize the synthesis evaluator.

        Args:
            provide_examples (bool): Whether to provide examples of predictions
            full_accuracies (bool): Whether to compute detailed accuracies by token type
        """
        self.provide_examples = provide_examples
        self.full_accuracies = full_accuracies

        # Main metrics
        self.metrics = {
            "token_accuracy": _BasicMetric(),
            "sequence_accuracy": _BasicMetric(),
        }

        # Storage for molecule-synthesis matching
        self.synthesis_data = []  # Storage for synthesis pathway data

        # Full accuracy metrics
        if self.full_accuracies:
            self.metrics["action_cond_accuracy"] = _BasicMetric()
            self.metrics["action_avg_num_per_seq"] = _BasicMetric()
            self.metrics["total_avg_num_per_seq"] = _BasicMetric()

            # Action-specific metrics
            self._action_token_to_arg_metric_name = {
                vocab.SPECIAL_TOKENS_LIBRARY.idx_frm_token(vocab.SPECIAL_TOKENS.BB.value): "BB-ID",
                vocab.SPECIAL_TOKENS_LIBRARY.idx_frm_token(
                    vocab.SPECIAL_TOKENS.FWD_RXN.value
                ): "FWD-SELECTION",
            }

            for metric_name in self._action_token_to_arg_metric_name.values():
                self.metrics[f"{metric_name}_cond_accuracy"] = _BasicMetric()
                self.metrics[f"{metric_name}_avg_num_per_seq"] = _BasicMetric()
                self.metrics[f"{metric_name}_cond_accuracy_invariant"] = _BasicMetric()

        # Storage for examples
        self.output_examples = []
        self.total_number = 0

        # Storage for accumulated data
        self.accumulated_batches = []
        self.accumulated_preds = []

    def clear(self):
        """Clear stored predictions and reset metrics."""
        for metric in self.metrics.values():
            metric.reset()
        self.output_examples = []
        self.total_number = 0
        self.accumulated_batches = []
        self.accumulated_preds = []
        self.synthesis_data = []

    def update(self, batch, logits, batch_idx=None, global_rank=0):
        """
        Update the evaluator with a batch of predictions.

        Args:
            batch: The input batch containing pathways data
            logits: Model logits for the batch
            batch_idx: Batch index for calculating sample indices
            global_rank: Global rank for distributed training
        """
        # Store the batch and logits for later processing
        batch_size = len(batch)
        self.accumulated_batches.append(batch)

        preds = logits.argmax(dim=-1)  # [B, S*] (can be nested)
        self.accumulated_preds.append(preds)

        accuracies = _evaluate_predictions(batch, preds, self.full_accuracies, return_padded=True)
        if self.full_accuracies:
            accuracies, all_accuracies = accuracies

        self.metrics["token_accuracy"].add_running_value(accuracies.sum().item(), batch_size)
        self.metrics["sequence_accuracy"].add_running_value(
            (accuracies == 1).sum().item(), batch_size
        )
        self.total_number += batch_size

        # Handle full accuracies if enabled
        if self.full_accuracies:
            padded_batch = batch.to_padded()
            token_types = synthesis_dataset.padded_tokens_to_token_type(
                padded_batch.output_sequences, padded_batch.tkn_lib_collection
            )

            # first do this for action tokens.
            action_tokens = torch.logical_and(
                torch.logical_and(token_types == 0, padded_batch.output_from_model_masks),
                padded_batch.input_nonpad_masks,
            )
            num_actions = action_tokens.sum().item()
            num_correct_actions = all_accuracies[action_tokens].sum().item()
            self.metrics["action_cond_accuracy"].add_running_value(num_correct_actions, num_actions)
            self.metrics["action_avg_num_per_seq"].add_running_value(num_actions, batch_size)
            self.metrics["total_avg_num_per_seq"].add_running_value(
                torch.logical_and(
                    padded_batch.output_from_model_masks, padded_batch.input_nonpad_masks
                )
                .sum()
                .item(),
                batch_size,
            )

            # then do this for the action arguments.
            action_argument_locations = synthesis_dataset.padded_tokens_defining_action_type(
                padded_batch.output_sequences, padded_batch.tkn_lib_collection
            )
            # ^ by doing this on the input sequences, we will not get the correct sequence type for the prompt
            # but this is not that important as not evaluating the accuracy here (it is always given!).
            for action_token, metric_name in self._action_token_to_arg_metric_name.items():
                action_argument_tokens = torch.logical_and(
                    torch.logical_and(
                        action_argument_locations == action_token,
                        padded_batch.output_from_model_masks,
                    ),
                    padded_batch.input_nonpad_masks,
                )
                num_action_arguments = action_argument_tokens.sum().item()
                num_correct_action_arguments = all_accuracies[action_argument_tokens].sum().item()
                self.metrics[f"{metric_name}_cond_accuracy"].add_running_value(
                    num_correct_action_arguments, num_action_arguments
                )
                self.metrics[f"{metric_name}_avg_num_per_seq"].add_running_value(
                    num_action_arguments, batch_size
                )

            invariant_accuracies = _predict_invariant_full_accuracies(preds, padded_batch)
            for action_token_idx, metric_name in self._action_token_to_arg_metric_name.items():
                invariant_acc_correct_and_total = invariant_accuracies[action_token_idx]
                self.metrics[f"{metric_name}_cond_accuracy_invariant"].add_running_value(
                    *invariant_acc_correct_and_total
                )

        if self.provide_examples:
            if hasattr(batch, "tkn_lib_collection") and batch_idx is not None:
                tkn_lib = batch.tkn_lib_collection
                for j in range(min(25, len(batch))):  # Only logging 25 examples
                    sample_idx = (batch_idx + global_rank) * batch_size + j
                    ce_for_prediction = F.cross_entropy(
                        input=logits[j], target=batch.output_sequences[j], reduction="none"
                    )
                    prob = torch.softmax(logits[j], dim=-1)
                    self.output_examples.append(
                        {
                            "sample_idx": sample_idx,
                            "task": tkn_lib.token_frm_idx(batch.input_sequences[j][0].item()),
                            "target": [
                                tkn_lib.token_frm_idx(int(el.item()))
                                for el in batch.output_sequences[j]
                            ],
                            "prediction": [
                                tkn_lib.token_frm_idx(int(el.item())) for el in preds[j]
                            ],
                            "losses": [
                                ce.item() if v else np.nan
                                for ce, v in zip(
                                    ce_for_prediction,
                                    batch.output_from_model_masks[j],
                                    strict=False,
                                )
                            ],
                            "probs_target": [
                                prob[ii, el].item()
                                for ii, el in enumerate(batch.output_sequences[j])
                            ],
                            "probs_prediction": [
                                prob[ii, el].item() for ii, el in enumerate(preds[j])
                            ],
                        }
                    )

        # Store synthesis data for molecule matching
        for j in range(batch_size):
            self.synthesis_data.append([tkn_lib.token_frm_idx(int(el.item())) for el in preds[j]])

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

    def compute_metrics(self) -> dict[str, float]:
        """Compute final metrics from accumulated data."""
        output = {k: v.get_value() for k, v in self.metrics.items()}

        return output

    def get_examples(self) -> list[dict[str, Any]]:
        """Get collected examples."""
        return self.output_examples
