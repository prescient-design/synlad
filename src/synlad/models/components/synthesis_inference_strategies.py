import abc
import enum
import functools
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch.nn import functional as F

from synlad.data.components import synthesis_dataset as dataset
from synlad.models import synthesis_decoder
from synlad.models.components.synthesis_token_action_handler import TokenActionHandler
from synlad.tokenization import synthesis_vocab as vocab
from synlad.utils import synthesis_utils as utils


class StoppingCriteria(enum.Enum):
    """Enum for different stopping criteria for inference strategies."""

    ONE_TOKEN = "one_token"
    COMPLETE_ACTION = "complete_action"
    UNTIL_STOP = "until_stop"


@dataclass(frozen=True)
@functools.total_ordering
class CompletedSequence:
    """Represents a completed sequence with its score.

    The sequence can be either integer token indices (tuple) or string tokens (tuple).
    Supports comparison operations for sorting/heap operations based on score.
    """

    sequence: tuple[int, ...] | tuple[str, ...]
    score: float

    def __len__(self) -> int:
        """Returns the length of the sequence."""
        return len(self.sequence)

    @property
    def is_int_form(self) -> bool:
        """Returns True if sequence is in integer token form."""
        return len(self.sequence) > 0 and isinstance(self.sequence[0], int)

    def to_str_form(
        self, token_lib_collection: vocab.TokenLibraryCollection
    ) -> "CompletedSequence":
        """Convert to string token form using the token library collection.

        Returns a new instance with string tokens.
        """
        if not self.is_int_form:
            return self  # Already in string form

        str_tokens = tuple(token_lib_collection.token_frm_idx(idx) for idx in self.sequence)
        return CompletedSequence(sequence=str_tokens, score=self.score)

    def _check_comparable(self, other: "CompletedSequence") -> None:
        """Check if two sequences can be compared."""
        if not isinstance(other, CompletedSequence):
            raise TypeError(f"Cannot compa re CompletedSequence with {type(other)}")

        if self.is_int_form != other.is_int_form:
            raise ValueError("Cannot compare integer form sequence with string form sequence")

    def __lt__(self, other: "CompletedSequence") -> bool:
        self._check_comparable(other)
        return self.score < other.score

    def __eq__(self, other: object) -> bool:
        # if not same type, not equal
        if not isinstance(other, CompletedSequence):
            return False
        self._check_comparable(other)
        if self.sequence == other.sequence:
            if not self.score == other.score:
                raise ValueError("Sequence is equal but scores are not")
            return True
        return False


class BatchCompletedSequences:
    """Manages completed sequences for a batch using lists for each batch index.

    Maps batch indices to lists of completed sequences, with score normalization
    and automatic padding removal.
    """

    def __init__(
        self,
        token_lib_collection: vocab.TokenLibraryCollection,
        score_normalizer: Callable[[tuple[int, ...], float], float] | None = None,
    ):
        self.token_lib_collection = token_lib_collection
        self.score_normalizer = score_normalizer
        self.completed_sequences: defaultdict[int, list[CompletedSequence]] = defaultdict(list)

    def _normalize_score(self, sequence: tuple[int, ...], score: float) -> float:
        """Apply score normalization if provided."""
        if self.score_normalizer is not None:
            return self.score_normalizer(sequence, score)
        return score

    def _convert_sequence_to_tuple(self, sequence, remove_padding: bool = True) -> tuple[int, ...]:
        """Convert sequence from tensor/array to tuple, optionally removing padding."""
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.detach().cpu().numpy()
        if isinstance(sequence, np.ndarray):
            sequence = sequence.tolist()

        sequence = tuple(sequence)

        if remove_padding:
            # Remove padding tokens (assuming PAD_VALUE is defined)
            pad_value = vocab.PAD_VALUE
            # Find first padding token and truncate there
            try:
                pad_idx = sequence.index(pad_value)
            except ValueError:
                # No padding found, keep full sequence
                pass
            else:
                # Validate that all subsequent tokens are also padding
                if any(sequence[i] != pad_value for i in range(pad_idx + 1, len(sequence))):
                    raise ValueError(f"Non-padding token found after padding at position {pad_idx}")
                sequence = sequence[:pad_idx]

        return sequence

    def add_single_completion(self, batch_idx: int, sequence, score: float) -> None:
        """Add a single completed sequence to the specified batch index."""
        # Convert and clean sequence
        seq_tuple = self._convert_sequence_to_tuple(sequence)

        # Normalize score
        normalized_score = self._normalize_score(seq_tuple, score)

        # Create completed sequence
        completed_seq = CompletedSequence(sequence=seq_tuple, score=normalized_score)

        # Add to list for this batch index
        self.completed_sequences[batch_idx].append(completed_seq)
        # could consider heapq for this, but at moment just using list.

    def add_batch_completions(self, sequences, scores, batch_indices) -> None:
        """Add multiple completed sequences from batch tensors."""
        # Add each completion
        for seq, score, batch_idx in zip(sequences, scores, batch_indices, strict=False):
            self.add_single_completion(int(batch_idx), seq, float(score))

    def get_top_k(self, k: int = 1) -> dict[int, list[CompletedSequence]]:
        """Get top-k completed sequences for each batch index."""
        result = {}
        for batch_idx, sequences in self.completed_sequences.items():
            # Remove duplicates using set (since CompletedSequence is hashable)
            unique_sequences = list(set(sequences))
            # Sort by score (descending) and take top k
            sorted_sequences = sorted(unique_sequences, key=lambda x: x.score, reverse=True)
            result[batch_idx] = sorted_sequences[:k]
        return result

    def get_completion_counts(self) -> dict[int, int]:
        """Get number of completed sequences for each batch index."""
        return {
            batch_idx: len(sequences) for batch_idx, sequences in self.completed_sequences.items()
        }

    def get_all_batch_indices(self) -> list[int]:
        """Get all batch indices that have completed sequences."""
        return list(self.completed_sequences.keys())

    def to_str_form(self) -> "BatchCompletedSequences":
        """Convert all sequences to string form using token library."""
        new_instance = BatchCompletedSequences(
            token_lib_collection=self.token_lib_collection, score_normalizer=self.score_normalizer
        )

        for batch_idx, sequences in self.completed_sequences.items():
            str_sequences = [seq.to_str_form(self.token_lib_collection) for seq in sequences]
            new_instance.completed_sequences[batch_idx] = str_sequences

        return new_instance


class InferenceStrategy(Protocol):
    """Protocol for inference strategies (sampling, beam search, etc.) that drive token generation.

    Implementations advance ``current_sequences`` until their stopping criteria are met
    and record any newly completed sequences in ``current_full_completions``.
    """

    def generate(
        self,
        current_sequences: dataset.CurrentSequences,
        current_full_completions: BatchCompletedSequences,
    ) -> tuple[dataset.CurrentSequences, BatchCompletedSequences]:
        """
        Completes the `current_sequences` according to the stopping criteria of the inference strategy.

        Returns:
          A new CurrentSequences dataclass with the new actions added to end of the old sequences.
          An updated BatchCompletedSequences (changes made in place) with any completions that are now complete recorded.
        """
        ...


class OneStepActor(abc.ABC):
    """
    One step actors represent the policy for next possible actions. What makes up an action depends on the
    particular implementation. For example, a model may only output a single next token, but an inference strategy
    may output a series of next tokens (depending on its stopping criteria) when used as a one step actor.
    """

    def __init__(self, token_action_handler: TokenActionHandler):
        self.token_action_handler = token_action_handler

    def get_actions_scores_values(
        self,
        current_sequences: dataset.CurrentSequences,
        seqs_over: torch.TensorType | None = None,
        conditioning: torch.TensorType | None = None,
        conditioning_mask: torch.TensorType | None = None,
    ) -> tuple[torch.TensorType, torch.TensorType, torch.TensorType | None]:
        """
        Gets the actions, scores, and values for the members of the current sequences that are not over.

        Actions [B, A, S] (B batch size, A action size, S sequence length of new action -- by default 1, unless multi-step)
        (can contain padding).

        Scores [B, A] (B batch size, A action size); higher is better. Gives scores of different actions for current sequence.
        if action is padding, score is -inf

        Values [B] (if available, None otherwise). Gives the value of the current sequence if if applicable

        If all sequences are over, returns `None, None, None`.

        Parameters:
            current_sequences: CurrentSequences to get actions, scores, and values for.
            seqs_over: Optional mask of sequences that are over. If provided, only the sequences that are not over
              will be considered. If not provided, all sequences will be considered.

        Returns:
            Tuple of actions, scores, and values.
        """
        # Use the new method from RxnNetTaskBatch to determine which sequences are over
        if seqs_over is not None and seqs_over.all():
            return None, None, None
        if seqs_over is not None:
            current_sequences = current_sequences.select_submask(~seqs_over)
            # Also filter conditioning to match the non-over sequences
            if conditioning is not None:
                # Use example_indices to properly map beam sequences back to original conditioning
                continuing_example_indices = current_sequences.example_indices
                conditioning = conditioning[continuing_example_indices]
                conditioning_mask = conditioning_mask[continuing_example_indices]

        return self._get_actions_scores_values(current_sequences, conditioning, conditioning_mask)

    @classmethod
    def get_last_action(cls, actions: torch.TensorType) -> torch.TensorType:
        """
        Gets the last (non-pad) action token for each element in actions.

        Args:
            actions: Tensor of shape [B, A, S] where B is batch size, A is action size,
                    S is sequence length of new action

        Returns:
            Tensor of shape [B, A] containing the last non-padding action token for each sequence. (if all padding then
            first token is returned for that sequence).
        """
        # if a single token, then just return it the only one:
        if actions.shape[-1] == 1:
            return actions.squeeze(-1)  # [B, A]

        # note that padding must only come at end of sequence, so can just count number of non-pad tokens:
        non_pad_mask = actions != vocab.PAD_VALUE  # [B, A, S]
        seq_lengths = non_pad_mask.sum(dim=-1)  # [B, A]
        last_positions = seq_lengths - 1  # [B, A]

        # Handle edge case where all tokens are padding (seq_lengths == 0). In this case, we'll just take
        #  the first token (which will be PAD_VALUE)
        last_positions = torch.clamp(last_positions, min=0)  # [B, A]

        last_actions = torch.gather(
            actions, dim=-1, index=last_positions.unsqueeze(-1)
        )  # [B, A, 1]
        return last_actions.squeeze(-1)  # [B, A]

    @abc.abstractmethod
    def _get_actions_scores_values(
        self,
        current_sequences: dataset.CurrentSequences,
        conditioning: torch.TensorType | None = None,
        conditioning_mask: torch.TensorType | None = None,
    ) -> tuple[torch.TensorType, torch.TensorType, torch.TensorType | None]:
        """
        Actions [B', A, S] (B' batch size, A action size, S sequence length of new action -- by default 1, unless multi-step)
        (can contain padding)

        Scores [B', A] (B' batch size, A action size); higher is better. Gives scores of different actions for current sequence.
        if action is padding, score is -inf

        Values [B'] (if available, None otherwise). Gives the value of the current sequence if if applicable

        B' is the batch size of the current sequences that are not over.
        """
        ...

    @torch.no_grad()
    def turn_chosen_actions_into_new_sequences(
        self,
        current_sequences: dataset.CurrentSequences,
        actions: torch.TensorType,
        scores: torch.TensorType,
        chosen_actions: torch.TensorType,
        seqs_over_mask: torch.TensorType | None = None,
    ) -> dataset.CurrentSequences:
        """
        Takes the chosen actions and adds them to the end of the current sequences to create new sequences. Note that
        multiple actions may be chosen for each current sequence. (i.e., dim A below is > 1). Note that the
        members of the current sequences that are already over are left unchanged -- these are indicated by the
        `seqs_over_mask` parameter. If all sequences are already over, then raises ValueError.

        NOTE THAT PADDING FOR CHOSEN ACTIONS IS REPRESENTED BY -1.
        (cannot use 0 as padding as this is a valid action index).

        B' is the batch size of the current sequences that are not over.

        Parameters:
            current_sequences: CurrentSequences to be adapted. Size B.
            actions: Actions to be added to the end of the current sequences. Size B', A, S.
            scores: Scores of the actions. Size B', A.
            chosen_actions: Indices of actions chosen for each current sequence. Size B', A'. -1 to represent padding.
               If no actions are chosen then whole row should be -1 and sequence will be dropped.
            seqs_over_mask: Optional mask of sequences that are over. If provided, only the sequences that are not over will
                be considered. The sequences over will be left unchanged and stitched back together with the changed
                sequences at the end. If not provided, all sequences will be considered. This differs from keeping a
                whole chosen action row as -1, where the sequence is **instead dropped**.

        Returns:
            CurrentSequences with the new actions added to end of the old sequences, merged with unchanged old sequences.
            Note does not currently remove graphs or token library items that are no longer needed (i.e., exist only
            in dropped sequences).
        """
        # === 1. Copy any inputs that we will modify in place ===
        chosen_actions = chosen_actions.clone()  # we will modify this in place, so need to clone

        # === 2. Check that the inputs are consistent and pull out the sequences which actually need to be processed
        # (i.e., those that are not over) ===
        if seqs_over_mask is not None and seqs_over_mask.all():
            raise ValueError("Current sequences are all complete, so no new actions can be added!")
        if seqs_over_mask is None:
            current_sequences_not_over = current_sequences
            seqs_over_mask = torch.zeros(
                len(current_sequences), device=current_sequences.device, dtype=torch.bool
            )
        else:
            current_sequences_not_over = current_sequences.select_submask(~seqs_over_mask)
            num_seqs_not_over = (~seqs_over_mask).sum().item()  # this is B'

            # this is just checking that the actions, scores, and chosen actions have a first dim of B' rather than B
            if num_seqs_not_over != chosen_actions.shape[0]:
                raise ValueError(
                    "chosen_actions should be consistent with the number of sequences that are not over"
                )
            if num_seqs_not_over != actions.shape[0]:
                raise ValueError(
                    "actions should be consistent with the number of sequences that are not over"
                )
            if num_seqs_not_over != scores.shape[0]:
                raise ValueError(
                    "scores should be consistent with the number of sequences that are not over"
                )

        # === 3. Process sequences that are not over ===
        # 3a. work out how many actions are chosen for each sequence and pick out those which were selected (i.e., not
        # padded)
        num_actions_per_choice = torch.sum(chosen_actions != -1, dim=-1)
        if (num_actions_per_choice == 0).all():
            raise NotImplementedError("Currently do not support dropping all sequences.")
        if num_actions_per_choice.max() > 1 or (num_actions_per_choice.min() == 0):
            new_current_sequences = current_sequences_not_over.repeat(num_actions_per_choice)

            # repeat should work with 0s, but being defensive here and checking that it has:
            if (num_actions_per_choice == 0).any():
                assert len(new_current_sequences) == num_actions_per_choice.sum()
        else:
            new_current_sequences = current_sequences_not_over

        chosen_actions_packed = chosen_actions.view(-1)  # [B'* A']
        non_padded_actions_packed = chosen_actions_packed != -1
        chosen_actions[chosen_actions == -1] = (
            0  # makes the gathers below work (doesnt matter indx here as discard later) [B', A']
        )

        # implementation note:
        #  we over select (taking advantage of the fact that changed padding to 0) and then pick out the ones which we actually
        #  wanted later. this is a bit wasteful (memory and also compute wise), so maybe could revisit this later.
        actions = torch.gather(
            actions,
            dim=1,
            index=chosen_actions[:, :, None].expand(*chosen_actions.shape, actions.shape[2]),
        )  # [B', A', S]
        all_actions_packed = actions.view(-1, actions.shape[-1])  # [B'* A', S]
        all_actions_packed = all_actions_packed[
            non_padded_actions_packed
        ]  # [Q, S], where Q is total number of non-padded actions
        # across all sequences

        # 3b. Do the same thing for scores of last step using same pattern as actions
        selected_scores = torch.gather(scores, dim=1, index=chosen_actions)
        all_scores_packed = selected_scores.view(-1)  # [B'* A']
        all_scores_packed = all_scores_packed[non_padded_actions_packed]  # [Q]

        # 3c. Process these selected actions (i.e., fill in any missing tokens -- from reactions -- )
        processed_sequences = self.token_action_handler(new_current_sequences, all_actions_packed)

        # === 4. Merge back together processed and unchanged sequences (i.e., those that were already over). or... ===
        if seqs_over_mask is not None and seqs_over_mask.any():
            # 4a. Create the indices needed for doing the merge (i.e., where the finsihed and changed sequences
            # get stitched/interleaved together)

            over_sequences = current_sequences.select_submask(
                seqs_over_mask
            )  # <- seqs that are already over (unchanged)

            # In order to stitch together the processed and unchanged sequences we need to know where they go.
            # This is made a bit more complicated by the fact that we have to account for the fact that the sequences
            # that were processed had multiple actions chosen for them. We therefore need to create offsets first
            # to work out how much to shift the indices of the changed and unchanged sequences.
            indx_offsets_in_original_space = torch.zeros_like(
                seqs_over_mask, dtype=torch.int64
            )  # [B]
            offset_indices = (
                torch.nonzero(~seqs_over_mask, as_tuple=False).squeeze(1) + 1
            )  # [B'] < [B]
            # ^ note that these offsets start affect the next index (why a plus one),
            offset_indices = offset_indices[
                : -1 if offset_indices[-1] == seqs_over_mask.shape[0] else None
            ]  # [B' or B' -1] < [B]
            # ^ we also do not care about getting the offset right for sequences after the end! (also affects slice below)
            indx_offsets_in_original_space[offset_indices] = (
                num_actions_per_choice[: len(offset_indices)] - 1
            )
            # ^ only more than one action leads to an index offset. note if no actions chosen, then offset will be -1,
            # as the sequence is dropped.
            indx_offsets_in_original_space = torch.cumsum(
                indx_offsets_in_original_space, dim=0
            )  # [B]
            old_indices_to_new_indices = (
                torch.arange(seqs_over_mask.shape[0], device=seqs_over_mask.device)
                + indx_offsets_in_original_space
            )  # [B]
            # ^ note that if we have a load of sequences that are being dropped at the beginning, then this will show
            # up as a negative index at -1. This is fine, as we never actually use these indices (and we check this below).
            # similar thing also happens where we use duplicated indices for sequences that are dropped, fine, as not
            # actually ever picked.

            # and then having done this we can now map the indices of unprocessed and processed sequences to their new
            # locations (not for processed there will be multiple for each original sequence, depending on the number
            # of actions chosen for it).
            indices_for_seqs_over = old_indices_to_new_indices[seqs_over_mask]
            assert indices_for_seqs_over.min() >= 0, "Indices should not be negative"
            indices_for_seqs_not_over = (
                torch.arange(chosen_actions.shape[1], device=chosen_actions.device)[None, :]
                + old_indices_to_new_indices[~seqs_over_mask][:, None]
            )
            indices_for_seqs_not_over = indices_for_seqs_not_over.view(-1)
            indices_for_seqs_not_over = indices_for_seqs_not_over[non_padded_actions_packed]
            assert indices_for_seqs_not_over.min() >= 0, "Indices should not be negative"

            # 4b. Now we will create the new datastructures. Means stitching together the processed and unchanged sequences.
            new_max_seq_size = max(
                processed_sequences.input_sequences.shape[-1],
                over_sequences.input_sequences.shape[-1],
            )
            new_batch_size = indices_for_seqs_over.shape[0] + indices_for_seqs_not_over.shape[0]

            # Helper function to create and fill tensors
            def create_merged_tensor(processed_tensor, over_tensor, fill_value=None):
                # Both tensors must have the same None status
                if (processed_tensor is None) != (over_tensor is None):
                    raise ValueError("Both tensors must be None or both must be not None")

                if processed_tensor is None:
                    return None

                # Determine dtype and fill_value
                dtype = processed_tensor.dtype
                if fill_value is None:
                    if dtype == torch.bool:
                        fill_value = False
                    else:
                        fill_value = vocab.PAD_VALUE

                # Then create the new tensor with the correct shape
                shape_map = {
                    1: (new_batch_size,),
                    2: (new_batch_size, new_max_seq_size),
                    3: (new_batch_size, new_max_seq_size, processed_tensor.shape[-1]),
                    # ^ note for 3 dimensions, the last dimension is the vocab size, this may change in processed
                    # compared to vocab (but only at the end), hence take shape from this rather than vocab.
                }

                ndim = len(processed_tensor.shape)
                if ndim not in shape_map:
                    raise ValueError(f"Unsupported tensor shape: {processed_tensor.shape}")

                new_tensor = torch.full(
                    shape_map[ndim], fill_value, device=processed_sequences.device, dtype=dtype
                )

                # Finally add the processed and unchanged sequences to the new tensor in the correct places
                proc_slices = [indices_for_seqs_not_over] + [
                    slice(0, s_) for s_ in processed_tensor.shape[1:]
                ]
                over_slices = [indices_for_seqs_over] + [
                    slice(0, s_) for s_ in over_tensor.shape[1:]
                ]

                new_tensor[tuple(proc_slices)] = processed_tensor
                new_tensor[tuple(over_slices)] = over_tensor

                return new_tensor

            # Create all the merged tensors
            new_input_sequences = create_merged_tensor(
                processed_sequences.input_sequences, over_sequences.input_sequences, vocab.PAD_VALUE
            )
            new_input_mol_scores = create_merged_tensor(
                processed_sequences.input_mol_scores, over_sequences.input_mol_scores, 0.0
            )
            new_input_predictive_nxt_tkn_masks = create_merged_tensor(
                processed_sequences.input_predictive_nxt_tkn_masks,
                over_sequences.input_predictive_nxt_tkn_masks,
                False,
            )
            new_input_frm_model_masks = create_merged_tensor(
                processed_sequences.input_frm_model_masks,
                over_sequences.input_frm_model_masks,
                False,
            )
            new_input_mol_masks = create_merged_tensor(
                processed_sequences.input_mol_masks, over_sequences.input_mol_masks, False
            )
            new_input_nonpad_masks = create_merged_tensor(
                processed_sequences.input_nonpad_masks, over_sequences.input_nonpad_masks, False
            )
            new_output_sequences = create_merged_tensor(
                processed_sequences.output_sequences,
                over_sequences.output_sequences,
                vocab.PAD_VALUE,
            )
            new_output_from_model_masks = create_merged_tensor(
                processed_sequences.output_from_model_masks,
                over_sequences.output_from_model_masks,
                False,
            )

            # Handle CurrentSequences-specific fields
            new_example_indices = create_merged_tensor(
                new_current_sequences.example_indices, over_sequences.example_indices, 0
            )

            new_scores = dataset.add_new_tensors_to_jagged_padded(
                new_current_sequences.scores, all_scores_packed[:, None], 0.0
            )
            new_scores = create_merged_tensor(new_scores, over_sequences.scores, 0.0)
            new_scores_of_last_step = create_merged_tensor(
                all_scores_packed, over_sequences.scores_of_last_step, 0.0
            )

            return dataset.CurrentSequences(
                tkn_lib_collection=processed_sequences.tkn_lib_collection,
                graphs=processed_sequences.graphs,
                input_sequences=new_input_sequences,
                input_mol_scores=new_input_mol_scores,
                input_predictive_nxt_tkn_masks=new_input_predictive_nxt_tkn_masks,
                input_frm_model_masks=new_input_frm_model_masks,
                input_mol_masks=new_input_mol_masks,
                input_nonpad_masks=new_input_nonpad_masks,
                output_sequences=new_output_sequences,
                output_from_model_masks=new_output_from_model_masks,
                example_indices=new_example_indices,
                scores=new_scores,
                scores_of_last_step=new_scores_of_last_step,
                mol_scorer=current_sequences.mol_scorer,  # use old one, as will have both scorers from old and new
            )
        else:
            # === ...5. No sequences were over, just return the processed sequences. ===
            # No sequences were over, just return the processed sequences.
            return dataset.CurrentSequences(
                **processed_sequences.__dict__,
                example_indices=new_current_sequences.example_indices,
                scores=dataset.add_new_tensors_to_jagged_padded(
                    new_current_sequences.scores, all_scores_packed[:, None], 0.0
                ),
                scores_of_last_step=all_scores_packed,
                mol_scorer=new_current_sequences.mol_scorer,
            )


class PDistributionWarper(abc.ABC):
    """
    Warps the distribution of the probability distribution of next action
    (e.g., temperature scaling, top-k/top-p filtering, etc.)
    """

    @abc.abstractmethod
    def __call__(self, scores: torch.TensorType) -> torch.TensorType:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ComposePDistributionWarper(PDistributionWarper):
    """
    Composes multiple PDistributionWarpers.
    """

    def __init__(self, *warpers: PDistributionWarper):
        self.warpers = warpers

    def __call__(self, scores: torch.TensorType) -> torch.TensorType:
        for warper in self.warpers:
            scores = warper(scores)
        return scores

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(repr(warper) for warper in self.warpers)})"


class LogitsToLogProbs(PDistributionWarper):
    """Convert raw logits to log-probabilities via ``log_softmax`` over the last dimension."""

    def __call__(self, scores: torch.TensorType) -> torch.TensorType:
        return F.log_softmax(scores, dim=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class TemperatureScaling(PDistributionWarper):
    """
    Scales the scores by a temperature.
    """

    def __init__(self, temperature: float, on_logits: bool = True):
        self.temperature = temperature
        self.on_logits = on_logits
        if not on_logits:
            raise ValueError("TemperatureScaling only currently works on logits, not probabilities")

    def __call__(self, scores: torch.TensorType) -> torch.TensorType:
        return scores / self.temperature

    def __repr__(self):
        return f"{self.__class__.__name__}(temperature={self.temperature})"


class TopK(PDistributionWarper):
    """
    Keeps the top k scores and sets the rest to -inf.
    """

    def __init__(self, k: int, on_logits: bool = True):
        self.k = k
        self.on_logits = on_logits

    def __call__(self, scores: torch.TensorType) -> torch.TensorType:
        # Get the top k values and indices along the last dimension
        top_k_values, top_k_indices = torch.topk(scores, self.k, dim=-1)

        if self.on_logits:
            # Create a tensor filled with -inf
            result = torch.full_like(scores, float("-inf"))
            # Set the top k values back
            result.scatter_(-1, top_k_indices, top_k_values)
        else:
            # Working with probabilities
            # Create a tensor filled with zeros
            result = torch.zeros_like(scores)
            # Set the top k values back
            result.scatter_(-1, top_k_indices, top_k_values)
            # Renormalize the probability distribution along the last dimension
            result = result / result.sum(dim=-1, keepdim=True)

        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(k={self.k})"


class ModelActorWrapper(OneStepActor):
    """Wraps SynthesisDecoderModel to get logits for next token."""

    def __init__(
        self,
        model: synthesis_decoder.SynthesisDecoderModel,
        token_action_handler: TokenActionHandler,
        scores_as_probs: bool = False,
    ):
        super().__init__(token_action_handler)
        if model.nested:
            raise ValueError("ModelActorWrapper only supports non-nested models")
        self.model = model
        self.scores_as_probs = scores_as_probs

    @torch.no_grad()
    def _get_actions_scores_values(
        self,
        current_sequences: dataset.CurrentSequences,
        conditioning: torch.TensorType | None = None,
        conditioning_mask: torch.TensorType | None = None,
    ) -> tuple[torch.TensorType, torch.TensorType, torch.TensorType | None]:
        out, logits = self.model(
            current_sequences,
            convert_logits_to_probs=self.scores_as_probs,
            conditioning=conditioning,
            conditioning_mask=conditioning_mask,
        )

        # The scores are the logits/probs of the next token.
        last_non_pad_token_pos = current_sequences.last_input_non_pad_pos  # [B]

        final_scores = torch.gather(
            out, dim=1, index=last_non_pad_token_pos[:, None, None].expand(-1, -1, out.shape[-1])
        )  # [B, 1, V]
        final_scores = final_scores.squeeze(dim=1)  # [B, V]

        # The actions are just indices of all tokens.
        actions = torch.arange(out.shape[-1], device=out.device)[None, :, None].expand(
            out.shape[0], -1, -1
        )  # [B, V, 1]

        # The values are the losses so far (note that this ignores pad tokens -- which we want!).
        values = self.model.compute_loss(current_sequences, logits, reduction="none")  # [B, S]
        values = values.sum(dim=1)  # [B]

        return actions, final_scores, values


def _mask_for_completions(i, current_sequences, stopping_criteria):
    if i == 0:
        return torch.zeros(
            len(current_sequences), device=current_sequences.device, dtype=torch.bool
        )
    eos_mask = current_sequences.get_sequences_over_mask(
        treat_eos_as_over=True, treat_pad_as_over=False
    )
    # ^ note that we should not treat pad as over, as the different sequences might have different lengths,
    # due to different reactions being run, and so may be padded, but not actually over.
    if stopping_criteria == StoppingCriteria.ONE_TOKEN:
        stopping_mask = torch.ones_like(eos_mask, dtype=torch.bool)
        # ^ after the first token all are then over.
    elif stopping_criteria == StoppingCriteria.COMPLETE_ACTION:
        last_token_not_from_model = torch.logical_not(
            torch.gather(
                current_sequences.input_frm_model_masks,
                dim=1,
                index=current_sequences.last_input_non_pad_pos[:, None],
            )
        )
        stopping_mask = torch.logical_or(eos_mask, last_token_not_from_model)
        # if the last token is not from the model, then it must be from the end of an action, and hence we
        # stop.
    elif stopping_criteria == StoppingCriteria.UNTIL_STOP:
        return eos_mask
    else:
        raise ValueError(f"Stopping criteria {stopping_criteria} not yet implemented")
    return stopping_mask


class SamplingInferenceStrategy(InferenceStrategy, OneStepActor):
    """Sampling-based inference strategy that samples tokens from the distribution. Note currently we sample with
    replacement=True, so we can sample the same token multiple times. (this is partly done as an implementation
    consideration, as communicating between the different equivalent samples would be a bit tricky).

    Scores here are treated like log probabilities and so are summed over sampling steps when creating final scores.
    The one step actor and warper should return something that looks like log_probs (i.e., sums to 1 when exponentiated).
    (We won't actually check that they sum to one so the onus is on the caller to actually set this up correctly!).
    """

    def __init__(
        self,
        token_action_handler: TokenActionHandler,
        one_step_actor: OneStepActor,
        warper: PDistributionWarper | None = None,
        stopping_criteria: StoppingCriteria = StoppingCriteria.ONE_TOKEN,
        number_samples_per_example: int = 1,
        max_num_steps: int = 100,
    ):
        """

        Keyword Arguments:
            warper -- Should convert scores from one_step_actor to log_probs (default: {None})
            stopping_criteria -- how long to run generation for before we hand back control to the caller
                (default: {StoppingCriteria.ONE_TOKEN})
            number_samples_per_example -- how many samples to take for each example, note that the different samples are
                done _with_ replacement, so we could sample the same sequence multiple times.  (default: {1})
            max_num_steps -- the max number of sampling steps to generate for before handling back control to the caller
                (assuming that the sequences have not already stopped for other reasons). (default: {100})
        """
        super().__init__(token_action_handler)
        self.one_step_actor = one_step_actor
        self.warper = warper
        self.stopping_criteria = stopping_criteria
        self.number_samples_per_example = number_samples_per_example
        self.max_num_steps = max_num_steps

    def generate(
        self,
        current_sequences: dataset.CurrentSequences,
        current_full_completions: BatchCompletedSequences,
        conditioning: torch.TensorType | None = None,
        conditioning_mask: torch.TensorType | None = None,
    ) -> tuple[dataset.CurrentSequences, BatchCompletedSequences]:
        return self._generate(
            current_sequences, current_full_completions, conditioning, conditioning_mask
        ), current_full_completions

    @torch.no_grad()
    def _generate(
        self,
        current_sequences: dataset.CurrentSequences,
        current_full_completions: BatchCompletedSequences | None = None,
        conditioning: torch.TensorType | None = None,
        conditioning_mask: torch.TensorType | None = None,
    ) -> dataset.CurrentSequences:
        # === 1. Expand out the current sequences to the number of samples per example. ===
        current_sequences = current_sequences.repeat(self.number_samples_per_example)

        # Also expand conditioning to match the repeated sequences
        # conditioning should be [B, conditioning_dim], expand to [B * number_samples_per_example, conditioning_dim]
        if conditioning is not None:
            conditioning = conditioning.repeat_interleave(self.number_samples_per_example, dim=0)
            conditioning_mask = conditioning_mask.repeat_interleave(
                self.number_samples_per_example, dim=0
            )

        # === 2. Create a mask for the completions: for the different stopping criteria, we work out which methods
        # are complete. ===
        def mask_for_completions(i, current_sequences):
            return _mask_for_completions(i, current_sequences, self.stopping_criteria)

        # === 3. Generate the completions, step by step. ===
        for i in range(self.max_num_steps):
            stopping_mask = mask_for_completions(i, current_sequences)  # [B]
            if stopping_mask.all():
                break
            completion_indices = torch.nonzero(~stopping_mask, as_tuple=True)[0]  # [B']
            actions, scores, values = self.one_step_actor.get_actions_scores_values(
                current_sequences, stopping_mask, conditioning, conditioning_mask
            )
            # ^ note the one step actor will return actions, scores, values wrt the non-stopping sequences.

            if self.warper is not None:
                scores = self.warper(scores)  # scores should be log probs

            # So while generating, we collect any sequences that we have predicted the score of ending and add them to
            # the completed sequences collection. While it's not strictly true that we have "sampled" these sequences,
            # we have assessed their score already, and so it does not seem to make sense to discard this information.
            if current_full_completions is not None:
                # add all the possible EOS choices to the completed sequences.
                last_actions = self.get_last_action(actions)  # [B, A]
                last_actions_are_eos = last_actions == vocab.EOS_VALUE
                for completed_i, completed_j in torch.nonzero(last_actions_are_eos, as_tuple=False):
                    # ^ note this currently means we also collect sequenced for which the score may be -inf. These can
                    # be filtered out later if desired. (We could change this, but given some might have been set to -inf),
                    # by the warper, think interesting to collect all of these at the moment.
                    completed_i_orig = completion_indices[completed_i].item()
                    orig_indx_ = current_sequences.example_indices[completed_i_orig].item()
                    full_seq_ = torch.cat(
                        (
                            current_sequences.input_sequences[completed_i_orig][
                                current_sequences.input_nonpad_masks[completed_i_orig]
                            ],
                            actions[completed_i, completed_j, :],
                        )
                    )
                    current_scores_ = (
                        current_sequences.scores[completed_i_orig].sum()
                        if current_sequences.scores is not None
                        else 0.0
                    )
                    score_ = (current_scores_ + scores[completed_i, completed_j]).item()
                    current_full_completions.add_single_completion(orig_indx_, full_seq_, score_)

            probs = torch.exp(scores)
            chosen_actions = torch.multinomial(probs, 1, replacement=True)  # [B, 1]
            current_sequences = self.one_step_actor.turn_chosen_actions_into_new_sequences(
                current_sequences, actions, scores, chosen_actions, stopping_mask
            )
            if current_full_completions is not None:
                current_full_completions.token_lib_collection = current_sequences.tkn_lib_collection
            # ^ this may have been updated.
        return current_sequences

    def _get_actions_scores_values(
        self,
        current_sequences: dataset.CurrentSequences,
        conditioning: torch.TensorType | None = None,
        conditioning_mask: torch.TensorType | None = None,
    ) -> tuple[torch.TensorType, torch.TensorType, torch.TensorType | None]:

        original_last_indices = current_sequences.last_input_non_pad_pos  # [B']
        original_last_scores = current_sequences.last_scores_non_pad_pos  # [B']
        original_batch_size = len(current_sequences)

        new_current_sequences = self._generate(
            current_sequences, conditioning=conditioning, conditioning_mask=conditioning_mask
        )
        new_last_indices_p1 = (
            new_current_sequences.last_input_non_pad_pos + 1
        )  # [B'' = B' * number_samples_per_example]
        # where number_samples_per_example is the number of possible actions for each sequence.
        new_last_scores_p1 = (
            new_current_sequences.last_scores_non_pad_pos + 1
        )  # [B'' = B' * number_samples_per_example]

        # Extract actions using the helper function
        # +1 because we want to start extracting AFTER the last original token (inclusive behavior)
        start_of_new_tokens = (original_last_indices + 1).repeat(self.number_samples_per_example)
        actions = utils.extract_new_tokens(
            start_of_new_tokens,
            new_last_indices_p1,
            new_current_sequences.input_sequences,
            vocab.PAD_VALUE,
        )
        actions = actions.view(
            original_batch_size, self.number_samples_per_example, -1
        )  # [B', A, max_action_length]

        start_of_new_scores = (original_last_scores + 1).repeat(self.number_samples_per_example)
        scores = utils.extract_new_tokens(
            start_of_new_scores, new_last_scores_p1, new_current_sequences.input_sequences, 0.0
        )
        scores = scores.view(
            original_batch_size, self.number_samples_per_example, -1
        )  # [B', A, max_seqlength]
        scores = scores.sum(dim=-1)  # [B', A]

        values = current_sequences.scores.sum(dim=-1)  # [B']
        return actions, scores, values


class BeamSearchInferenceStrategy(InferenceStrategy):
    """Beam search inference strategy.

    **Note** that if you use this method on an existing expanded current_sequences (i.e.,
    with duplicate example_indices), then this method will create new beams for each of the original examples, but
    will aggregate across all the different examples when consolidating the beams at the end -- this might not be the
    behaviour you want.

    Scores here are treated like log probabilities and so are summed over sampling steps when creating final scores.
    The one step actor and warper should return something that looks like log_probs (i.e., sums to 1 when exponentiated).
    (We won't actually check that they sum to one so the onus is on the caller to actually set this up correctly!).
    """

    def __init__(
        self,
        token_action_handler: TokenActionHandler,
        one_step_actor: OneStepActor,
        warper: PDistributionWarper | None = None,
        stopping_criteria: StoppingCriteria = StoppingCriteria.UNTIL_STOP,
        beam_width: int = 1,
        max_num_steps: int = 100,
    ):
        super().__init__(token_action_handler)
        self.one_step_actor = one_step_actor
        self.warper = warper
        self.stopping_criteria = stopping_criteria
        self.beam_width = beam_width
        self.max_num_steps = max_num_steps

    def generate(
        self,
        current_sequences: dataset.CurrentSequences,
        current_full_completions: BatchCompletedSequences,
        conditioning: torch.TensorType | None = None,
        conditioning_mask: torch.TensorType | None = None,
    ) -> tuple[dataset.CurrentSequences, BatchCompletedSequences]:
        return self._generate(
            current_sequences, current_full_completions, conditioning, conditioning_mask
        ), current_full_completions

    @torch.no_grad()
    def _generate(
        self,
        current_sequences: dataset.CurrentSequences,
        current_full_completions: BatchCompletedSequences | None = None,
        conditioning: torch.TensorType | None = None,
        conditioning_mask: torch.TensorType | None = None,
    ) -> dataset.CurrentSequences:

        # === 0. Create a tensor to record how many beams have been "completed" for each example. ===
        num_examples_p1 = current_sequences.example_indices.max() + 1  # B + 1
        # ^ [ B := num original sequences ]

        # === 1. Create a mask for the completions: for the different stopping criteria, we work out which methods
        # are complete. ===
        def mask_for_completions(i, current_sequences):
            return _mask_for_completions(i, current_sequences, self.stopping_criteria)

        # === 2. Generate the completions, step by step. ===
        for i in range(self.max_num_steps):
            stopping_mask = mask_for_completions(i, current_sequences)  # [B'' ]
            if stopping_mask.all():
                break

            num_beams_remaining = self.beam_width - torch.bincount(
                current_sequences.example_indices[stopping_mask], minlength=num_examples_p1
            )
            num_beams_remaining = num_beams_remaining.clamp(min=0)

            completion_indices = torch.nonzero(~stopping_mask, as_tuple=True)[0]  # [B']
            actions, scores, values = self.one_step_actor.get_actions_scores_values(
                current_sequences, stopping_mask, conditioning, conditioning_mask
            )
            # ^ note the one step actor will return actions, scores, values wrt the non-stopping sequences.
            num_actions = actions.shape[1]

            if self.warper is not None:
                scores = self.warper(scores)  # scores should be log probs

            # So while generating, we collect any sequences that we have predicted the score of ending and add them to
            # the completed sequences collection. While it's not strictly true that we have "sampled" these sequences,
            # we have assessed their score already, and so it does not seem to make sense to discard this information.
            if current_full_completions is not None:
                # add all the possible EOS choices to the completed sequences.
                last_actions = OneStepActor.get_last_action(actions)  # [B', A]
                last_actions_are_eos = last_actions == vocab.EOS_VALUE
                for completed_i, completed_j in torch.nonzero(last_actions_are_eos, as_tuple=False):
                    completed_i_orig = completion_indices[completed_i].item()
                    orig_indx_ = current_sequences.example_indices[completed_i_orig].item()
                    full_seq_ = torch.cat(
                        (
                            current_sequences.input_sequences[completed_i_orig][
                                current_sequences.input_nonpad_masks[completed_i_orig]
                            ],
                            actions[completed_i, completed_j, :],
                        )
                    )
                    current_scores_ = (
                        current_sequences.scores[completed_i_orig].sum()
                        if current_sequences.scores is not None
                        else 0.0
                    )
                    score_ = (current_scores_ + scores[completed_i, completed_j]).item()
                    current_full_completions.add_single_completion(orig_indx_, full_seq_, score_)

            # for beam search the "score" of a current sequence is its total log likelihood, that is the sum of the
            # log likelihoods of the prefix (i.e., actions taken so far), with the log likelihood of the next potential
            # action.
            # note we may carry forward beams with -inf scores (if there are not enough plausible actions).
            if current_sequences.scores is not None:  # i.e., on first step scores will be 0/None
                full_seq_scores = (
                    current_sequences.scores[completion_indices].sum(dim=-1, keepdim=True) + scores
                )  # [B', A]
            else:
                full_seq_scores = scores

            # we will now go through and get the chosen beam actions for each of the examples that have not yet stopped.
            max_num_beams_remaining = num_beams_remaining.max()
            chosen_actions = torch.full(
                (len(completion_indices), max_num_beams_remaining),
                fill_value=-1,
                device=current_sequences.device,
                dtype=torch.int64,
            )
            # resorting to a for loop here -- if slow can pick up the vectorized version I was previously working with:
            for example_idx in torch.unique(current_sequences.example_indices[completion_indices]):
                # get the indices of the prefix sequences that correspond to this set of beams:
                example_idx_mask_wrt_continuing_seqs = (
                    current_sequences.example_indices[completion_indices] == example_idx
                )
                num_rows_for_example = example_idx_mask_wrt_continuing_seqs.sum()

                # then get the top-k prefix-action pairs to take forward as a continuation of this beam
                remaining_beam_size = min(
                    num_beams_remaining[example_idx], num_actions * num_rows_for_example
                )
                # note to do this we have to flatten the scores, and then unflatten the indices after to get back to their
                # column and row indices.
                if remaining_beam_size == 0:
                    continue  # note no actions to chose for this beam!
                topk_for_example = torch.topk(
                    full_seq_scores[example_idx_mask_wrt_continuing_seqs].view(-1),
                    k=remaining_beam_size,
                    dim=0,
                )
                # we  remove examples with scores of -inf from the topk -- these could be invalid,
                # eg masked out actions and so not worth persuing.
                non_inf_top_k = topk_for_example.values != float("-inf")
                if non_inf_top_k.sum() == 0:
                    raise ValueError("No non-inf token choices to explore for this example!")
                    # ^ at the moment there should always be at least one non-inf token available, but check this!
                topk_arg_places_for_example = topk_for_example.indices[non_inf_top_k]
                topk_arg_places_for_example_i = topk_arg_places_for_example // num_actions
                topk_arg_places_for_example_j = topk_arg_places_for_example % num_actions

                # we now will form the respective part of the chosen action tensor, noting that if no chosen actions
                # are taken for that current sequence, then the row will be all -1s.
                order_wrt_i = torch.argsort(topk_arg_places_for_example_i, dim=0)
                full_lengths = torch.bincount(
                    topk_arg_places_for_example_i, minlength=num_rows_for_example
                )
                chosen_actions_jagged = torch.nested.nested_tensor_from_jagged(
                    topk_arg_places_for_example_j[order_wrt_i],
                    lengths=full_lengths,
                    max_seqlen=remaining_beam_size,
                )
                chosen_actions_for_example_idx = torch.nested.to_padded_tensor(
                    chosen_actions_jagged,
                    padding=-1,
                    output_size=(len(full_lengths), max_num_beams_remaining),
                )

                assert chosen_actions_for_example_idx.shape == (
                    num_rows_for_example,
                    max_num_beams_remaining,
                )

                # finally we can add these back into the chosen action tensor
                chosen_actions[example_idx_mask_wrt_continuing_seqs] = (
                    chosen_actions_for_example_idx
                )

            current_sequences = self.one_step_actor.turn_chosen_actions_into_new_sequences(
                current_sequences, actions, scores, chosen_actions, stopping_mask
            )

            if current_full_completions is not None:
                current_full_completions.token_lib_collection = current_sequences.tkn_lib_collection
            # ^ this may have been updated.
        return current_sequences
