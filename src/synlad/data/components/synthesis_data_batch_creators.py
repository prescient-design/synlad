"""Batch creators that pad and tensorize synthesis-pathway inputs for training."""

import functools

import torch
from torch_geometric import data as pyg_data

from synlad.data.components import synthesis_molecular_graphs as molecular_graphs
from synlad.tokenization import synthesis_serialization as serialization
from synlad.tokenization import synthesis_vocab as vocab
from synlad.utils import synthesis_torch_settings as torch_settings
from synlad.utils import synthesis_utils as utils


class PaddedInputMolScores:
    """Converts input molecular scores to padded tensor format. None molecules are given a score of 0."""

    def __init__(
        self,
        tkn_lib_collection: vocab.TokenLibraryCollection,
        input_sequences_as_value_list: list[list[int]] = None,
        max_seq_len: int = None,
    ):
        self.tkn_lib_collection = tkn_lib_collection
        self.input_sequences_as_value_list = input_sequences_as_value_list
        self.max_seq_len = max_seq_len

    @functools.singledispatchmethod
    def create_padded_batch(self, input_mol_scores) -> torch.Tensor:
        raise NotImplementedError("Not implemented for this type")

    @create_padded_batch.register
    def _(self, input_mol_scores: torch.Tensor) -> torch.Tensor:
        """input_mol_scores: already padded tensor of shape [B, S] containing molecular scores"""
        if input_mol_scores.is_nested:
            raise ValueError("Nested tensors not supported, expected padded tensor")
        return input_mol_scores

    @create_padded_batch.register
    def _(self, input_mol_scores: list) -> torch.Tensor:
        """input_mol_scores: list of either score lists or callable functions (uses if statements to distinguish as
        beyond functools.singledispatch)

        List[List[float]] -- list of float scores for each sequence
        List[Callable] -- list of callable functions that score molecular tokens for each sequence.
        List[Tensors] -- list of tensors to pad and stack
        """
        if len(input_mol_scores) == 0:
            raise ValueError("Cannot process empty list")

        # Check if it's a list of lists (scores) or list of callables or list of tensors
        if callable(input_mol_scores[0]):
            if self.input_sequences_as_value_list is None:
                raise ValueError(
                    "input_sequences_as_value_list required in constructor when input_mol_scores is a list of callables"
                )

            all_mol_scores = []
            for seq, mol_scorer in zip(
                self.input_sequences_as_value_list, input_mol_scores, strict=False
            ):
                scores = torch.zeros(len(seq), dtype=torch_settings.TORCH_FLT_TYPE)
                for j, token_idx in enumerate(seq):
                    if token_idx >= self.tkn_lib_collection.start_of_molecular_indices:
                        token_str = self.tkn_lib_collection.token_frm_idx(token_idx)
                        scores[j] = mol_scorer(token_str)
                all_mol_scores.append(scores)
            return self.create_padded_batch(all_mol_scores)
        elif isinstance(input_mol_scores[0], list):
            # input_mol_scores: list of lists where each inner list contains float scores for one sequence
            return self.create_padded_batch(
                [
                    torch.tensor(scores, dtype=torch_settings.TORCH_FLT_TYPE)
                    for scores in input_mol_scores
                ]
            )
        elif isinstance(input_mol_scores[0], torch.Tensor):
            # input_mol_scores: list of tensors to pad and stack
            return utils.pad_then_stack(input_mol_scores, 0.0, padded_final_length=self.max_seq_len)
        else:
            raise ValueError(f"Unsupported input type: {type(input_mol_scores[0])}")


class PaddedGraphs:
    """Handles graph generation from token library collection."""

    def __init__(
        self,
        tkn_lib_collection: vocab.TokenLibraryCollection,
        bb_graphs: list[pyg_data.Data] | None = None,
    ):
        self.tkn_lib_collection = tkn_lib_collection
        self.bb_graphs = bb_graphs

    @functools.singledispatchmethod
    def create_batch(self, graphs) -> pyg_data.Batch:
        raise NotImplementedError("Not implemented for this type")

    @create_batch.register
    def _(self, graphs: list) -> pyg_data.Batch:
        # list of pyg_data.Data
        return molecular_graphs.from_data_list(graphs)

    @create_batch.register
    def _(self, graphs: pyg_data.Batch) -> pyg_data.Batch:
        if graphs.batch_size < self._num_graphs:
            raise ValueError(
                f"Graph batch size {graphs.batch_size} is less than the number of graphs {self._num_graphs}"
            )
        return graphs
        # ^ we could recompute this, but likely some of them would be reused, so we leave this to the user.

    @create_batch.register
    def _(self, graphs: type(None)) -> pyg_data.Batch:
        bb_graphs = (
            self.bb_graphs
            if self.bb_graphs is not None
            else [
                molecular_graphs.from_smiles(smi)
                for smi in self.tkn_lib_collection.bb_tkn_library.tokens
            ]
        )
        return self.create_batch(
            bb_graphs
            + [
                molecular_graphs.from_smiles(smi)
                for smi in self.tkn_lib_collection.mol_tkn_library.tokens
            ]
        )

    @property
    def _num_graphs(self):
        return len(self.tkn_lib_collection.bb_tkn_library) + len(
            self.tkn_lib_collection.mol_tkn_library
        )


class PaddedInputPredictiveNxtTknMasks:
    """Converts input predictive next token masks to padded tensor format."""

    def __init__(
        self,
        tkn_lib_collection: vocab.TokenLibraryCollection,
        input_sequences_as_value_list: list[list[int]] = None,
        max_seq_len: int = None,
    ):
        self.tkn_lib_collection = tkn_lib_collection
        self.input_sequences_as_value_list = input_sequences_as_value_list
        self.max_seq_len = max_seq_len

    @functools.singledispatchmethod
    def create_padded_batch(self, input_predictive_nxt_tkn_masks) -> torch.Tensor:
        raise NotImplementedError("Not implemented for this type")

    @create_padded_batch.register
    def _(self, input_predictive_nxt_tkn_masks: torch.Tensor) -> torch.Tensor:
        """input_predictive_nxt_tkn_masks: already padded tensor of shape [B, S, V] containing next token prediction masks"""
        if input_predictive_nxt_tkn_masks.is_nested:
            raise ValueError("Nested tensors not supported, expected padded tensor")
        return input_predictive_nxt_tkn_masks

    @create_padded_batch.register
    def _(self, input_predictive_nxt_tkn_masks: list) -> torch.Tensor:  # list of np.array
        """input_predictive_nxt_tkn_masks: list of numpy arrays where each array contains prediction masks for one sequence"""
        all_masks = [
            torch.tensor(mask, dtype=torch.bool) for mask in input_predictive_nxt_tkn_masks
        ]
        return utils.pad_then_stack(
            all_masks, False, pad_dim=0, padded_final_length=self.max_seq_len
        )

    @create_padded_batch.register
    def _(self, input_predictive_nxt_tkn_masks: type(None)) -> torch.Tensor:
        """input_predictive_nxt_tkn_masks: None, indicating masks should be generated from sequences"""
        if self.input_sequences_as_value_list is None:
            raise ValueError(
                "input_sequences_as_value_list required in constructor when input_predictive_nxt_tkn_masks is None"
            )

        all_predictive_tkn_masks = []
        for seq in self.input_sequences_as_value_list:
            seq_masks = []
            for pos in range(len(seq)):
                mask = serialization.Serializer.get_mask(seq[: pos + 1], self.tkn_lib_collection)
                seq_masks.append(torch.tensor(mask, dtype=torch.bool))
            all_predictive_tkn_masks.append(torch.stack(seq_masks))

        return utils.pad_then_stack(
            all_predictive_tkn_masks, False, pad_dim=0, padded_final_length=self.max_seq_len
        )


class PaddedInputFrmModelMasks:
    """Converts input from-model masks to padded tensor format."""

    def __init__(
        self,
        tkn_lib_collection: vocab.TokenLibraryCollection,
        input_sequences_as_value_list: list[list[int]] = None,
        max_seq_len: int = None,
    ):
        self.tkn_lib_collection = tkn_lib_collection
        self.input_sequences_as_value_list = input_sequences_as_value_list
        self.max_seq_len = max_seq_len

    @functools.singledispatchmethod
    def create_padded_batch(self, input_frm_model_masks) -> torch.Tensor:
        raise NotImplementedError("Not implemented for this type")

    @create_padded_batch.register
    def _(self, input_frm_model_masks: torch.Tensor) -> torch.Tensor:
        """input_frm_model_masks: already padded tensor of shape [B, S] containing from-model masks"""
        if input_frm_model_masks.is_nested:
            raise ValueError("Nested tensors not supported, expected padded tensor")
        return input_frm_model_masks

    @create_padded_batch.register
    def _(self, input_frm_model_masks: list) -> torch.Tensor:
        """input_frm_model_masks: list of boolean sequences indicating which tokens come from model"""
        all_masks = [torch.tensor(mask, dtype=torch.bool) for mask in input_frm_model_masks]
        return utils.pad_then_stack(all_masks, False, padded_final_length=self.max_seq_len)

    @create_padded_batch.register
    def _(self, input_frm_model_masks: type(None)) -> torch.Tensor:
        """input_frm_model_masks: None, indicating masks should be generated from sequences"""
        if self.input_sequences_as_value_list is None:
            raise ValueError(
                "input_sequences_as_value_list required in constructor when input_frm_model_masks is None"
            )

        all_masks = []
        for seq in self.input_sequences_as_value_list:
            mask = torch.tensor(
                serialization.Serializer.get_if_from_model(seq, self.tkn_lib_collection),
                dtype=torch.bool,
            )
            all_masks.append(mask)

        return utils.pad_then_stack(all_masks, False, padded_final_length=self.max_seq_len)


class PaddedInputMolMasks:
    """Converts input molecular masks to padded tensor format."""

    def __init__(
        self, input_sequences_tensor: torch.Tensor, tkn_lib_collection: vocab.TokenLibraryCollection
    ):
        self.input_sequences_tensor = input_sequences_tensor
        self.tkn_lib_collection = tkn_lib_collection

    @functools.singledispatchmethod
    def create_padded_batch(self, input_mol_masks) -> torch.Tensor:
        raise NotImplementedError("Not implemented for this type")

    @create_padded_batch.register
    def _(self, input_mol_masks: torch.Tensor) -> torch.Tensor:
        """input_mol_masks: already padded tensor of shape [B, S] containing molecular masks"""
        if input_mol_masks.is_nested:
            raise ValueError("Nested tensors not supported, expected padded tensor")
        return input_mol_masks

    @create_padded_batch.register
    def _(self, input_mol_masks: list) -> torch.Tensor:
        """input_mol_masks: list of boolean sequences indicating which tokens are molecular"""
        return utils.pad_then_stack(
            [torch.tensor(mask, dtype=torch.bool) for mask in input_mol_masks],
            False,
            padded_final_length=self.input_sequences_tensor.shape[1],
        )

    @create_padded_batch.register
    def _(self, input_mol_masks: type(None)) -> torch.Tensor:
        """input_mol_masks: None, indicating masks should be generated from input sequences"""
        return self.input_sequences_tensor >= self.tkn_lib_collection.start_of_molecular_indices


class PaddedInputNonpadMasks:
    """Converts input non-padding masks to padded tensor format."""

    def __init__(self, input_sequences_tensor: torch.Tensor):
        self.input_sequences_tensor = input_sequences_tensor

    @functools.singledispatchmethod
    def create_padded_batch(self, input_nonpad_masks) -> torch.Tensor:
        raise NotImplementedError("Not implemented for this type")

    @create_padded_batch.register
    def _(self, input_nonpad_masks: torch.Tensor) -> torch.Tensor:
        """input_nonpad_masks: already padded tensor of shape [B, S] containing non-padding masks"""
        if input_nonpad_masks.is_nested:
            raise ValueError("Nested tensors not supported, expected padded tensor")
        return input_nonpad_masks

    @create_padded_batch.register
    def _(self, input_nonpad_masks: list) -> torch.Tensor:
        """input_nonpad_masks: list of boolean sequences indicating which positions are not padding"""
        return self.input_sequences_tensor != vocab.PAD_VALUE
        # ^ we actually ignore input, so we dont need to deal with padding etc.

    @create_padded_batch.register
    def _(self, input_nonpad_masks: type(None)) -> torch.Tensor:
        """input_nonpad_masks: None, indicating masks should be generated from input sequences"""
        return self.input_sequences_tensor != vocab.PAD_VALUE
