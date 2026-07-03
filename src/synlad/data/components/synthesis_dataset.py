"""Dataset and collator for synthesis-pathway generation tasks."""

import copy
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
from numpy import random as np_random
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch.utils import data
from torch_geometric import data as pyg_data

from synlad.data.components import synthesis_data_batch_creators as data_batch_creators
from synlad.data.components import synthesis_molecular_graphs as molecular_graphs
from synlad.data.components import synthesis_reaction_network_graph as reaction_network_graph
from synlad.tokenization import synthesis_serialization as serialization
from synlad.tokenization import synthesis_vocab as vocab
from synlad.utils import synthesis_torch_settings as torch_settings
from synlad.utils import synthesis_utils as utils


@dataclass
class RxnNetTask:
    """A single training/evaluation example for synthesis-pathway generation.

    Bundles the reaction-network context, a prompt token sequence, a scoring
    function used to grade candidate target molecules, and any precomputed 3D
    conformations.
    """

    rxn_net: reaction_network_graph.ReactionNetwork
    prompt: list[str]
    mol_scorer: Callable[
        [str], float
    ]  # smiles string to score on how well it achieves the objective.
    molecule_conformations: list | None = (
        None  # List of OEMol objects with 3D conformations, or None if not generated
    )

    def has_conformations(self) -> bool:
        """Check if this task has generated conformations."""
        return self.molecule_conformations is not None and len(self.molecule_conformations) > 0

    def get_conformations(self) -> list | None:
        """Get the list of molecular conformations, or None if not available."""
        return self.molecule_conformations

    def get_num_conformations(self) -> int:
        """Get the total number of conformations available across all molecules."""
        if not self.molecule_conformations:
            return 0
        # Sum up conformers from all RDKit molecules
        total_conformers = 0
        for mol in self.molecule_conformations or []:
            if hasattr(mol, "GetNumConformers"):  # RDKit Mol
                total_conformers += mol.GetNumConformers()
        return total_conformers

    def get_conformer_positions(self) -> list | None:
        """Get 3D positions for all conformers as a list of numpy arrays."""
        if not self.has_conformations():
            return None

        all_positions = []
        for mol in self.molecule_conformations or []:
            # Handle RDKit molecules
            if hasattr(mol, "GetConformers"):
                for conf in mol.GetConformers():
                    all_positions.append(conf.GetPositions())

        return all_positions

    def get_individual_conformers(self) -> list | None:
        """Get individual conformers as separate RDKit molecules (one conformer each)."""
        if not self.has_conformations() or Chem is None:
            return None

        individual_conformers = []
        for mol in self.molecule_conformations or []:
            if hasattr(mol, "GetConformers"):  # RDKit Mol
                for conf in mol.GetConformers():
                    # Create a new molecule with just this conformer
                    new_mol = Chem.Mol(mol)
                    new_mol.RemoveAllConformers()
                    new_mol.AddConformer(conf, assignId=True)
                    individual_conformers.append(new_mol)

        return individual_conformers


class RxnNetTaskDataset(data.Dataset):
    """Dataset around a reaction network.

    Thin wrapper around a list. (the heavy lifting of getting ready for PyTorch is done in the collate function.).
    """

    def __init__(self, data_list: list[RxnNetTask]):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


@dataclass
class RxnNetTaskBatch:
    """Batched form of :class:`RxnNetTask` consumed by the synthesis decoder.

    Holds the padded (or nested) input/output token tensors plus the per-token
    masks the model needs to distinguish molecule tokens from special tokens,
    model-generated positions from oracle positions, and real tokens from
    padding. Tensors share the same ``B`` (batch) and ``S`` (sequence-length)
    leading dimensions.
    """

    # at moment we pad the tensors to fit together, but could consider using nested tensors instead.

    # Sizes: B batch size, S sequence length (-1 of actual length when include EOS), V vocabulary size.
    tkn_lib_collection: vocab.TokenLibraryCollection  # V tokens
    graphs: pyg_data.Batch  # represents V - (# special tokens) graphs

    input_sequences: torch.TensorType  # [B, S], the tokenzied sequences. in integer form
    input_mol_scores: (
        torch.TensorType
    )  # [B, S], the scores for the molecules in the sequences. non-molecule tokens scores
    # undefined i.e. any number

    input_predictive_nxt_tkn_masks: (
        torch.TensorType
    )  # [B, S, V], for each token the next tokens that it can predict.
    # should be coerced with input sequences (if nested)

    input_frm_model_masks: (
        torch.TensorType
    )  # [B, S], for each input in sequence whether it comes from model
    # the input_frm_model_masks masks are wrt to the sequence inputs (i.e., have not yet been left-shifted).
    # i.e., they indicate whether the input has come from the model or an outside "oracle".
    input_mol_masks: torch.TensorType  # [B, S] True for molecular tokens, False for special tokens.
    input_nonpad_masks: (
        torch.TensorType
    )  # [B, S] True for non-padding tokens, False for padding tokens.

    # Below are the optional arguments

    sequence_indcs: torch.TensorType | None = (
        None  # [B, S], the indices of the sequences in the batch.
    )
    # should be coerced with input sequences (if nested); should be [[0,1,2,3,4,...], [0,1,2,3,4,...], ...]

    # at moment these following variables are prefilled but probably could be generated from the equivalent inputs
    # using narrow for jagged tensors.
    output_sequences: torch.TensorType | None = (
        None  # [B, S], the output sequences. in integer form
    )
    output_from_model_masks: torch.TensorType | None = (
        None  # [B, S], for each output in sequence whether it comes
    )
    # from model and should be precalculated

    # Conditioning vector
    conditioning: torch.TensorType | None = (
        None  # [B, conditioning_dim], latent conditioning vectors
    )

    # Target molecules for retro tasks
    target_molecules: list[str | None] | None = (
        None  # [B], target molecule SMILES for retro tasks, None for non-retro tasks
    )

    rxn_nets: list[reaction_network_graph.ReactionNetwork] | None = (
        None  # [B] the original reaction networks. (optional)
    )

    NON_NESTED_TENSOR_PROPS = {
        "tkn_lib_collection",
        "graphs",
        "conditioning",
        "target_molecules",
        "rxn_nets",
    }

    def __post_init__(self):
        if self.sequence_indcs is None:
            if self.input_sequences.is_nested:
                max_seq_len = max(o.shape[0] for o in self.input_sequences)
                all_sequence_indcs = torch.arange(
                    max_seq_len,
                    dtype=torch_settings.TORCH_INT_TYPE,
                    device=self.input_sequences.device,
                )
                all_sequence_indcs = coerce_offsets(
                    torch.nested.nested_tensor(
                        [all_sequence_indcs[: r.shape[0]] for r in self.input_sequences],
                        layout=torch.jagged,
                    ),
                    self.input_sequences,
                )
            else:
                all_sequence_indcs = torch.arange(
                    self.input_sequences.shape[1],
                    dtype=torch_settings.TORCH_INT_TYPE,
                    device=self.input_sequences.device,
                )
                all_sequence_indcs = all_sequence_indcs[None, :].repeat(
                    self.input_sequences.shape[0], 1
                )

            if self.input_sequences.is_nested:
                self.input_predictive_nxt_tkn_masks = coerce_offsets(
                    self.input_predictive_nxt_tkn_masks, self.input_sequences
                )

            self.sequence_indcs = all_sequence_indcs

    @classmethod
    def from_tokens(
        cls,
        # As a minimum, we need some version of the below properties:
        input_sequences: list[list[int | str]] | torch.Tensor,
        tkn_lib_collection: vocab.TokenLibraryCollection,
        input_mol_scores: list[list[float]] | torch.Tensor | list[Callable[[str], float]],
        # The below can be generated from the above inputs (and code elsewhere), if required.
        # (note that the below are not required to be provided, but if they are, then they will be used
        #  to create the batch to save time, and we will not check that they are consistent).
        graphs: pyg_data.Batch | None = None,
        bb_graph_list: list[pyg_data.Data] | None = None,
        input_predictive_nxt_tkn_masks: list[np.array] | torch.Tensor | None = None,
        input_frm_model_masks: list[np.array] | torch.Tensor | None = None,
        input_mol_masks: list[np.array] | torch.Tensor | None = None,
        input_nonpad_masks: list[np.array] | torch.Tensor | None = None,
        sequence_indcs: torch.Tensor | None = None,
        output_sequences: torch.Tensor | None = None,
        output_from_model_masks: torch.Tensor | None = None,
    ):
        """Creates a RxnNetTaskBatch from a selection of the given inputs. (Only creates padded version at moment.)
        Infers what type of object the different inputs are from their first element -- so be consistent!

        Arguments:
            input_sequences -- the serialized input sequences. Can be list of lists (of tokens in str or integer value form).
            tkn_lib_collection -- token library collection, containing relevant vocab list. Note will add molecules to
                end if not already present (does this on a copy of the collection so can continue using.)
            input_mol_scores -- list of list of floats of token scores, or a tensor of scores (padded), or a list
             of functions which we can call on each molecule string to get the score (note non molecular tokens will be
             given a score of 0).

        Keyword Arguments (these will be generated separately if not provided):
            graphs -- Pytorch Geometric Batch of graphs. If None will create from the graphs in the tkn_lib_collection.
            bb_graph_list -- list of pytorch geometric data objects for the building block graphs. If None will create from the
                graphs in the tkn_lib_collection if needed. (ignored and not made if graphs are provided)
            input_predictive_nxt_tkn_masks -- List of masks, tensor of masks (already padded), or None and will be generated.
            input_frm_model_masks -- List of masks, tensor of masks (already padded), or None and will be generated.
            input_mol_masks -- List of masks, tensor of masks (already padded), or None and will be generated.
            input_nonpad_masks -- List of masks, tensor of masks (already padded), or None and will be generated.
            sequence_indcs -- pre-computed tensor of indices or None and will get generated (default: {None})
            output_sequences -- precomputed tensor of output sequences or None and will get generated (default: {None})
            output_from_model_masks -- precomputed tensor of output from model masks or None and will get generated (default: {None})
        """

        # === 0. Input validation ===
        if (output_sequences is None) ^ (output_from_model_masks is None):
            raise ValueError(
                "output_sequences and output_from_model_masks must both be provided or both be None"
            )
        tkn_lib_collection = copy.copy(tkn_lib_collection)

        # === 1. Convert input_sequences to padded tensor, and record particular features ===
        if isinstance(input_sequences, torch.Tensor):
            if input_sequences.is_nested:
                raise ValueError("Nested tensors not currently supported, expected padded tensor")
            # Already a padded tensor
        elif isinstance(input_sequences, list):
            # Convert list of lists to list of tensors
            input_seqs_as_tensor_lists = []
            for seq in input_sequences:
                if isinstance(seq[0], str):
                    # Convert string tokens to integer indices
                    int_seq = [
                        tkn_lib_collection.idx_frm_token(token, fail_on_not_found=False)
                        for token in seq
                    ]
                    input_seqs_as_tensor_lists.append(
                        torch.tensor(int_seq, dtype=torch_settings.TORCH_INT_TYPE)
                    )
                else:
                    # Already integer tokens
                    input_seqs_as_tensor_lists.append(
                        torch.tensor(seq, dtype=torch_settings.TORCH_INT_TYPE)
                    )
            input_sequences = utils.pad_then_stack(input_seqs_as_tensor_lists, vocab.PAD_VALUE)
        else:
            raise ValueError(f"Unsupported input_sequences type: {type(input_sequences)}")

        # === 2. Create most of the other padded tensors (as needed) ===

        # Get dimensions and helper functions from input sequences for creating the other padded tensors
        max_seq_len = input_sequences.shape[1]
        _input_seqs_as_value_list = [
            row[row != vocab.PAD_VALUE].detach().cpu().numpy().tolist() for row in input_sequences
        ]

        # go through and create them
        input_mol_scores = data_batch_creators.PaddedInputMolScores(
            tkn_lib_collection, _input_seqs_as_value_list, max_seq_len
        ).create_padded_batch(input_mol_scores)
        input_predictive_nxt_tkn_masks = data_batch_creators.PaddedInputPredictiveNxtTknMasks(
            tkn_lib_collection, _input_seqs_as_value_list, max_seq_len
        ).create_padded_batch(input_predictive_nxt_tkn_masks)
        input_frm_model_masks = data_batch_creators.PaddedInputFrmModelMasks(
            tkn_lib_collection, _input_seqs_as_value_list, max_seq_len
        ).create_padded_batch(input_frm_model_masks)
        input_mol_masks = data_batch_creators.PaddedInputMolMasks(
            input_sequences, tkn_lib_collection
        ).create_padded_batch(input_mol_masks)
        input_nonpad_masks = data_batch_creators.PaddedInputNonpadMasks(
            input_sequences
        ).create_padded_batch(input_nonpad_masks)
        graphs = data_batch_creators.PaddedGraphs(tkn_lib_collection, bb_graph_list).create_batch(
            graphs
        )
        graphs = graphs.to(input_sequences.device)

        # === 3. Create the batch (and sequence indices/output sequences) ===
        out = cls(
            tkn_lib_collection=tkn_lib_collection,
            graphs=graphs,
            input_sequences=input_sequences,
            input_mol_scores=input_mol_scores,
            input_predictive_nxt_tkn_masks=input_predictive_nxt_tkn_masks,
            input_frm_model_masks=input_frm_model_masks,
            input_mol_masks=input_mol_masks,
            input_nonpad_masks=input_nonpad_masks,
            # The below can still be None at this point:
            sequence_indcs=sequence_indcs,  # <-- added by __post_init__
            output_sequences=output_sequences,  # <--- added next if necessary
            output_from_model_masks=output_from_model_masks,  # <--- added next if necessary
        )
        if out.output_sequences is None:
            out.add_output_sequences()
        return out

    def select_submask(self, submask: torch.TensorType) -> "RxnNetTaskBatch":
        # at moment this does not adjust tkn_lib_collection or graphs...
        if self.is_nested:
            raise NotImplementedError("select_submask is not supported for nested tensors")

        # Subselect the reaction networks if they exist. (this is a regular list, so do via list comprehension.)
        if self.rxn_nets is not None:
            rxn_nets = [self.rxn_nets[i] for i in submask if bool(i)]
        else:
            rxn_nets = None

        return RxnNetTaskBatch(
            tkn_lib_collection=self.tkn_lib_collection,
            graphs=self.graphs,
            input_sequences=self.input_sequences[submask],
            input_mol_scores=self.input_mol_scores[submask],
            input_predictive_nxt_tkn_masks=self.input_predictive_nxt_tkn_masks[submask],
            input_frm_model_masks=self.input_frm_model_masks[submask],
            input_mol_masks=self.input_mol_masks[submask],
            input_nonpad_masks=self.input_nonpad_masks[submask],
            sequence_indcs=self.sequence_indcs[submask]
            if self.sequence_indcs is not None
            else None,
            output_sequences=self.output_sequences[submask]
            if self.output_sequences is not None
            else None,
            output_from_model_masks=self.output_from_model_masks[submask]
            if self.output_from_model_masks is not None
            else None,
            rxn_nets=rxn_nets,
        )

    def add_output_sequences(self, exist_okay=False):
        if not exist_okay:
            assert self.output_sequences is None, "output sequences already exist!"
            assert self.output_from_model_masks is None, "output from model masks already exist!"

        output_sequences = torch.full_like(self.input_sequences, fill_value=vocab.PAD_VALUE)
        output_sequences[:, :-1] = self.input_sequences[:, 1:]
        self.output_sequences = output_sequences

        output_from_model_masks = torch.full_like(self.input_frm_model_masks, fill_value=False)
        output_from_model_masks[:, :-1] = self.input_frm_model_masks[:, 1:]
        self.output_from_model_masks = output_from_model_masks

    def repeat(self, repeat_sizes: torch.TensorType) -> "RxnNetTaskBatch":
        """
        Repeats the batch for each element in repeat_sizes.

        Note that repeat_sizes of 0 should work, as this is supported (but not documented?) by repeat_interleave.
        However, we don't reduce the graphs or token library collection when doing so, so the class may be bigger
        than it needs to be. Also, advise you to be cautious if relying on this, as it may change in the future depending
        on how repeat_interleave works.
        """

        new_batch = type(self)(**self._repeat_create_args(repeat_sizes))
        return new_batch

    def _repeat_create_args(self, repeat_sizes: torch.TensorType) -> dict:
        if self.is_nested:
            raise NotImplementedError("Repeat not currently supported for nested tensors")

        # rxn nets are a separate list so do this separately
        if self.rxn_nets is not None:
            out_rxn_nets = []
            for i, repeat_num in enumerate(repeat_sizes):
                out_rxn_nets.extend([self.rxn_nets[i]] * int(repeat_num))
        else:
            out_rxn_nets = None

        out = dict(
            tkn_lib_collection=self.tkn_lib_collection,
            graphs=self.graphs,
            input_sequences=torch.repeat_interleave(self.input_sequences, repeat_sizes, dim=0),
            input_mol_scores=torch.repeat_interleave(self.input_mol_scores, repeat_sizes, dim=0),
            input_predictive_nxt_tkn_masks=torch.repeat_interleave(
                self.input_predictive_nxt_tkn_masks, repeat_sizes, dim=0
            ),
            input_frm_model_masks=torch.repeat_interleave(
                self.input_frm_model_masks, repeat_sizes, dim=0
            ),
            input_mol_masks=torch.repeat_interleave(self.input_mol_masks, repeat_sizes, dim=0),
            input_nonpad_masks=torch.repeat_interleave(
                self.input_nonpad_masks, repeat_sizes, dim=0
            ),
            sequence_indcs=(
                torch.repeat_interleave(self.sequence_indcs, repeat_sizes, dim=0)
                if self.sequence_indcs is not None
                else None
            ),
            output_sequences=(
                torch.repeat_interleave(self.output_sequences, repeat_sizes, dim=0)
                if self.output_sequences is not None
                else None
            ),
            output_from_model_masks=(
                torch.repeat_interleave(self.output_from_model_masks, repeat_sizes, dim=0)
                if self.output_from_model_masks is not None
                else None
            ),
            rxn_nets=out_rxn_nets,
        )
        return out

    def to_device(self, device):
        # moves everything but the token library collection to the device.
        self.graphs = self.graphs.to(device)
        self.input_sequences = self.input_sequences.to(device)
        self.input_mol_scores = self.input_mol_scores.to(device)
        self.input_predictive_nxt_tkn_masks = self.input_predictive_nxt_tkn_masks.to(device)
        self.input_frm_model_masks = self.input_frm_model_masks.to(device)
        self.input_mol_masks = self.input_mol_masks.to(device)
        self.input_nonpad_masks = self.input_nonpad_masks.to(device)
        if self.sequence_indcs is not None:
            self.sequence_indcs = self.sequence_indcs.to(device)
        if self.output_sequences is not None:
            self.output_sequences = self.output_sequences.to(device)
        if self.output_from_model_masks is not None:
            self.output_from_model_masks = self.output_from_model_masks.to(device)
        if self.conditioning is not None:
            self.conditioning = self.conditioning.to(device)
        return self

    def __len__(self):
        """Returns the number of sequences in the batch."""
        return self.input_sequences.shape[0]

    @property
    def max_seq_len(self):
        """Returns the maximum sequence length in the batch."""
        if self.input_sequences.is_nested:
            return max(o.shape[0] for o in self.input_sequences)
        else:
            return self.input_sequences.shape[1]

    @property
    def is_nested(self):
        """Returns whether the batch is nested or not."""
        return self.input_sequences.is_nested

    @property
    def device(self):
        return self.input_sequences.device

    def to_padded(self):
        """Returns the batch as a padded batch."""
        if not self.is_nested:
            return self
        else:
            self_as_dict = asdict(self)
            non_nested_tensor_props = {k: self_as_dict.pop(k) for k in self.NON_NESTED_TENSOR_PROPS}

            padded_props = {}
            for k, v in self_as_dict.items():
                if v is None:
                    padded_props[k] = None
                elif torch.is_floating_point(v):
                    padded_props[k] = torch.nested.to_padded_tensor(
                        v, padding=float(vocab.PAD_VALUE)
                    )
                elif v.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
                    padded_props[k] = torch.nested.to_padded_tensor(v, padding=int(vocab.PAD_VALUE))
                elif v.dtype == torch.bool:
                    padded_props[k] = torch.nested.to_padded_tensor(v, padding=False)

            return type(self)(
                **non_nested_tensor_props,
                **padded_props,
            )

    def get_sequences_over_mask(
        self, treat_eos_as_over: bool = True, treat_pad_as_over: bool = True
    ) -> torch.TensorType:
        """Determine which sequences are over based on padding and optionally EOS tokens.

        Args:
            treat_eos_as_over: If True, sequences ending with EOS are considered over.

        Returns:
            Boolean tensor [B] indicating which sequences are over.
        """
        if self.is_nested:
            raise NotImplementedError("get_sequences_over_mask is not supported for nested tensors")

        # Check for padding tokens
        if treat_pad_as_over:
            seqs_padded = (~self.input_nonpad_masks).any(dim=-1)
            # ^ if padded, then by definition over.
        else:
            seqs_padded = torch.zeros(
                self.input_nonpad_masks.shape[0],
                dtype=torch.bool,
                device=self.input_nonpad_masks.device,
            )

        if treat_eos_as_over:
            # Find the last non-padding position for each sequence, and gather the entries
            last_non_pad_pos = (self.output_sequences != vocab.PAD_VALUE).sum(dim=-1) - 1
            last_tokens = torch.gather(
                self.output_sequences, dim=1, index=last_non_pad_pos.unsqueeze(1)
            )  # [B, 1]
            last_tokens = last_tokens.squeeze(1)  # [B]

            # Check if last token is EOS
            seqs_eos = last_tokens.eq(vocab.EOS_VALUE)

            # Combine with padding condition
            seqs_over = torch.logical_or(seqs_padded, seqs_eos)
        else:
            seqs_over = seqs_padded

        return seqs_over

    @property
    def last_input_non_pad_pos(self) -> torch.TensorType:
        """Returns the last non-padding position for each sequence."""
        return self.input_nonpad_masks.sum(dim=-1) - 1  # [B]

    def cut_off_to_max_seq_len(self, max_seq_len):
        """Cuts off the batch to the max sequence length."""
        if max_seq_len >= self.max_seq_len:  # no need to cut off -- already fits
            return self

        if self.input_sequences.is_nested:

            def select_max_seq_len(t):
                return torch.nested.nested_tensor([o[:max_seq_len] for o in t], layout=torch.jagged)

            self.input_sequences = select_max_seq_len(self.input_sequences)
            self.input_mol_scores = select_max_seq_len(self.input_mol_scores)
            self.input_predictive_nxt_tkn_masks = coerce_offsets(
                select_max_seq_len(self.input_predictive_nxt_tkn_masks), self.input_sequences
            )

            self.input_frm_model_masks = select_max_seq_len(self.input_frm_model_masks)
            self.input_mol_masks = select_max_seq_len(self.input_mol_masks)
            self.input_nonpad_masks = select_max_seq_len(self.input_nonpad_masks)

            if self.sequence_indcs is not None:
                self.sequence_indcs = coerce_offsets(
                    select_max_seq_len(self.sequence_indcs), self.input_sequences
                )
            if self.output_sequences is not None:
                self.output_sequences = select_max_seq_len(self.output_sequences)
            if self.output_from_model_masks is not None:
                self.output_from_model_masks = select_max_seq_len(self.output_from_model_masks)
        else:
            self.input_sequences = self.input_sequences[:, :max_seq_len]
            self.input_mol_scores = self.input_mol_scores[:, :max_seq_len]
            self.input_predictive_nxt_tkn_masks = self.input_predictive_nxt_tkn_masks[
                :, :max_seq_len
            ]
            self.input_frm_model_masks = self.input_frm_model_masks[:, :max_seq_len]
            self.input_mol_masks = self.input_mol_masks[:, :max_seq_len]
            self.input_nonpad_masks = self.input_nonpad_masks[:, :max_seq_len]
            if self.sequence_indcs is not None:
                self.sequence_indcs = self.sequence_indcs[:, :max_seq_len]
            if self.output_sequences is not None:
                self.output_sequences = self.output_sequences[:, :max_seq_len]
            if self.output_from_model_masks is not None:
                self.output_from_model_masks = self.output_from_model_masks[:, :max_seq_len]
        return self


def padded_tokens_to_token_type(
    padded_tkns: torch.TensorType, tkn_lib_collection: vocab.TokenLibraryCollection
) -> torch.TensorType:
    """
    Marks each token in the sequence with its type (i.e., whether it is a molecule or a special token, and if
    a molecule token whether building block or final molecule).

    Note this function does not do anything special for padding tokens, i.e., they are marked as special tokens.
    """
    out_tensor = torch.zeros_like(padded_tkns, dtype=torch_settings.TORCH_INT_TYPE)
    # by default set all tokens as special; note that this includes padding tokens.
    out_tensor[padded_tkns >= tkn_lib_collection.bb_tkn_library.start_idx] = 1  # bb tokens
    out_tensor[padded_tkns >= tkn_lib_collection.mol_tkn_library.start_idx] = 2  # mol tokens
    return out_tensor


def padded_tokens_defining_action_type(
    padded_tkns: torch.TensorType, tkn_lib_collection: vocab.TokenLibraryCollection
) -> torch.TensorType:
    """
    Marks the what action token each molecular token is associated with. (i.e., which action token index it acts as an
    argument for). Action token indices are always set as -1.

    So for instance say we have action token indices 0, 1, and 2, with the rest of the tokens above
    these being molecular tokens, then if we have a padded_tkns tensor of:
    [
        [1,5,7,1,0,0],
        [1,9,1,6,2,3]]
    ]
    then the output tensor will be:
    [
        [-1,1,1,-1,-1,-1],
        [-1,1,-1,1,-1,2]]
    ]
    """
    out_tensor = torch.full_like(padded_tkns, fill_value=-1, dtype=torch_settings.TORCH_INT_TYPE)
    # ^ by default set tokens as -1

    # we're will iterate through the tensor forwards and propagate any action tokens to the right.
    # this is because the action tokens are always to the left of the molecular tokens.

    for i in range(padded_tkns.shape[1]):
        padded_tkns_i = padded_tkns[:, i]
        padded_tkns_i_is_action = padded_tkns_i <= tkn_lib_collection.special_tkn_library.end_idx

        # the action tokens are set back to -1: they define a new action and so should not be associated with any
        # previous action
        out_tensor[padded_tkns_i_is_action, i] = -1

        # and then we propagate the action tokens to the right -- note the parts that are incorrect will be subsequently
        # overwritten.
        out_tensor[padded_tkns_i_is_action, i + 1 :] = padded_tkns_i[
            padded_tkns_i_is_action
        ].unsqueeze(1)

    return out_tensor


def coerce_offsets(src, tgt):
    # from https://github.com/pytorch/pytorch/issues/138180
    # noted edited the max and min seq length as was swapped in issue (and also ternary operator didnt seem
    # to actually be doing anything...)
    assert torch.eq(src.offsets(), tgt.offsets()).all().item()
    assert src._ragged_idx == tgt._ragged_idx

    def mb_get_size(t):
        return t.shape[0] if t is not None else None

    return torch.nested.nested_tensor_from_jagged(
        src.values(),
        tgt.offsets(),
        None,
        src._ragged_idx,
        mb_get_size(src._min_seqlen_tensor),
        mb_get_size(src._max_seqlen_tensor),
    )


def create_padded_batch_from_non_padded_tensors(
    tkn_lib_collection: vocab.TokenLibraryCollection,
    graph_batch: pyg_data.Batch,
    all_sequences: list[torch.TensorType],
    all_predictive_tkn_masks: list[torch.TensorType],
    all_input_frm_model_masks: list[torch.TensorType],
    all_input_mol_masks: list[torch.TensorType],
    all_mol_scores: list[torch.TensorType],
    all_output_sequences: list[torch.TensorType] | None = None,
    all_output_frm_model_masks: list[torch.TensorType] | None = None,
    conditioning: torch.TensorType | None = None,
    target_molecules: list[str] | None = None,
    all_rxn_nets: list[reaction_network_graph.ReactionNetwork] | None = None,
):
    """
    Creates a padded batch from a list of non-padded tensors.
    """

    sequence_lengths = [seq.shape[0] for seq in all_sequences]
    max_seq_len = max(sequence_lengths)  # S

    def setup_pad_mask(seq_length):
        mask = torch.zeros((max_seq_len,), dtype=torch.bool)
        mask[:seq_length] = True
        return mask

    all_input_nonpad_masks = [setup_pad_mask(seq_length) for seq_length in sequence_lengths]

    # Use pad_end_of_tensor for 1D tensors (sequences, masks, scores)
    all_sequences = utils.pad_then_stack(
        all_sequences, vocab.PAD_VALUE, padded_final_length=max_seq_len
    )
    all_input_frm_model_masks = utils.pad_then_stack(
        all_input_frm_model_masks, False, padded_final_length=max_seq_len
    )
    all_input_mol_masks = utils.pad_then_stack(
        all_input_mol_masks, False, padded_final_length=max_seq_len
    )
    all_mol_scores = utils.pad_then_stack(all_mol_scores, 0.0, padded_final_length=max_seq_len)

    # Use pad_end_of_tensor for 2D tensors (predictive masks) - pad the first dimension (pad_dim=0)
    all_predictive_tkn_masks = utils.pad_then_stack(
        all_predictive_tkn_masks, False, pad_dim=0, padded_final_length=max_seq_len
    )

    # Either the output sequences are provided, in which case we pad them (in a similar manner to the input sequences)
    # or we use the method in the batch object to add them at the end.
    if (all_output_sequences is None) ^ (all_output_frm_model_masks is None):
        raise ValueError(
            "all_output_sequences and all_output_frm_model_masks must both be provided or both be None"
        )

    if all_output_sequences is not None:
        all_output_sequences = utils.pad_then_stack(
            all_output_sequences, vocab.PAD_VALUE, padded_final_length=max_seq_len
        )
        all_output_frm_model_masks = utils.pad_then_stack(
            all_output_frm_model_masks, False, padded_final_length=max_seq_len
        )
        add_output_seqs_at_end = False
    else:
        all_output_sequences = None
        all_output_frm_model_masks = None
        add_output_seqs_at_end = True

    batch = RxnNetTaskBatch(
        tkn_lib_collection=tkn_lib_collection,
        graphs=graph_batch,
        input_sequences=all_sequences,
        input_mol_scores=all_mol_scores,
        input_predictive_nxt_tkn_masks=all_predictive_tkn_masks,
        input_frm_model_masks=all_input_frm_model_masks,
        input_mol_masks=all_input_mol_masks,
        input_nonpad_masks=torch.stack(all_input_nonpad_masks),
        output_sequences=all_output_sequences,
        output_from_model_masks=all_output_frm_model_masks,
        conditioning=conditioning,
        target_molecules=target_molecules,
        rxn_nets=all_rxn_nets,
    )
    if add_output_seqs_at_end:
        batch.add_output_sequences()
    return batch


class RxnNetTaskCollateFunc:
    """
    Collate function for the ReactionNetworkDataset.
    """

    def __init__(
        self,
        rng: np_random.RandomState,
        prompt_to_action_probabilizer: Callable[[list[str]], float],
        bb_graphs: list[pyg_data.Data],
        bb_tkn_library: vocab.TokenLibrary,
        nested_tensors: bool = True,
        max_seq_len: int = 500,
        enable_conditioning: bool = True,
        prefix_free_generation: bool = True,
    ):
        self.rng = rng
        self.prompt_to_action_probabilizer = prompt_to_action_probabilizer
        self.bb_graphs = bb_graphs
        self.bb_tkn_library = bb_tkn_library

        self.serializer = serialization.Serializer(special_tkn_mol_score_values=None)
        self.nested_tensors = nested_tensors
        self.max_seq_len = max_seq_len
        self.enable_conditioning = enable_conditioning
        self.prefix_free_generation = prefix_free_generation

        # Initialize Morgan fingerprint generator only if conditioning is enabled
        if self.enable_conditioning and rdFingerprintGenerator is not None:
            self.morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        else:
            self.morgan_fp_gen = None

    def _extract_retro_target_molecule(self, prompt: list[str]) -> str | None:
        """
        Extract the target molecule from a retrosynthesis prompt.

        Args:
            prompt: List of tokens in the prompt

        Returns:
            Target molecule SMILES string or None if not a retrosynthesis task
        """
        if len(prompt) >= 2 and prompt[0] == vocab.SPECIAL_TOKENS.RETRO:
            return prompt[1]  # Target molecule is right after <RETRO>
        return None

    def _compute_morgan_fingerprint(self, smiles: str) -> np.ndarray:
        """
        Compute Morgan fingerprint for a SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Morgan fingerprint as numpy array
        """
        if Chem is None or self.morgan_fp_gen is None:
            raise ImportError("RDKit is required for Morgan fingerprint computation")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        fp = self.morgan_fp_gen.GetFingerprint(mol)
        # Convert to numpy array (bit vector)
        fp_array = np.zeros(1024, dtype=np.float32)
        for i in range(fp.GetNumBits()):
            if fp.GetBit(i):
                fp_array[i] = 1.0
        return fp_array

    def __call__(self, list_of_reaction_net_tasks: list[RxnNetTask]) -> RxnNetTaskBatch:
        # 1. Create the token library collection.
        all_mols_seen = set(
            node.canon_smi for o in list_of_reaction_net_tasks for node in o.rxn_net.nodes
        )
        new_mols_seen = all_mols_seen - self.bb_tkn_library.token_set
        new_mol_tkn_library = vocab.TokenLibrary(
            sorted(list(new_mols_seen)),
            start_idx=self.bb_tkn_library.end_idx + 1,
            molecule_tokens=True,
        )
        tkn_lib_collection = vocab.TokenLibraryCollection(
            [vocab.SPECIAL_TOKENS_LIBRARY, self.bb_tkn_library, new_mol_tkn_library]
        )

        # 2. Create the graphs.
        new_graphs = [molecular_graphs.from_smiles(smi) for smi in new_mol_tkn_library.tokens]
        batch = molecular_graphs.from_data_list(self.bb_graphs + new_graphs)

        # 3. Create the sequences (i.e., serialize)
        all_sequences = []
        all_output_sequences = []
        all_output_frm_model_masks = []
        all_mol_scores = []
        all_predictive_tkn_masks = []
        all_input_mol_masks = []
        all_input_frm_model_masks = []
        all_conditioning_vectors = []
        all_target_molecules = []

        all_rxn_nets = [el.rxn_net for el in list_of_reaction_net_tasks]
        for rxn_net_task in list_of_reaction_net_tasks:
            action_to_prob = self.prompt_to_action_probabilizer(rxn_net_task.prompt)

            # Use the prompt as initial sequence, or empty list for prefix-free generation
            initial_seq = [] if self.prefix_free_generation else rxn_net_task.prompt

            seq, out_masks, tkns_predicted_by_model, out_scores = self.serializer(
                rxn_net_task.rxn_net,
                initial_seq=initial_seq,
                mol_scorer=rxn_net_task.mol_scorer,
                action_to_prob=action_to_prob,
                tkn_lib_collection=tkn_lib_collection,
                rng=self.rng,
                return_as_str=False,
            )
            seq = torch.tensor(seq, dtype=torch_settings.TORCH_INT_TYPE)  # [S']
            all_sequences.append(seq[:-1])
            all_output_sequences.append(seq[1:])

            # np then torch as apparently quicker:
            out_mask = torch.tensor(np.array(out_masks)[:-1], dtype=torch.bool)  # [S', V]
            all_predictive_tkn_masks.append(out_mask)

            tkns_predicted_by_model = torch.tensor(
                tkns_predicted_by_model, dtype=torch.bool
            )  # [S']
            all_input_frm_model_masks.append(tkns_predicted_by_model[:-1])
            all_output_frm_model_masks.append(tkns_predicted_by_model[1:])  # [S']

            input_mol_masks = torch.tensor(
                [True if o is not None else False for o in out_scores[:-1]], dtype=torch.bool
            )  # [S']
            all_input_mol_masks.append(input_mol_masks)

            mol_scores = torch.tensor(
                [(o if o is not None else 0.0) for o in out_scores[:-1]],
                dtype=torch_settings.TORCH_FLT_TYPE,
            )  # [S']
            all_mol_scores.append(mol_scores)

            # 4. Compute conditioning vector from retrosynthesis target molecule
            target_smiles = self._extract_retro_target_molecule(rxn_net_task.prompt)
            all_target_molecules.append(target_smiles)

            if self.enable_conditioning:
                if target_smiles is not None:
                    # Compute Morgan fingerprint for retrosynthesis target
                    try:
                        morgan_fp = self._compute_morgan_fingerprint(target_smiles)
                        conditioning_vector = torch.tensor(
                            morgan_fp, dtype=torch_settings.TORCH_FLT_TYPE
                        )
                    except (ValueError, ImportError):
                        # Fallback to zero vector if fingerprint computation fails
                        conditioning_vector = torch.zeros(1024, dtype=torch_settings.TORCH_FLT_TYPE)
                else:
                    # Not a retrosynthesis task - use zero vector
                    conditioning_vector = torch.zeros(1024, dtype=torch_settings.TORCH_FLT_TYPE)

                all_conditioning_vectors.append(conditioning_vector)
            else:
                all_conditioning_vectors.append(
                    torch.zeros(1024, dtype=torch_settings.TORCH_FLT_TYPE)
                )

        # 5. Combine conditioning vectors
        if self.enable_conditioning:
            conditioning_tensor = torch.stack(all_conditioning_vectors)  # [B, 1024]
        else:
            conditioning_tensor = None

        if self.nested_tensors:
            input_mol_masks = torch.nested.nested_tensor(all_input_mol_masks, layout=torch.jagged)
            input_nonpad_masks = torch.ones_like(input_mol_masks, dtype=torch.bool)
            batch = RxnNetTaskBatch(
                tkn_lib_collection=tkn_lib_collection,
                graphs=batch,
                input_sequences=torch.nested.nested_tensor(all_sequences, layout=torch.jagged),
                output_sequences=torch.nested.nested_tensor(
                    all_output_sequences, layout=torch.jagged
                ),
                input_mol_scores=torch.nested.nested_tensor(all_mol_scores, layout=torch.jagged),
                input_predictive_nxt_tkn_masks=torch.nested.nested_tensor(
                    all_predictive_tkn_masks, layout=torch.jagged
                ),
                input_frm_model_masks=torch.nested.nested_tensor(
                    all_input_frm_model_masks, layout=torch.jagged
                ),
                input_mol_masks=input_mol_masks,
                input_nonpad_masks=input_nonpad_masks,
                output_from_model_masks=torch.nested.nested_tensor(
                    all_output_frm_model_masks, layout=torch.jagged
                ),
                conditioning=conditioning_tensor,
                target_molecules=all_target_molecules,
                rxn_nets=all_rxn_nets,
            )
        else:
            # 4. Pad and collate
            batch = create_padded_batch_from_non_padded_tensors(
                tkn_lib_collection=tkn_lib_collection,
                graph_batch=batch,
                all_sequences=all_sequences,
                all_predictive_tkn_masks=all_predictive_tkn_masks,
                all_input_frm_model_masks=all_input_frm_model_masks,
                all_input_mol_masks=all_input_mol_masks,
                all_mol_scores=all_mol_scores,
                all_output_sequences=all_output_sequences,
                all_output_frm_model_masks=all_output_frm_model_masks,
                conditioning=conditioning_tensor,
                target_molecules=all_target_molecules,
                all_rxn_nets=all_rxn_nets,
            )

        batch = batch.cut_off_to_max_seq_len(self.max_seq_len)
        return batch


def add_new_tensors_to_jagged_padded(
    padded_tensors: torch.TensorType | None, tensor_to_append: torch.TensorType, pad_value=0.0
):
    """Append new tensors to existing padded tensors representing jagged arrays.

    Takes a batch of padded sequences and appends new tensors to the end of each sequence
    (after the non-padding part). Useful for extending jagged arrays stored in padded format.

    Args:
        padded_tensors: 2D tensor [B, S] with padding at the end of each row. If None then just return tensor_to_append.
        tensor_to_append: 2D tensor [B, S'] to append to each row after non-padding content
        pad_value: Value used for padding (default: 0.0)

    Returns:
        New 2D tensor [B, S''] with tensor_to_append added after non-padding content,
        with minimal padding removed.

    Example:
        >>> padded = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])  # 0 is padding
        >>> to_append = torch.tensor([[4, 5], [6, 7]])
        >>> result = add_new_tensors_to_jagged_padded(padded, to_append, pad_value=0)
        >>> # Result: [[1, 2, 4, 5], [3, 6, 7, 0]]
    """

    if padded_tensors is None:
        return tensor_to_append

    # Currently we have made assumption that the new tensors are 2D.
    if not len(padded_tensors.shape) == 2:
        raise NotImplementedError(
            "add_new_tensors_to_jagged_padded only supports 2D tensors currently"
        )

    # Do some checks that the input tensors are compatible.
    if tensor_to_append.shape[0] != padded_tensors.shape[0]:
        raise ValueError(
            "new_tensors and padded_tensors must have the same number of first dimension"
        )
    if (
        padded_tensors.dtype != tensor_to_append.dtype
        or padded_tensors.device != tensor_to_append.device
    ):
        raise ValueError("new_tensors and padded_tensors must have the same dtype and device")

    # Work out where the padding currently starts:
    stop_location = (padded_tensors != pad_value).sum(dim=-1)

    # Now will create the new tensor (defensively with maximum possible width), and prefill the first bit with the old
    # tensor
    new_tensor = torch.full(
        (padded_tensors.shape[0], padded_tensors.shape[1] + tensor_to_append.shape[1]),
        pad_value,
        dtype=padded_tensors.dtype,
        device=padded_tensors.device,
    )
    new_tensor[:, : padded_tensors.shape[1]] = padded_tensors

    # We now work out where to overwrite the new_tensor with the tensor_to_append and do so
    scatter_idx = torch.arange(
        tensor_to_append.shape[1], dtype=torch.long, device=padded_tensors.device
    )[None, :]
    scatter_idx = scatter_idx + stop_location[:, None]
    new_tensor.scatter_(dim=1, index=scatter_idx, src=tensor_to_append)

    # remove any excess padding (we were defensive when we created new_tensor)
    max_new_length = torch.max((new_tensor != pad_value).sum(dim=-1)).item()
    new_tensor = new_tensor[:, :max_new_length]

    return new_tensor


@dataclass
class CurrentSequences(RxnNetTaskBatch):
    """
    CurrentSequences extend RxnNetTaskBatch for inference. It can hold incomplete sequences, multiple proposals
    of completions etc.
    """

    # Tensor mapping each completion back to its original example index in the batch. Should be consecutive.
    # E.g., for beam search with width 5 and 2 examples: [0,0,0,0,0, 1,1,1,1,1] (B'' = B * num_samples_per_example)
    # Note: Only optional here for the dataclass field ordering; should always be provided AND WE WILL CHECK THIS.
    example_indices: torch.TensorType | None = None  # [B'']

    # Scores for each completion (not length normalized)
    # Higher scores indicate better completions
    scores: torch.TensorType | None = None
    # [B'', S']  (S' as steps for generation may conist of multiple tokens, so may not match the length of the input sequences)

    # Scores for each completion at the last step.
    scores_of_last_step: torch.TensorType | None = None  # [B']

    # Molscorer dict, given the example index, return the mol scorer for that example. Each original example should
    # have its own molscorer, linked to the prompt.
    mol_scorer: dict[int, Callable[[str], float]] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.is_nested:
            raise ValueError("CurrentSequences only supports padded tensors, not nested tensors")
        if self.example_indices is None:
            raise ValueError(
                "example_indices must be provided (only optional for dataclass field ordering)"
            )

    def select_submask(self, submask: torch.TensorType) -> "CurrentSequences":
        new_current_sequences = CurrentSequences(
            **super().select_submask(submask).__dict__,
            example_indices=self.example_indices[submask],
            scores=self.scores[submask] if self.scores is not None else None,
            scores_of_last_step=self.scores_of_last_step[submask]
            if self.scores_of_last_step is not None
            else None,
        )
        used_example_indices = set(
            torch.unique(self.example_indices[submask]).detach().cpu().tolist()
        )
        new_current_sequences.mol_scorer = {k: self.mol_scorer[k] for k in used_example_indices}
        return new_current_sequences

    def get_mol_score_for_input_seq_idx(self, input_seq_idx: int) -> Callable[[str], float]:
        example_idx = self.example_indices[input_seq_idx].item()
        return self.mol_scorer[example_idx]

    @property
    def last_scores_non_pad_pos(self) -> torch.TensorType:
        """Returns the last non-padding position for each score sequence."""
        if self.scores is None:
            raise ValueError("Cannot compute last_scores_non_pad_pos when scores is None")
        return (self.scores != vocab.PAD_VALUE).sum(dim=-1) - 1  # [B']

    def to_device(self, device: torch.device) -> "CurrentSequences":
        super().to_device(device)
        self.example_indices = self.example_indices.to(device)
        if self.scores is not None:
            self.scores = self.scores.to(device)
        if self.scores_of_last_step is not None:
            self.scores_of_last_step = self.scores_of_last_step.to(device)
        return self

    def to_padded(self) -> "CurrentSequences":
        if self.is_nested:
            raise ValueError("CurrentSequences only supports padded tensors, not nested tensors")
        return self

    def _repeat_create_args(self, repeat_sizes: torch.TensorType) -> dict:
        if self.is_nested:
            raise ValueError("Repeat not currently supported for nested tensors")
        # ^ this is technically covered by the super method, but as this method's implementation also currently
        # has same limitation, we shall check again here.
        out = super()._repeat_create_args(repeat_sizes)
        out["example_indices"] = torch.repeat_interleave(self.example_indices, repeat_sizes, dim=0)
        if self.scores is not None:
            out["scores"] = torch.repeat_interleave(self.scores, repeat_sizes, dim=0)
        if self.scores_of_last_step is not None:
            out["scores_of_last_step"] = torch.repeat_interleave(
                self.scores_of_last_step, repeat_sizes, dim=0
            )
        if self.mol_scorer is not None:
            # note that we can just copy mol_scorer across -> don't need to repeat as the mol scorers are stored wrt
            # the original example indices.
            out["mol_scorer"] = self.mol_scorer
        return out

    def cut_off_to_max_seq_len(
        self, max_seq_len: int, max_seq_len_for_scores: int | None = None
    ) -> "CurrentSequences":
        # note that scores len can be different from the input sequences len, so we need to specify this.
        super().cut_off_to_max_seq_len(max_seq_len)
        if max_seq_len_for_scores is not None and self.scores is not None:
            self.scores = self.scores[:, :max_seq_len_for_scores]
        return self

    @classmethod
    def from_tokens(cls, *args, **kwargs):
        """CurrentSequences does not support from_tokens creation."""
        raise NotImplementedError(
            "CurrentSequences does not currently support from_tokens creation"
        )
