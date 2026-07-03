"""

See :
https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
https://github.com/karpathy/nanoGPT/blob/master/model.py
https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
https://github.com/pytorch-labs/gpt-fast
"""

import warnings
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from synlad.data.components import synthesis_dataset as dataset
from synlad.models.components import synthesis_mha as mha
from synlad.tokenization import synthesis_vocab as vocab
from synlad.utils import synthesis_utils as utils


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization with a learnable per-feature scale."""

    def __init__(self, dim: int, eps: float = 1e-5, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@dataclass
class SwigFeedForwardParams:
    """Hyperparameters for :class:`SwigFeedForward`."""

    dim: int
    intermediate_size: int
    dtype: torch.dtype = torch.float32


class SwigFeedForward(nn.Module):
    """SwiGLU feed-forward block: ``w2(SiLU(w1(x)) * w3(x))``."""

    def __init__(self, params: SwigFeedForwardParams) -> None:
        super().__init__()
        self.w1 = nn.Linear(params.dim, params.intermediate_size, bias=False, dtype=params.dtype)
        self.w3 = nn.Linear(params.dim, params.intermediate_size, bias=False, dtype=params.dtype)
        self.w2 = nn.Linear(params.intermediate_size, params.dim, bias=False, dtype=params.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


@dataclass
class TransformerBlockParams:
    """Hyperparameters for :class:`TransformerBlock`. ``conditioning_dim > 0`` enables cross-attention."""

    dim: int
    n_heads: int = 8
    eps: float = 1e-5
    attn_bias: bool = True
    dropout: float = 0.0
    dtype: torch.dtype = torch.float32
    conditioning_dim: int = 0  # If > 0, enables cross-attention conditioning


class TransformerBlock(nn.Module):
    """Self-attention + SwiGLU FFN block, with optional cross-attention conditioning."""

    PADDED_NESTED_PARAMETER_NAMES = [
        ("attn.in_proj_weight", "attn.packed_proj.weight"),
        ("attn.in_proj_bias", "attn.packed_proj.bias"),
    ]

    def __init__(self, params: TransformerBlockParams, nested: bool = True) -> None:
        super().__init__()

        assert params.dim % params.n_heads == 0

        self.ffn_norm = RMSNorm(params.dim, eps=params.eps, dtype=params.dtype)
        self.attn_norm = RMSNorm(params.dim, eps=params.eps, dtype=params.dtype)
        if nested:
            self.attn = mha.MultiHeadAttention(
                params.dim,
                params.dim,
                params.dim,
                params.dim,
                nheads=params.n_heads,
                dropout=params.dropout,
                bias=params.attn_bias,
                batch_first=True,
            )
        else:
            self.attn = nn.MultiheadAttention(
                params.dim,
                params.n_heads,
                dropout=params.dropout,
                bias=params.attn_bias,
                batch_first=True,
            )

        # --- Cross-Attention Block (New) ---
        # This block will process the conditioning tensor.
        self.cross_attn_norm = RMSNorm(params.dim, eps=params.eps, dtype=params.dtype)

        # Cross-attention layer - only create if conditioning is enabled
        if hasattr(params, "conditioning_dim") and params.conditioning_dim > 0:
            if nested:
                # Use our custom MultiHeadAttention for nested tensors
                self.cross_attn = mha.MultiHeadAttention(
                    E_q=params.dim,  # query from sequence
                    E_k=params.conditioning_dim,  # key from conditioning
                    E_v=params.conditioning_dim,  # value from conditioning
                    E_total=params.dim,  # output dimension
                    nheads=params.n_heads,
                    dropout=params.dropout,
                    bias=params.attn_bias,
                    dtype=params.dtype,
                )
            else:
                # Use PyTorch's standard MultiheadAttention for padded tensors
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=params.dim,
                    num_heads=params.n_heads,
                    kdim=params.conditioning_dim,
                    vdim=params.conditioning_dim,
                    dropout=params.dropout,
                    bias=params.attn_bias,
                    batch_first=True,
                )
            self.has_cross_attn = True
        else:
            self.has_cross_attn = False

        self.ffn = SwigFeedForward(
            SwigFeedForwardParams(
                dim=params.dim, intermediate_size=params.dim * 4, dtype=params.dtype
            )
        )
        self._params = params
        self.nested = nested

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        super_state_dict = super().state_dict(destination, prefix, keep_vars)
        if (
            not self.nested
        ):  # Check if the currently instantiated attn is the padded version, if so change
            # the state_dict keys to match the nested version
            for old_key, new_key in self.PADDED_NESTED_PARAMETER_NAMES:
                old_key = prefix + old_key
                new_key = prefix + new_key
                if old_key in super_state_dict:
                    super_state_dict[new_key] = super_state_dict.pop(old_key)
        return super_state_dict

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not self.nested:
            # Check if the currently instantiated attn is the padded version, if so change
            # the state_dict keys to match the nested version
            for padded_key, saved_key in self.PADDED_NESTED_PARAMETER_NAMES:
                padded_key = prefix + padded_key
                saved_key = prefix + saved_key
                if saved_key in state_dict:
                    state_dict[padded_key] = state_dict.pop(saved_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(
        self, x, attn_causal_mask=None, pad_mask=None, conditioning=None, conditioning_mask=None
    ):
        """
        Args:
            x (torch.Tensor): input sequence embeddings
            attn_causal_mask (torch.Tensor, optional): causal attention mask. Defaults to None.
            pad_mask (torch.Tensor, optional): padding mask of shape (N, S). Defaults to None.
                True means does not attend!
            conditioning (torch.Tensor, optional): conditioning vectors of shape (N, L_cond, conditioning_dim)
            conditioning_mask (torch.Tensor, optional): mask for conditioning vectors
        """
        # Self-attention
        hx = self.attn_norm(x)
        if self.nested:
            if attn_causal_mask is not None:
                raise NotImplementedError(
                    "attn_causal_mask is not None, but nested attention does not support it."
                )
            if pad_mask is not None:
                warnings.warn(
                    "pad_mask is not None, but nested attention does require it. Ignoring it.",
                    stacklevel=2,
                )
            hx = self.attn(hx, hx, hx, is_causal=True)
        else:
            hx = self.attn(
                hx,
                hx,
                hx,
                attn_mask=attn_causal_mask,
                key_padding_mask=pad_mask,
                is_causal=True,
                need_weights=False,
            )[0]
        h = x + hx

        # Cross-attention with conditioning (if available)
        if self.has_cross_attn and conditioning is not None:
            assert conditioning.size(0) == x.size(0), "Batch size mismatch"
            assert conditioning.size(-1) == self._params.conditioning_dim, (
                "Conditioning dim mismatch"
            )
            if self.nested:
                raise NotImplementedError(
                    "Cross-attention with conditioning is not implemented for nested tensors."
                )
            else:
                # Apply layer norm to the output of self-attention + residual
                cross_hx = self.cross_attn_norm(h)

                # Padded tensor path
                if conditioning_mask is not None:
                    conditioning_mask = ~conditioning_mask  # Invert: True=attend -> False=ignore

                cross_out = self.cross_attn(
                    query=cross_hx,  # (N, L_seq, dim)
                    key=conditioning,  # (N, L_cond, conditioning_dim)
                    value=conditioning,  # (N, L_cond, conditioning_dim)
                    key_padding_mask=conditioning_mask,  # (N, L_cond) - True=ignore
                    need_weights=False,
                )[0]

                # Residual connection with cross-attention
                h = h + cross_out

        # Feed-forward
        o = h + self.ffn(self.ffn_norm(h))
        return o


class FPBlock(nn.Module):
    """Two-layer MLP block with LayerNorm, GELU, and a residual connection (with optional projection)."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        out_dim: int | None = None,
        dtype=torch.float32,
    ):
        super().__init__()
        out_dim = out_dim or dim
        self.ln = nn.LayerNorm(dim, dtype=dtype)
        self.fc1 = nn.Linear(dim, hidden_dim, dtype=dtype)
        self.fc2 = nn.Linear(hidden_dim, out_dim, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions differ (local skip per block)
        if dim != out_dim:
            self.residual_proj = nn.Linear(dim, out_dim, bias=False, dtype=dtype)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return self.residual_proj(x) + h


class FingerprintEncoder(nn.Module):
    """Stack of :class:`FPBlock`s that maps a molecule fingerprint to a fixed-size embedding."""

    def __init__(
        self,
        fp_dim: int,
        dims: tuple[int],
        hidden_dims: tuple[int],
        final_dim: int,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        assert len(dims) == len(hidden_dims), "dims and hidden_dims must have the same length"

        # Build blocks
        blocks = []
        in_dim = fp_dim
        for out_dim, hidden_dim in zip(dims, hidden_dims, strict=False):
            blocks.append(FPBlock(in_dim, hidden_dim, dropout, out_dim=out_dim, dtype=dtype))
            in_dim = out_dim
        self.blocks = nn.ModuleList(blocks)
        self.out_layer = nn.Linear(out_dim, final_dim)
        self.out_ln = nn.LayerNorm(final_dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x  # [num_graphs, fp_dim]
        for block in self.blocks:
            h = block(h)
        h = self.out_layer(h)
        h = self.out_ln(h)
        return h


@dataclass
class SynthesisDecoderParams:
    """Hyperparameters for :class:`SynthesisDecoderModel` (depth, width, fingerprint dims, etc.)."""

    max_special_token_size: int
    # ^ we will learn this number of embeddings for the special tokens

    dim: int = 768
    max_seq_size: int = 1024
    num_transformer_blocks: int = 12
    n_heads: int = 12

    dropout: float = 0.0
    dtype: str = "float32"

    # Fingerprint params
    fingerprint_dim: int = 2048
    fingerprint_dims: tuple[int] = (1024, 1024)
    fingerprint_hidden_dims: tuple[int] = (1536, 1024)

    nested: bool = False

    conditioning_dim: int = 1024  # 0 means no conditioning, >0 enables conditioning

    @property
    def dtype_as_torch_type(self):
        if isinstance(self.dtype, str):
            dtype = getattr(torch, self.dtype)
            return dtype
        elif isinstance(self.dtype, torch.dtype):
            return self.dtype
        else:
            raise ValueError(f"Invalid dtype: {self.dtype}. Must be a string or torch.dtype.")


DEBUG_PARAMS = {
    "dim": 16,
    "num_transformer_blocks": 2,
    "n_heads": 2,
    "dropout": 0.0,
    "fingerprint_dim": 1024,
    "fingerprint_hidden_dim": 16,
    "max_seq_size": 128,
}


class SynthesisDecoderModel(nn.Module):
    """
    Synthesis decoder model.

    A decoder-only transformer model, which encodes the graphs using a fingerprint encoder. Our implementation here supports
    both padded and nested sequence inputs. Nested tensors should be faster and so preferred, but the padded tensors
    support more operations (for future development).
    """

    def __init__(self, params: SynthesisDecoderParams):
        super().__init__()
        # Embedding layers
        self.special_token_embeddings = nn.Embedding(
            params.max_special_token_size, params.dim, dtype=params.dtype_as_torch_type
        )
        self.positional_embeddings = nn.Embedding(
            params.max_seq_size,
            params.dim,
            dtype=params.dtype_as_torch_type,
            padding_idx=vocab.PAD_VALUE,
        )
        self.token_type_embeddings = nn.Embedding(3, params.dim, dtype=params.dtype_as_torch_type)

        self.fingerprint_mlp = FingerprintEncoder(
            fp_dim=params.fingerprint_dim,
            dims=params.fingerprint_dims,
            hidden_dims=params.fingerprint_hidden_dims,
            dropout=params.dropout,
            dtype=params.dtype_as_torch_type,
            final_dim=params.dim,
        )

        # Transformer layers
        transformer_params = TransformerBlockParams(
            dim=params.dim,
            n_heads=params.n_heads,
            eps=1e-5,
            attn_bias=True,
            dropout=params.dropout,
            dtype=params.dtype_as_torch_type,
            conditioning_dim=params.conditioning_dim,  # Pass conditioning dimension
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(transformer_params, params.nested)
                for _ in range(params.num_transformer_blocks)
            ]
        )
        self._params = params
        if not params.nested:
            self.register_buffer(
                "attn_causal_mask",
                torch.triu(
                    torch.ones((params.max_seq_size, params.max_seq_size), dtype=torch.bool),
                    diagonal=1,
                ),
                persistent=False,
            )
            # ^ for Torch's MHA True means we do not attend!
            self.nested = False
        else:
            self.nested = True

    def forward(
        self,
        batch: dataset.RxnNetTaskBatch,
        convert_logits_to_probs: bool = False,
        conditioning: torch.Tensor = None,
        conditioning_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: logits or probs based on convert_logits_to_probs and logits.

        Args:
            batch: The input batch
            convert_logits_to_probs: Whether to convert logits to probabilities
            conditioning: Optional conditioning vector of shape [B, conditioning_dim]
        """
        # ### Step 1: Embed the graphs (using fingerprints) ###
        embedded_graphs = self.fingerprint_mlp(batch.graphs.fingerprints)

        # ### Step 2: Get all the embeddings for special tokens ###
        all_special_token_embeddings = self.special_token_embeddings.weight

        # ### Step 3: Create all the embeddings ###
        all_embeddings = torch.cat([all_special_token_embeddings, embedded_graphs], dim=0)
        repeats_ = torch.tensor(
            [len(el) for el in batch.tkn_lib_collection.token_libraries],
            device=all_embeddings.device,
        )
        type_embeddings = torch.repeat_interleave(
            self.token_type_embeddings.weight, repeats_, dim=0
        )
        all_embeddings = all_embeddings + type_embeddings

        # ### Step 4: Collect out the correct embeddings ###
        if batch.input_sequences.is_nested:
            embedded_sequences = F.embedding(
                batch.input_sequences, all_embeddings, padding_idx=vocab.PAD_VALUE
            )
        else:
            embedded_sequences = F.embedding(
                batch.input_sequences, all_embeddings, padding_idx=vocab.PAD_VALUE
            )

        # ### Step 5: Add positional embeddings ###
        positionally_embedded = self.positional_embeddings(batch.sequence_indcs)

        # ### Step 6: Combine embeddings ###
        x = embedded_sequences + positionally_embedded

        # ### Step 6.5: Prepare conditioning if provided ###
        # Use conditioning from parameter first, then from batch
        if conditioning is None:
            conditioning = batch.conditioning

        # Fix conditioning shape for cross-attention: [B, conditioning_dim] -> [B, 1, conditioning_dim]
        if conditioning is not None and conditioning.dim() == 2:
            conditioning = conditioning.unsqueeze(1)  # Add sequence dimension of 1
            # Also adjust conditioning_mask to match the new sequence dimension
            if conditioning_mask is not None:
                conditioning_mask = conditioning_mask.unsqueeze(1)  # [B] -> [B, 1]

        # ### Step 7: Pass through transformer blocks ###
        if not batch.input_sequences.is_nested:
            max_seq_len = batch.input_sequences.shape[1]
            attn_causal_mask = self.attn_causal_mask[:max_seq_len, :max_seq_len]
            pad_mask = torch.logical_not(batch.input_nonpad_masks)
            # ^ for Torch's MHA True means we do not attend, so we need to invert this mask
        else:
            attn_causal_mask = None
            pad_mask = None

        # ### Step 7: Pass through transformer blocks with conditioning ###
        x_f = self._run_transformer_blocks(
            x,
            attn_causal_mask=attn_causal_mask,
            pad_mask=pad_mask,
            conditioning=conditioning,
            conditioning_mask=conditioning_mask,
        )

        # ### Step 8: Compute logits ### (do this against embeddings so flexible for new graphs)
        logits = F.linear(x_f, all_embeddings)  # [B, S, V]
        # ^ matmul's backward pass is not implemented for nested tensors, but linear is, so going with linear ...

        # ### Step 9: Apply maskings ###
        if batch.input_sequences.is_nested:
            mask = torch.logical_not(batch.input_predictive_nxt_tkn_masks)
        else:
            mask = torch.logical_and(
                torch.logical_not(batch.input_predictive_nxt_tkn_masks),
                batch.input_nonpad_masks[:, :, None],
            )
            # ^ for the pad positions we don't want to change the logits, as otherwise all will be set to -inf
        logits = torch.masked_fill(logits, mask, float("-inf"))

        # ### Step 10: Apply softmax to create probabilities if applicable ###
        if convert_logits_to_probs:
            out_0 = self.logits_to_probs(logits, batch.input_nonpad_masks)
        else:
            out_0 = logits
            if not batch.input_sequences.is_nested:
                out_0[torch.logical_not(batch.input_nonpad_masks)] = vocab.PAD_VALUE

        return out_0, logits

    def logits_to_probs(
        self, logits: torch.Tensor, nonpad_masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        if logits.is_nested:
            out = logits.softmax(dim=-1)
        else:
            assert nonpad_masks is not None, "nonpad_masks must be provided for non-nested tensors"
            out = torch.full_like(logits, fill_value=vocab.PAD_VALUE)
            out[nonpad_masks] = logits[nonpad_masks].softmax(dim=-1)
        return out

    def _run_transformer_blocks(
        self, x, attn_causal_mask=None, pad_mask=None, conditioning=None, conditioning_mask=None
    ):
        # we broke this out into a separate function so that can be compiled separately. this is because
        # compiling the whole of forward when in nested mode causes an error due to a guard fail, whereas compiling
        # just this (which is a lot of the forward) works fine.
        for block in self.transformer_blocks:
            x = block(
                x,
                attn_causal_mask=attn_causal_mask,
                pad_mask=pad_mask,
                conditioning=conditioning,
                conditioning_mask=conditioning_mask,
            )
        return x

    def forward_to_loss(
        self,
        batch: dataset.RxnNetTaskBatch,
        convert_logits_to_probs: bool = False,
        conditioning: torch.Tensor = None,
        conditioning_mask: torch.Tensor = None,
        reduction: str = "mean",
    ):
        """
        Convenience method for computing the loss during with forward pass. This is not in the main forward pass as
        as the unbind used in compute_loss for nested tensors is not compilable.
        """
        # note that when compiled this will use original function,
        # see https://discuss.pytorch.org/t/forward-function-not-being-compiled-by-default/214270
        out_0, logits = self(
            batch,
            convert_logits_to_probs=convert_logits_to_probs,
            conditioning=conditioning,
            conditioning_mask=conditioning_mask,
        )
        loss = self.compute_loss(batch, logits, reduction=reduction)
        return out_0, loss

    def compute_loss(self, batch, logits, reduction: str = "mean"):
        if batch.input_sequences.is_nested:
            if reduction == "none":
                raise NotImplementedError(
                    "reduction='none' is not currently supported for nested tensors"
                )
            all_logits_together = torch.cat(logits.unbind(), dim=0)
            all_output_sequences = torch.cat(batch.output_sequences.unbind(), dim=0)
            valid_locs = torch.cat(batch.output_from_model_masks.unbind(), dim=0)
            all_logits_together = all_logits_together[valid_locs]
            all_output_sequences = all_output_sequences[valid_locs]
        else:
            # if want to keep the same shape we call out this other function:
            if reduction == "none":
                return utils.masked_cross_entropy_efficient(
                    logits, batch.output_sequences, batch.output_from_model_masks
                )

            valid_locs = torch.logical_and(batch.input_nonpad_masks, batch.output_from_model_masks)
            all_logits_together = logits[valid_locs]
            all_output_sequences = batch.output_sequences[valid_locs]

        assert len(all_logits_together.shape) - 1 == len(all_output_sequences.shape)
        loss = F.cross_entropy(
            input=all_logits_together, target=all_output_sequences, reduction=reduction
        )
        return loss

    def evaluate_predictions(
        self, batch, picked_predictions, return_all_accuracies=False, return_padded=False
    ):
        """Returns the per token accuracy for each element of the batch.

        Args:
            batch (dataset.RxnNetTaskBatch): The batch to evaluate.
            picked_predictions (torch.Tensor): The predictions to evaluate.
            return_all_accuracies (bool, optional):
                If True return all accuracies (i.e., for each token in the batch) or if False just return the accuracy
                per token. Defaults to False.
        """
        if batch.input_sequences.is_nested:
            # We will deal with nested tensors for now by just converting to padded form.
            # only use this function during inference, so does not have to be the "most" efficient...
            all_preds_together = torch.nested.to_padded_tensor(picked_predictions, padding=0.0)
            all_output_sequences = torch.nested.to_padded_tensor(
                batch.output_sequences, padding=0.0
            )
            valid_locs = torch.nested.to_padded_tensor(batch.output_from_model_masks, padding=False)
            # ^ valid locs will also indicate "False" for the pad positions.
        else:
            all_preds_together = picked_predictions
            all_output_sequences = batch.output_sequences
            valid_locs = torch.logical_and(batch.input_nonpad_masks, batch.output_from_model_masks)
            # ^ the output from model masks should be False for the pad positions, but being defensive here.

        # We do this a slightly round about way below to maintain the accuracy computation per batch (where each
        # element in the batch could have a different number of "valid" tokens). (could use sparse instead, but leave
        # that as an exercise for the future...)
        num_tokens_per_sequence = valid_locs.sum(dim=1)  # [B]
        num_correct = all_preds_together == all_output_sequences  # [B, S']
        num_correct[torch.logical_not(valid_locs)] = 0  # [B, S']
        num_correct_per_seqence = num_correct.sum(dim=1)  # [B]
        accuracy_per_sequence = num_correct_per_seqence / num_tokens_per_sequence
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
