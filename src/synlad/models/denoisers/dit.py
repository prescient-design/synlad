"""From https://github.com/facebookresearch/all-atom-diffusion-transformer/blob/main/src/models/denoisers/dit.py

Copyright (c) Meta Platforms, Inc. and affiliates.
This code is released under the CC-BY-NC License -- see https://github.com/facebookresearch/all-atom-diffusion-transformer for further details.
"""

import math

import torch
import torch.nn as nn

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_dim, frequency_embedding_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_dim, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, 0, labels)
        # NOTE: 0 is the label for the null class
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def get_pos_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine poDiTional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: embedding dimension
        max_len: maximum length

    Returns:
        poDiTional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


#################################################################################
#                               Transformer blocks                              #
#################################################################################


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=0, bias=True, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(_x, _x, _x, key_padding_mask=mask, need_weights=False)[0]
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTBlockWithConditioning(nn.Module):
    """A DiT block MODIFIED to use cross-attention for conditioning."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # Self-Attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=0, bias=True, batch_first=True
        )

        # ADDED: Cross-Attention
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=0, bias=True, batch_first=True
        )

        # MLP
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )

        # AdaLN modulation now produces 8 chunks (for 3 norms + 1 attn gate)
        # scale/shift for norm1, norm2, norm3 and a gate for self-attn and mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 8 * hidden_dim, bias=True)
        )

    def forward(self, x, t, cond_seq, x_mask=None, cond_mask=None):
        # t is the time embedding (B, d), cond_seq is the pharmacophore embedding (B, N_cond, d)

        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_cross,
            scale_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(t).chunk(8, dim=1)

        # 1. Self-Attention
        x_sa = modulate(self.norm1(x), shift_sa, scale_sa)
        sa_out, _ = self.attn(x_sa, x_sa, x_sa, key_padding_mask=x_mask, need_weights=False)
        x = x + gate_sa.unsqueeze(1) * sa_out

        # 2. Cross-Attention
        x_cross = modulate(self.norm2(x), shift_cross, scale_cross)
        # Query from x, Key/Value from the conditioning sequence
        cross_attn_out, _ = self.cross_attn(
            query=x_cross,
            key=cond_seq,
            value=cond_seq,
            key_padding_mask=cond_mask,
            need_weights=False,
        )
        # Note: No gate on cross-attention is a common choice, but you could add one.
        x = x + cross_attn_out

        # 3. MLP
        x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)

        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """Diffusion model with a Transformer backbone.

    Args:
        d_x (int): Input dimension
        d_model (int): Model dimension
        num_layers (int): Number of Transformer layers
        nhead (int): Number of attention heads
        mlp_ratio (float): Ratio of hidden to input dimension in MLP
        use_conditioning (bool): Whether to use conditioning. If False, only timestep embedding is used.

    Note:
        When use_conditioning=False, the model ignores conditioning inputs and uses only timestep information.
        This is useful for unconditional generation or testing without conditioning data.
    """

    def __init__(
        self,
        d_x=8,
        d_model=384,
        num_layers=12,
        nhead=6,
        mlp_ratio=4.0,
        use_conditioning=False,
        dropout_prob=0.2,
    ):
        super().__init__()
        self.d_x = d_x
        self.d_model = d_model
        self.nhead = nhead
        self.use_conditioning = use_conditioning
        self.dropout_prob = dropout_prob

        self.x_embedder = nn.Linear(2 * d_x, d_model, bias=True)
        self.t_embedder = TimestepEmbedder(d_model)

        # Only create conditioning embedder if needed
        if self.use_conditioning:
            self.blocks = nn.ModuleList(
                [
                    DiTBlockWithConditioning(d_model, nhead, mlp_ratio=mlp_ratio)
                    for _ in range(num_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
            )
        self.final_layer = FinalLayer(d_model, d_x)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def cond_drop(self, cond, cond_mask=None, force_drop_ids=None):
        """Drops conditioning sequences to enable classifier-free guidance.

        Args:
            cond (torch.Tensor): Conditioning sequences (B, N_cond, d_cond)
            cond_mask (torch.Tensor): Mask for conditioning sequences (B, N_cond)
            force_drop_ids (torch.Tensor): Force specific samples to be dropped

        Returns:
            tuple: (dropped_cond, dropped_cond_mask) where dropped samples are zeroed
        """
        if self.dropout_prob == 0:
            return cond, cond_mask

        batch_size = cond.shape[0]

        # Determine which samples to drop
        if force_drop_ids is None:
            drop_ids = torch.rand(batch_size, device=cond.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1

        # Create null conditioning for dropped samples
        dropped_cond = torch.where(
            drop_ids[:, None, None],  # Broadcast to (B, 1, 1)
            torch.zeros_like(cond),  # Null conditioning
            cond,  # Original conditioning
        )

        # For dropped samples, mask all conditioning tokens (all True = all invalid)
        if cond_mask is not None:
            dropped_cond_mask = torch.where(
                drop_ids[:, None],  # Broadcast to (B, 1)
                torch.ones_like(cond_mask),
                cond_mask,  # Original mask
            )
        else:
            dropped_cond_mask = cond_mask

        return dropped_cond, dropped_cond_mask

    def forward(self, x, t, mask=None, x_sc=None, cond=None, cond_mask=None):
        """Forward pass of DiT.

        Args:
            x (torch.Tensor): Input data tensor (B, N, d_in)
            t (torch.Tensor): Time step for each sample (B,)
            cond (torch.Tensor, optional): Conditioning for each sample (B,)
            mask (torch.Tensor): True if valid token, False if padding (B, N)
            x_sc (torch.Tensor): Self-conditioning x (B, N, d_in)
        """
        # Positonal embedding
        token_index = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
        pos_emb = get_pos_embedding(token_index, self.d_model)

        # Self-conditioning and input embeddings: (B, N, d)
        if x_sc is None:
            x_sc = torch.zeros_like(x)
        x = self.x_embedder(torch.cat([x, x_sc], dim=-1)) + pos_emb

        # Conditioning embeddings
        t = self.t_embedder(t.squeeze(1))  # (B, d)

        # Apply conditioning dropout for CFG training
        if self.use_conditioning and cond is not None and self.training:
            cond, cond_mask = self.cond_drop(cond, cond_mask)

        # Transformer blocks
        if self.use_conditioning:
            for block in self.blocks:
                x = block(
                    x=x,
                    t=t,
                    cond_seq=cond,
                    x_mask=~mask,  # true=masked
                    cond_mask=~cond_mask if cond_mask is not None else None,
                )
        else:
            for block in self.blocks:
                x = block(x, t, ~mask)  # (B, N, d)

        # Prediction layer
        x = self.final_layer(x, t)  # (B, N, d_out)
        x = x * mask[..., None]
        return x

    def forward_with_cfg(
        self, x, t, cond=None, mask=None, cfg_scale=1.0, x_sc=None, cond_mask=None
    ):
        """Forward pass of DiT, but also batches the unconditional forward pass for classifier-free
        guidance.

        Assumes batch x's and conditioning sequences are ordered such that the first half are the conditional
        samples and the second half are the unconditional samples.
        """
        half_x = x[: len(x) // 2]
        combined_x = torch.cat([half_x, half_x], dim=0)

        # Handle conditioning for CFG
        if self.use_conditioning and cond is not None:
            half_cond = cond[: len(cond) // 2]
            # Create null conditioning for unconditional path
            null_cond = torch.zeros_like(half_cond)
            combined_cond = torch.cat([half_cond, null_cond], dim=0)

            # Handle conditioning mask
            if cond_mask is not None:
                half_cond_mask = cond_mask[: len(cond_mask) // 2]
                # All tokens masked for null conditioning
                null_cond_mask = torch.ones_like(half_cond_mask)
                combined_cond_mask = torch.cat([half_cond_mask, null_cond_mask], dim=0)
            else:
                combined_cond_mask = None
        else:
            combined_cond = cond
            combined_cond_mask = cond_mask

        # Handle other inputs
        if mask is not None:
            half_mask = mask[: len(mask) // 2]
            combined_mask = torch.cat([half_mask, half_mask], dim=0)
        else:
            combined_mask = mask

        if x_sc is not None:
            half_x_sc = x_sc[: len(x_sc) // 2]
            combined_x_sc = torch.cat([half_x_sc, half_x_sc], dim=0)
        else:
            combined_x_sc = x_sc

        model_out = self.forward(
            combined_x, t, combined_mask, combined_x_sc, combined_cond, combined_cond_mask
        )

        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps
