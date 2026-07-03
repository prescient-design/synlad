"""Adapted from https://github.com/facebookresearch/all-atom-diffusion-transformer/blob/main/src/models/vae_module.py

Copyright (c) Meta Platforms, Inc. and affiliates.
This code is released under the CC-BY-NC License -- see https://github.com/facebookresearch/all-atom-diffusion-transformer for further details.
"""

import copy
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter
from torchmetrics import MeanMetric

from synlad.eval.molecule_reconstruction import MoleculeReconstructionEvaluator
from synlad.eval.synthesis_generation import SynthesisGenerationEvaluator
from synlad.eval.synthesis_reconstruction import SynthesisEvaluator
from synlad.models.components import synthesis_pipelines as pipelines
from synlad.models.components import synthesis_rxn_predictor as rxn_predictor
from synlad.models.components import synthesis_token_action_handler as token_action_handler
from synlad.models.components.kabsch_utils import random_rotation_matrix
from synlad.tokenization import synthesis_serialization as serialization
from synlad.tokenization import synthesis_vocab as vocab
from synlad.utils import pylogger

log = pylogger.RankedLogger(__name__)


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution with mean and logvar parameters.

    Adapted from: https://github.com/CompVis/latent-diffusion, with modifications for our tensors,
    which are of shape (N, d) instead of (B, H, W, d) for 2D images.
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)  # split along channel dim
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=1
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=1,
                )

    def mode(self):
        return self.mean

    def __repr__(self):
        return f"DiagonalGaussianDistribution(mean={self.mean}, logvar={self.logvar})"


class VariationalAutoencoderLitModule(LightningModule):
    """LightningModule for autoencoding 3D molecules and synthesis pathways. Implements VAE loss."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoders: dict[str, torch.nn.Module],
        latent_dim: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: int,
        loss_weights: dict,
        augmentations: DictConfig,
        visualization: DictConfig,
        compile: bool,
        full_synthesis_accuracies: bool = False,
        provide_synthesis_examples: bool = False,
        detach_synthesis_from_encoder: bool = False,
        use_molecule_decoder: bool = True,
        rxn_server_url: str = "http://127.0.0.1:8000/predict",
        vocab_json: str | None = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        # encoder and decoder models
        self.encoder = encoder
        self.decoders = torch.nn.ModuleDict(decoders)

        # quantization layers (following naming convention from Latent Diffusion)
        self.quant_conv = torch.nn.Linear(self.encoder.d_model, 2 * latent_dim, bias=False)  # type: ignore
        if use_molecule_decoder:
            self.post_quant_conv = torch.nn.Linear(
                latent_dim, decoders["molecule_decoder"].d_model, bias=False
            )
        else:
            self.post_quant_conv = None

        # weights for scaling loss functions
        self.loss_weights = loss_weights
        # Extract single weight values (assuming config provides single values, not dict of datasets)
        if isinstance(loss_weights["loss_atom_types"], dict):
            # If still a dict, use first value
            self.weight_atom_types = list(loss_weights["loss_atom_types"].values())[0]
            self.weight_pos = list(loss_weights["loss_pos"].values())[0]
            self.weight_kl = list(loss_weights["loss_kl"].values())[0]
            # Handle backwards compatibility for sigma_sq (default to 1.0 if not present)
            self.sigma_sq = list(loss_weights.get("sigma_sq", {1: 1.0}).values())[0]
        else:
            # If already scalar values
            self.weight_atom_types = loss_weights["loss_atom_types"]
            self.weight_pos = loss_weights["loss_pos"]
            self.weight_kl = loss_weights["loss_kl"]
            # Handle backwards compatibility for sigma_sq (default to 1.0 if not present)
            self.sigma_sq = loss_weights.get("sigma_sq", 1.0)

        self.weight_synthesis = loss_weights["loss_synthesis"]
        if loss_weights.get("lambda_weight") is not None:
            lambda_weight = loss_weights["lambda_weight"]
            self.weight_pos = lambda_weight
            self.weight_atom_types = lambda_weight
            self.weight_synthesis = 1.0 - lambda_weight

        # evaluators
        self.reconstruction_evaluator = MoleculeReconstructionEvaluator()
        self.synthesis_evaluator = SynthesisEvaluator(
            full_accuracies=full_synthesis_accuracies, provide_examples=provide_synthesis_examples
        )
        self.test_synthesis_evaluator = SynthesisGenerationEvaluator()

        # metric objects for calculating and averaging across batches
        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "loss_atom_types": MeanMetric(),
                "loss_pos": MeanMetric(),
                "loss_kl": MeanMetric(),
                "loss_synthesis_mean": MeanMetric(),
                "unscaled/loss_atom_types": MeanMetric(),
                "unscaled/loss_pos": MeanMetric(),
                "unscaled/loss_kl": MeanMetric(),
                "unscaled/loss_synthesis_mean": MeanMetric(),
            }
        )
        self.val_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "loss_atom_types": MeanMetric(),
                "loss_pos": MeanMetric(),
                "loss_kl": MeanMetric(),
                "unscaled/loss_atom_types": MeanMetric(),
                "unscaled/loss_pos": MeanMetric(),
                "unscaled/loss_kl": MeanMetric(),
                "match_rate": MeanMetric(),
                "rms_dist": MeanMetric(),
                "loss_synthesis_mean": MeanMetric(),
                "unscaled/loss_synthesis_mean": MeanMetric(),
                "token_accuracy": MeanMetric(),
                "sequence_accuracy": MeanMetric(),
            }
        )
        if self.hparams.full_synthesis_accuracies:
            self.val_metrics.update(
                {
                    "action_cond_accuracy": MeanMetric(),
                    "action_avg_num_per_seq": MeanMetric(),
                    "total_avg_num_per_seq": MeanMetric(),
                    "BB-ID_cond_accuracy": MeanMetric(),
                    "BB-ID_avg_num_per_seq": MeanMetric(),
                    "BB-ID_cond_accuracy_invariant": MeanMetric(),
                    "FWD-SELECTION_cond_accuracy": MeanMetric(),
                    "FWD-SELECTION_avg_num_per_seq": MeanMetric(),
                    "FWD-SELECTION_cond_accuracy_invariant": MeanMetric(),
                }
            )

        self.test_metrics = copy.deepcopy(self.val_metrics)

        self.vocab_json = vocab_json
        self.rxn_server_url = rxn_server_url
        self.synthesis_pipeline = None

    def _get_synthesis_pipeline(self, vocab_json: str, rxn_server_url: str):
        rxn_pred_server = rxn_predictor.RxnLM(rxn_server_url, log)
        rxn_pred_server = rxn_predictor.CachedReactionPredictor(rxn_pred_server)

        with open(Path(self.vocab_json)) as f:
            vocab_dict = json.load(f)
        tkn_lib_collection = vocab.TokenLibraryCollection(
            [
                vocab.TokenLibrary.from_dict(vocab_dict["special_token_lib"]),
                vocab.TokenLibrary.from_dict(vocab_dict["bb_token_lib"]),
                vocab.TokenLibrary(
                    tokens=[],
                    start_idx=len(vocab_dict["special_token_lib"]["tokens"])
                    + len(vocab_dict["bb_token_lib"]["tokens"]),
                    molecule_tokens=True,
                ),
            ]
        )
        serializer = serialization.Serializer(0.0)
        tah = token_action_handler.TokenActionHandler(
            serializer=serializer, rxn_predictor=rxn_pred_server
        )
        # scorer_ = pipelines.get_score_normalizer(normalizer=pipelines.ScoreNormalizer.LENGTH)
        scorer_ = pipelines.get_score_normalizer(normalizer=pipelines.ScoreNormalizer.NONE)

        # Get generator and pipeline
        # Handle backward compatibility: use default "sample" if key doesn't exist
        synthesis_inference_method = getattr(self.hparams, "synthesis_inference_method", "sample")
        log.info(f"Using synthesis inference method: {synthesis_inference_method}")

        if synthesis_inference_method == "sample":
            method = pipelines.GeneratorMethods.SAMPLE
        elif synthesis_inference_method == "beam_search":
            method = pipelines.GeneratorMethods.BEAM_SEARCH
        else:
            raise ValueError(f"Invalid synthesis inference method: {synthesis_inference_method}.")
        generator = pipelines.get_generator(
            model=self.decoders["synthesis_decoder"],
            token_action_handler=tah,
            device=self.device,
            method=method,
            beam_width=self.hparams.beam_width,
            warper=self.hparams.synthesis_warper,
        )
        pipeline = pipelines.Pipeline(
            generator=generator,
            tkn_lib_collection=tkn_lib_collection,
            device=self.device,
            score_normalizer=scorer_,
            deserializer=serialization.Deserializer(),
            top_k=self.hparams.synthesis_top_k,
        )

        return pipeline

    def encode(self, batch):
        # for the encoding, we only need to pass the molecules
        batch_molecules = batch["molecules"]
        encoded_batch = self.encoder(batch_molecules)
        encoded_batch["moments"] = self.quant_conv(encoded_batch["x"])
        encoded_batch["posterior"] = DiagonalGaussianDistribution(encoded_batch["moments"])
        return encoded_batch

    def decode(self, encoded_batch, batch):
        outputs = {}
        # For molecule decoder:
        if self.hparams.use_molecule_decoder:
            decoder_encoded_batch = encoded_batch.copy()
            decoder_encoded_batch["x"] = self.post_quant_conv(encoded_batch["x"])
            outputs["molecule_decoder"] = self.decoders["molecule_decoder"](decoder_encoded_batch)

        if self.hparams.detach_synthesis_from_encoder:
            # DETACH gradients to prevent synthesis loss from backpropagating to encoder
            encoded_x = encoded_batch["x"].detach()
        else:
            encoded_x = encoded_batch["x"]

        x_dense, atom_mask = to_dense_batch(encoded_x, encoded_batch["batch"])
        conditioning = x_dense  # [num_molecules, max_atoms, latent_dim]

        outputs["synthesis_decoder"] = self.decoders["synthesis_decoder"](
            batch=batch["pathways"], conditioning=conditioning, conditioning_mask=atom_mask
        )[1]  # Get logits from the tuple return
        return outputs

    def decode_at_inference(self, encoded_batch):
        outputs = {}
        # For molecule decoder:
        if self.hparams.use_molecule_decoder:
            decoder_encoded_batch = encoded_batch.copy()
            decoder_encoded_batch["x"] = self.post_quant_conv(encoded_batch["x"])
            outputs["molecule_decoder"] = self.decoders["molecule_decoder"](decoder_encoded_batch)

        # For synthesis decoder:
        # Synthesis pipeline handled in LDM module
        x_dense, atom_mask = to_dense_batch(encoded_batch["x"], encoded_batch["batch"])
        conditioning = x_dense  # [num_molecules, max_atoms, latent_dim]

        return outputs, conditioning, atom_mask

    def forward(self, batch: Data, sample_posterior: bool = True):
        encoded_batch = self.encode(batch)
        if sample_posterior:
            encoded_batch["x"] = encoded_batch["posterior"].sample()
        else:
            encoded_batch["x"] = encoded_batch["posterior"].mode()
        outputs = self.decode(encoded_batch, batch)
        return outputs, encoded_batch

    def forward_at_inference(self, batch: Data):
        encoded_batch = self.encode(batch)
        encoded_batch["x"] = encoded_batch["posterior"].mode()
        outputs, conditioning, conditioning_mask = self.decode_at_inference(encoded_batch)
        outputs["synthesis_decoder"] = self.generate_synthesis(conditioning, conditioning_mask)
        return outputs, encoded_batch

    def generate_synthesis(self, conditioning, conditioning_mask):
        # Ensure synthesis pipeline is initialized
        if self.synthesis_pipeline is None:
            self.synthesis_pipeline = self._get_synthesis_pipeline(
                self.vocab_json, self.rxn_server_url
            )

        batch_size = conditioning.shape[0]
        initial_sequences = [["<BB>"] for _ in range(batch_size)]
        mol_scorers = [lambda x: 1.0 for _ in range(batch_size)]
        out = self.synthesis_pipeline.run(
            initial_sequences=initial_sequences,
            mol_scorers=mol_scorers,
            conditioning=conditioning,
            conditioning_mask=conditioning_mask,
        )
        return out

    #####################################################################################################

    def reconstruction_criterion(
        self, molecules_batch: Data, out: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Atom types loss
        loss_atom_types = F.cross_entropy(
            out["atom_types"], molecules_batch.atom_types, reduction="none"
        )

        # Coordinates loss after zero-centering, use nm as unit (not A)
        pos_pred = out["pos"]
        pos_true = molecules_batch.pos / 10.0  # A to nm
        pos_mean_pred = scatter(pos_pred, molecules_batch.batch, dim=0, reduce="mean")[
            molecules_batch.batch
        ]
        pos_mean_true = scatter(pos_true, molecules_batch.batch, dim=0, reduce="mean")[
            molecules_batch.batch
        ]
        loss_pos = F.mse_loss(
            pos_pred - pos_mean_pred, pos_true - pos_mean_true, reduction="none"
        ).mean(dim=1)

        return {
            "loss_atom_types": loss_atom_types,
            "loss_pos": loss_pos,
        }

    def compute_synthesis_loss(self, pathways_batch, logits):
        """Compute synthesis loss for the synthesis decoder."""
        if pathways_batch.input_sequences.is_nested:
            all_logits_together = torch.cat(logits.unbind(), dim=0)
            all_output_sequences = torch.cat(pathways_batch.output_sequences.unbind(), dim=0)
            valid_locs = torch.cat(pathways_batch.output_from_model_masks.unbind(), dim=0)
            all_logits_together = all_logits_together[valid_locs]
            all_output_sequences = all_output_sequences[valid_locs]
        else:
            valid_locs = torch.logical_and(
                pathways_batch.input_nonpad_masks, pathways_batch.output_from_model_masks
            )
            all_logits_together = logits[valid_locs]
            all_output_sequences = pathways_batch.output_sequences[valid_locs]

        assert len(all_logits_together.shape) - 1 == len(all_output_sequences.shape)
        loss = F.cross_entropy(
            input=all_logits_together, target=all_output_sequences, reduction="mean"
        )
        return loss

    def criterion(
        self, batch: Data, encoded_batch: dict[str, torch.Tensor], out: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # KL divergence loss
        loss_kl = encoded_batch["posterior"].kl()

        # Synthesis loss
        loss_synthesis_mean = self.compute_synthesis_loss(
            batch["pathways"], out["synthesis_decoder"]
        )

        # Apply loss weights
        weights_kl = self.weight_kl
        weights_synthesis = self.weight_synthesis

        result = {
            "loss_kl": weights_kl * loss_kl,
            "loss_synthesis_mean": weights_synthesis * loss_synthesis_mean,
            "unscaled/loss_kl": loss_kl,
            "unscaled/loss_synthesis_mean": loss_synthesis_mean,
        }

        loss = (weights_kl * loss_kl).mean() + (weights_synthesis * loss_synthesis_mean)

        if self.hparams.use_molecule_decoder:
            # Reconstruction loss for molecules
            loss_reconst = self.reconstruction_criterion(
                batch["molecules"], out["molecule_decoder"]
            )
            weights_atom_types = self.weight_atom_types
            weights_pos = self.weight_pos
            sigma_sq = self.sigma_sq

            loss += (weights_atom_types * loss_reconst["loss_atom_types"]).mean()
            loss += (weights_pos * loss_reconst["loss_pos"] / sigma_sq).mean()

            result.update(
                {
                    "loss_atom_types": weights_atom_types * loss_reconst["loss_atom_types"],
                    "loss_pos": weights_pos * loss_reconst["loss_pos"] / sigma_sq,
                    "unscaled/loss_atom_types": loss_reconst["loss_atom_types"],
                    "unscaled/loss_pos": loss_reconst["loss_pos"],
                }
            )

        result["loss"] = loss
        return result

    #####################################################################################################

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for metric in self.val_metrics.values():
            metric.reset()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch starts."""
        for metric in self.train_metrics.values():
            metric.reset()

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        with torch.no_grad():
            # Apply augmentations to molecules
            molecules_batch = batch["molecules"]

            if self.hparams.augmentations.pos:
                rot_mat = random_rotation_matrix(validate=True, device=self.device)
                pos_aug = molecules_batch.pos @ rot_mat.T
                molecules_batch.pos = pos_aug

            if self.hparams.augmentations.noise > 0.0:
                total_atoms = molecules_batch.num_atoms.sum().item()
                # select X% of atom types to be perturbed
                perturbed_idx = torch.tensor(
                    np.random.choice(
                        total_atoms,
                        int(total_atoms * self.hparams.augmentations.noise),
                        replace=False,
                    ),
                    device=self.device,
                )
                # save original atom types
                atom_types_ = molecules_batch.atom_types.clone()
                # set perturbed atom types to 0
                molecules_batch.atom_types[perturbed_idx] = 0

                # select X% of positions to be perturbed (new, may overlap)
                perturbed_idx = torch.tensor(
                    np.random.choice(
                        total_atoms,
                        int(total_atoms * self.hparams.augmentations.noise),
                        replace=False,
                    ),
                    device=self.device,
                )
                # save original positions
                pos_ = molecules_batch.pos.clone()
                # add random noise to perturbed positions
                corruption_scale = 0.1
                noise = (
                    torch.randn_like(molecules_batch.pos[perturbed_idx], device=self.device)
                    * corruption_scale
                )
                molecules_batch.pos[perturbed_idx] += noise

        # forward pass
        outputs, encoded_batch = self.forward(batch)

        # undo noise augmentation before calculating loss
        if self.hparams.augmentations.noise > 0.0:
            molecules_batch.atom_types = atom_types_
            molecules_batch.pos = pos_

        # Note: No need to reassign batch["molecules"] since we modified the same object

        # calculate loss
        loss_dict = self.criterion(batch, encoded_batch, outputs)

        # update and log train metrics
        for k, v in loss_dict.items():
            self.train_metrics[k](v)
            self.log(
                f"train/{k}",
                self.train_metrics[k],
                on_step=True,
                on_epoch=False,
                prog_bar=False if k != "loss" else True,
            )

        # return loss or backpropagation will fail
        return loss_dict["loss"]

    #####################################################################################################

    def on_validation_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="val")

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        self.evaluation_step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="val")

    #####################################################################################################

    def on_test_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="test")

    def test_step(self, batch: Data, batch_idx: int) -> None:
        outputs, encoded_batch = self.forward_at_inference(batch)

        # Save molecular predictions for reconstruction metrics
        if self.hparams.use_molecule_decoder:
            mol_out = outputs["molecule_decoder"]
            molecules_batch = batch["molecules"]

            start_idx = 0
            num_atoms_list = molecules_batch.num_atoms.tolist()

            for idx_in_batch, num_atom in enumerate(num_atoms_list):
                _atom_types = mol_out["atom_types"].narrow(0, start_idx, num_atom).argmax(dim=1)
                _pos = mol_out["pos"].narrow(0, start_idx, num_atom) * 10.0  # nm to A
                self.reconstruction_evaluator.append_pred_array(
                    {
                        "atom_types": _atom_types.detach().cpu().numpy(),
                        "pos": _pos.detach().cpu().numpy(),
                        "sample_idx": (batch_idx + self.global_rank) * molecules_batch.batch_size
                        + idx_in_batch,
                    }
                )
                start_idx = start_idx + num_atom

            # Save molecular ground truths
            data_list = molecules_batch.to_data_list()

            for idx_in_batch, _data in enumerate(data_list):
                self.reconstruction_evaluator.append_gt_array(
                    {
                        "atom_types": _data["atom_types"].detach().cpu().numpy(),
                        "pos": _data["pos"].detach().cpu().numpy(),
                        "sample_idx": (batch_idx + self.global_rank) * molecules_batch.batch_size
                        + idx_in_batch,
                    }
                )

        pathways_out = outputs["synthesis_decoder"]
        molecules_batch = batch["molecules"]
        for idx_in_batch in range(len(pathways_out)):
            # Store synthesis data without target molecule for now
            # Target molecules will be extracted after molecule evaluator processes them
            pred_data = {
                "synthesis_predictions": pathways_out[idx_in_batch],
                "target_molecule": None,  # Will be updated after molecule evaluation
                "sample_idx": (batch_idx + self.global_rank) * molecules_batch.batch_size
                + idx_in_batch,
            }
            self.test_synthesis_evaluator.append_pred_data(pred_data)

    def on_test_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="test")

    #####################################################################################################

    def on_evaluation_epoch_start(self, stage: Literal["val", "test"]) -> None:
        """Lightning hook that is called when a validation/test epoch starts."""
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        for metric in metrics.values():
            metric.reset()
        self.reconstruction_evaluator.clear()
        self.synthesis_evaluator.clear()
        if stage == "test":
            self.test_synthesis_evaluator.clear()
            # Reset synthesis pipeline to ensure it uses the current decoder weights
            self.synthesis_pipeline = None

    def evaluation_step(
        self,
        batch: Data,
        batch_idx: int,
        stage: Literal["val", "test"],
    ) -> None:
        """Perform a single evaluation step on a batch of data from the validation/test set."""
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")

        metrics = getattr(self, f"{stage}_metrics")

        # forward pass
        outputs, encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(batch, encoded_batch, outputs)

        # update and log per-step metrics
        for k, v in loss_dict.items():
            metrics[k](v)
            self.log(
                f"{stage}/{k}",
                metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        # Save molecular predictions for reconstruction metrics
        if self.hparams.use_molecule_decoder:
            mol_out = outputs["molecule_decoder"]
            molecules_batch = batch["molecules"]

            start_idx = 0
            num_atoms_list = molecules_batch.num_atoms.tolist()

            for idx_in_batch, num_atom in enumerate(num_atoms_list):
                _atom_types = mol_out["atom_types"].narrow(0, start_idx, num_atom).argmax(dim=1)
                _pos = mol_out["pos"].narrow(0, start_idx, num_atom) * 10.0  # nm to A
                self.reconstruction_evaluator.append_pred_array(
                    {
                        "atom_types": _atom_types.detach().cpu().numpy(),
                        "pos": _pos.detach().cpu().numpy(),
                        "sample_idx": (batch_idx + self.global_rank) * molecules_batch.batch_size
                        + idx_in_batch,
                    }
                )
                start_idx = start_idx + num_atom

            # Save molecular ground truths
            data_list = molecules_batch.to_data_list()

            for idx_in_batch, _data in enumerate(data_list):
                self.reconstruction_evaluator.append_gt_array(
                    {
                        "atom_types": _data["atom_types"].detach().cpu().numpy(),
                        "pos": _data["pos"].detach().cpu().numpy(),
                        "sample_idx": (batch_idx + self.global_rank) * molecules_batch.batch_size
                        + idx_in_batch,
                    }
                )

        pathways_out = outputs["synthesis_decoder"]
        pathways_batch = batch["pathways"]
        self.synthesis_evaluator.update(pathways_batch, pathways_out, batch_idx, self.global_rank)

    def on_evaluation_epoch_end(self, stage: Literal["val", "test"]) -> None:
        """Lightning hook that is called when a validation/test epoch ends."""
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")

        if stage == "val":
            metrics = getattr(self, f"{stage}_metrics")

            combined_metrics = {}

            if self.hparams.use_molecule_decoder:
                # Compute molecular reconstruction metrics
                rec_metrics_dict = self.reconstruction_evaluator.get_metrics(
                    save=self.hparams.visualization.visualize,
                    save_dir=self.hparams.visualization.save_dir
                    + f"/molecules_{stage}_{self.global_rank}",
                    device=metrics["loss"].device,
                )

                combined_metrics.update(rec_metrics_dict)

            # Compute synthesis metrics
            synthesis_metrics = self.synthesis_evaluator.compute_metrics()

            # Combine synthesis and reconstruction metrics
            combined_metrics.update(synthesis_metrics)

            for k, v in combined_metrics.items():
                metrics[k](v)
                self.log(
                    f"{stage}/{k}",
                    metrics[k],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True if k == "match_rate" else False,
                    sync_dist=True,
                )

            # Log visualizations if enabled (with frequency control to save space)
            visualization_frequency = self.hparams.visualization.log_frequency
            max_synthesis_examples = self.hparams.visualization.max_synthesis_examples

            should_visualize = (
                self.hparams.visualization.visualize
                and isinstance(self.logger, WandbLogger)
                and (self.current_epoch % visualization_frequency == 0 or self.current_epoch == 0)
            )

            if should_visualize and self.hparams.use_molecule_decoder:
                # Get synthesis examples
                synthesis_examples = self.synthesis_evaluator.get_examples()

                if synthesis_examples:
                    # Create combined table with both molecular and synthesis data
                    combined_table = self.reconstruction_evaluator.get_combined_wandb_table(
                        synthesis_examples=synthesis_examples,
                        current_epoch=self.current_epoch,
                        save_dir=self.hparams.visualization.save_dir
                        + f"/molecules_{stage}_{self.global_rank}",
                        max_examples=max_synthesis_examples,
                    )
                    self.logger.experiment.log(
                        {f"combined_{stage}_pred_table_device{self.global_rank}": combined_table}
                    )
                else:
                    # Fallback to molecular predictions table only if no synthesis examples
                    pred_table = self.reconstruction_evaluator.get_wandb_table(
                        current_epoch=self.current_epoch,
                        save_dir=self.hparams.visualization.save_dir
                        + f"/molecules_{stage}_{self.global_rank}",
                    )
                    self.logger.experiment.log(
                        {f"molecules_{stage}_pred_table_device{self.global_rank}": pred_table}
                    )

    #####################################################################################################

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or predict."""
        if self.hparams.compile and stage == "fit":
            self.encode = torch.compile(self.encode)
            self.decode = torch.compile(self.decode)
            self.quant_conv = torch.compile(self.quant_conv)
            if self.hparams.use_molecule_decoder:
                self.post_quant_conv = torch.compile(self.post_quant_conv)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            # Choose monitor based on whether molecule decoder is used
            monitor_key = "val/match_rate"
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor_key,
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}
