"""Adapted from https://github.com/facebookresearch/all-atom-diffusion-transformer/blob/main/src/models/ldm_module.py

Copyright (c) Meta Platforms, Inc. and affiliates.
This code is released under the CC-BY-NC License -- see https://github.com/facebookresearch/all-atom-diffusion-transformer for further details.
"""

import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric
from tqdm import tqdm

from synlad.eval.molecule_generation import MoleculeGenerationEvaluator
from synlad.eval.synthesis_generation import SynthesisGenerationEvaluator
from synlad.models.components import synthesis_pipelines as pipelines
from synlad.models.components import synthesis_rxn_predictor as rxn_predictor
from synlad.models.components import synthesis_token_action_handler as token_action_handler
from synlad.models.components.kabsch_utils import random_rotation_matrix
from synlad.models.vae_module import VariationalAutoencoderLitModule
from synlad.tokenization import synthesis_serialization as serialization
from synlad.tokenization import synthesis_vocab as vocab
from synlad.utils import pylogger

log = pylogger.RankedLogger(__name__)


class LatentDiffusionLitModule(LightningModule):
    """LightningModule for latent diffusion generative modellling of 3D atomic systems."""

    def __init__(
        self,
        autoencoder_ckpt: str,
        vocab_json: str,
        denoiser: torch.nn.Module,
        interpolant: DictConfig,
        augmentations: DictConfig,
        sampling: DictConfig,
        conditioning: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: str,
        compile: bool,
        rxn_server_url: str | None = None,
        ph4_encoder: torch.nn.Module = None,
        ph4_dropout_enabled: bool = False,
        bincount_path: str = None,
        train_smiles_path: str = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # autoencoder models (first-stage model)
        self.autoencoder_ckpt = autoencoder_ckpt
        log.info(f"Loading Autoencoder ckpt: {autoencoder_ckpt}")
        self.autoencoder = VariationalAutoencoderLitModule.load_from_checkpoint(
            autoencoder_ckpt, map_location="cpu", weights_only=False
        )
        # freeze autoencoder
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()

        # pharmacophore encoder
        self.ph4_encoder = ph4_encoder

        # denoiser model (second-stage model)
        self.denoiser = denoiser

        # interpolant for diffusion or flow matching training/sampling
        self.interpolant = interpolant

        self.bincount_path = bincount_path
        self.train_smiles_path = train_smiles_path

        # evaluator objects for computing metrics
        self.val_generation_evaluator = MoleculeGenerationEvaluator(
            dataset_smiles_list=torch.load(self.train_smiles_path),
            removeHs=self.hparams.sampling.removeHs,
        )

        self.test_generation_evaluator = copy.deepcopy(self.val_generation_evaluator)

        # Synthesis generation evaluators (device will be set dynamically)
        self.val_synthesis_evaluator = SynthesisGenerationEvaluator()
        self.test_synthesis_evaluator = copy.deepcopy(self.val_synthesis_evaluator)

        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "x_loss": MeanMetric(),
                "x_loss t=[0,25)": MeanMetric(),
                "x_loss t=[25,50)": MeanMetric(),
                "x_loss t=[50,75)": MeanMetric(),
                "x_loss t=[75,100)": MeanMetric(),
                "t_avg": MeanMetric(),
            }
        )

        self.val_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "x_loss": MeanMetric(),
                "x_loss t=[0,25)": MeanMetric(),
                "x_loss t=[25,50)": MeanMetric(),
                "x_loss t=[50,75)": MeanMetric(),
                "x_loss t=[75,100)": MeanMetric(),
                "t_avg": MeanMetric(),
                "valid_rate": MeanMetric(),
                "unique_rate": MeanMetric(),
                "novel_rate": MeanMetric(),
                "mol_pred_loaded": MeanMetric(),
                "sanitization": MeanMetric(),
                "inchi_convertible": MeanMetric(),
                "all_atoms_connected": MeanMetric(),
                "bond_lengths": MeanMetric(),
                "bond_angles": MeanMetric(),
                "internal_steric_clash": MeanMetric(),
                "aromatic_ring_flatness": MeanMetric(),
                "double_bond_flatness": MeanMetric(),
                "internal_energy": MeanMetric(),
                "sampling_time": MeanMetric(),
                "posebusters_sum": MeanMetric(),
            }
        )
        if self.hparams.denoiser.use_conditioning:
            self.val_metrics["rocs_combo_score"] = MeanMetric()
            self.val_metrics["rocs_shape_score"] = MeanMetric()
            self.val_metrics["rocs_colour_score"] = MeanMetric()

        self.test_metrics = copy.deepcopy(self.val_metrics)

        bincount_data = torch.load(self.bincount_path, map_location="cpu").float()
        self.num_nodes_bincount = torch.nn.Parameter(bincount_data, requires_grad=False)

        # Store parameters for synthesis pipeline (will be initialized in setup())
        self.vocab_json = vocab_json
        self.rxn_server_url = rxn_server_url
        self.synthesis_pipeline = None
        if self.hparams.denoiser.use_conditioning:
            self.cond_projection = torch.nn.Linear(
                self.hparams.ph4_encoder.d_model, self.hparams.denoiser.d_model
            )
        else:
            self.cond_projection = None

    def _get_synthesis_pipeline(self, vocab_json: str, rxn_server_url: str):
        rxn_pred_server = rxn_predictor.RxnLM(rxn_server_url, log)
        rxn_pred_server = rxn_predictor.CachedReactionPredictor(rxn_pred_server)

        # Load the vocabulary
        with open(Path(vocab_json)) as f:
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
        scorer_ = pipelines.get_score_normalizer(normalizer=pipelines.ScoreNormalizer.NONE)

        # Get generator and pipeline
        # Handle backward compatibility: use default "sample" if key doesn't exist
        synthesis_inference_method = getattr(
            self.hparams.sampling, "synthesis_inference_method", "sample"
        )
        log.info(f"Using synthesis inference method: {synthesis_inference_method}")

        if synthesis_inference_method == "sample":
            method = pipelines.GeneratorMethods.SAMPLE
        elif synthesis_inference_method == "beam_search":
            method = pipelines.GeneratorMethods.BEAM_SEARCH
        else:
            raise ValueError(f"Invalid synthesis inference method: {synthesis_inference_method}.")
        generator = pipelines.get_generator(
            model=self.autoencoder.decoders.synthesis_decoder,
            token_action_handler=tah,
            device=self.device,
            method=method,
            beam_width=self.hparams.sampling.beam_width,
            warper=self.hparams.sampling.synthesis_warper,
        )
        pipeline = pipelines.Pipeline(
            generator=generator,
            tkn_lib_collection=tkn_lib_collection,
            device=self.device,
            score_normalizer=scorer_,
            deserializer=serialization.Deserializer(),
            top_k=self.hparams.sampling.synthesis_top_k,
        )

        return pipeline

    def forward(self, batch: Data, sample_posterior: bool = True):
        # Encode batch to latent space
        with torch.no_grad():
            encoded_batch = self.autoencoder.encode(batch)
            if sample_posterior:
                encoded_batch["x"] = encoded_batch["posterior"].sample()
            else:
                encoded_batch["x"] = encoded_batch["posterior"].mode()
            x_1 = encoded_batch["x"]

            # Convert from PyG batch to dense batch with padding
            x_1, mask = to_dense_batch(x_1, encoded_batch["batch"])
            dense_encoded_batch = {"x_1": x_1, "token_mask": mask, "diffuse_mask": mask}

        # Corrupt batch using the interpolant
        self.interpolant.device = dense_encoded_batch["x_1"].device
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        if self.hparams.denoiser.use_conditioning:
            pharmacophore_batch = batch["pharmacophores"]

            # Apply pharmacophore dropout during training only
            if self.hparams.ph4_dropout_enabled and self.training:
                keep_mask = torch.ones(
                    len(pharmacophore_batch.ph4_coords),
                    dtype=torch.bool,
                    device=pharmacophore_batch.ph4_coords.device,
                )

                # For each molecule, randomly drop 0 to (num_ph4_channels - 3) pharmacophores
                for mol_idx in torch.unique(pharmacophore_batch.batch):
                    mol_ph4_indices = torch.where(pharmacophore_batch.batch == mol_idx)[0]
                    max_droppable = max(0, len(mol_ph4_indices) - 3)
                    num_to_drop = random.randint(0, max_droppable)
                    if num_to_drop > 0:
                        drop_indices = torch.randperm(len(mol_ph4_indices))[:num_to_drop]
                        keep_mask[mol_ph4_indices[drop_indices]] = False

                # Filter all attributes that have one entry per pharmacophore
                pharmacophore_batch.ph4_coords = pharmacophore_batch.ph4_coords[keep_mask]
                pharmacophore_batch.ph4_channels = pharmacophore_batch.ph4_channels[keep_mask]
                pharmacophore_batch.token_idx = pharmacophore_batch.token_idx[keep_mask]

                # The 'batch' vector itself must also be filtered
                new_batch_vector = pharmacophore_batch.batch[keep_mask]
                pharmacophore_batch.batch = new_batch_vector

                # Recalculate the number of pharmacophores for each molecule
                # bincount counts the occurrences of each mol_idx in the new batch vector
                new_num_ph4s = torch.bincount(
                    new_batch_vector, minlength=pharmacophore_batch.num_graphs
                )
                pharmacophore_batch.num_ph4_channels = new_num_ph4s

            cond = self.ph4_encoder(pharmacophore_batch)
            cond_dense, cond_mask = to_dense_batch(cond["x"], cond["batch"])
            cond_dense = self.cond_projection(cond_dense)
        else:
            cond_dense, cond_mask = None, None

        # Use self-conditioning for ~half of the training batches
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=noisy_dense_encoded_batch["x_t"],
                    t=noisy_dense_encoded_batch["t"],
                    mask=mask,
                    x_sc=None,
                    cond=cond_dense,
                    cond_mask=cond_mask,
                )
        else:
            x_sc = None

        # Run denoiser model
        pred_x = self.denoiser(
            x=noisy_dense_encoded_batch["x_t"],
            t=noisy_dense_encoded_batch["t"],
            mask=mask,
            x_sc=x_sc,
            cond=cond_dense,
            cond_mask=cond_mask,
        )

        return pred_x, noisy_dense_encoded_batch

    def criterion(
        self,
        noisy_dense_encoded_batch: dict[str, torch.Tensor],
        pred_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Compute MSE loss w/ masking for padded tokens
        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}

        # add diffusion loss stratified across t
        num_bins = 4
        flat_losses = x_loss.detach().cpu().numpy().flatten()
        flat_t = noisy_dense_encoded_batch["t"].detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)
        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"x_loss t=[{int(bin_start * 100)},{int(bin_end * 100)})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            loss_dict[t_range] = range_loss
        loss_dict["t_avg"] = np.mean(flat_t)

        return loss_dict

    #####################################################################################################

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for metric in self.train_metrics.values():
            metric.reset()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch starts."""
        for metric in self.train_metrics.values():
            metric.reset()

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        with torch.no_grad():
            # save masks used to apply augmentations
            if self.hparams.augmentations.pos:
                molecules_batch = batch["molecules"]
                rot_mat = random_rotation_matrix(validate=True, device=self.device)
                pos_aug = molecules_batch.pos @ rot_mat.T
                molecules_batch.pos = pos_aug
                # rotate the pharmacophores
                if self.hparams.denoiser.use_conditioning:
                    pharmacophores_batch = batch["pharmacophores"]
                    pos_aug = pharmacophores_batch.ph4_coords @ rot_mat.T
                    pharmacophores_batch.ph4_coords = pos_aug
        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

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
        self.evaluation_step(batch, batch_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="test")

    #####################################################################################################

    def on_evaluation_epoch_start(self, stage: Literal["val", "test"]) -> None:
        "Lightning hook that is called when a validation/test epoch starts."
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        for metric in metrics.values():
            metric.reset()
        generation_evaluator = getattr(self, f"{stage}_generation_evaluator")
        generation_evaluator.clear()
        synthesis_evaluator = getattr(self, f"{stage}_synthesis_evaluator")
        synthesis_evaluator.clear()

        # Initialize list to store batches for later use in epoch_end
        if not hasattr(self, f"{stage}_batches"):
            setattr(self, f"{stage}_batches", [])
        getattr(self, f"{stage}_batches").clear()

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
        generation_evaluator = getattr(self, f"{stage}_generation_evaluator")
        generation_evaluator.device = metrics["loss"].device

        # Store the batch for later use in epoch_end
        batches = getattr(self, f"{stage}_batches")
        batches.append(batch)

        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

        # update and log per-step val metrics
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

    def on_evaluation_epoch_end(self, stage: Literal["val", "test"]) -> None:
        """Lightning hook that is called when a validation/test epoch ends."""

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        generation_evaluator = getattr(self, f"{stage}_generation_evaluator")
        synthesis_evaluator = getattr(self, f"{stage}_synthesis_evaluator")

        # Access stored validation batches
        batches = getattr(self, f"{stage}_batches")
        val_num_atoms = None

        if self.hparams.denoiser.use_conditioning and len(batches) > 0:
            # Use first validation batch for conditioning and num_nodes_bincount
            first_val_batch = batches[0]

            # Extract num_nodes_bincount from validation batch
            val_num_atoms = first_val_batch["molecules"]["num_atoms"]

            # Extract conditioning from validation batch pharmacophores
            if "pharmacophores" in first_val_batch:
                cond = self.ph4_encoder(first_val_batch["pharmacophores"])
                cond_dense, cond_mask = to_dense_batch(cond["x"], cond["batch"])
                cond_dense = self.cond_projection(cond_dense)
        else:
            cond_dense, cond_mask = None, None

        t_start = time.time()
        for samples_so_far in tqdm(
            range(0, self.hparams.sampling.num_samples, self.hparams.sampling.batch_size),
            desc="     Sampling",
        ):
            samples_remaining = self.hparams.sampling.num_samples - samples_so_far
            sampling_batch_size = min(self.hparams.sampling.batch_size, samples_remaining)
            cond_batch, cond_mask_batch = cond_dense, cond_mask
            sample_lengths = val_num_atoms
            if cond_dense is not None:
                sampling_batch_size = min(sampling_batch_size, cond_dense.shape[0])
                cond_batch = cond_dense[:sampling_batch_size]
                cond_mask_batch = cond_mask[:sampling_batch_size]
                sample_lengths = val_num_atoms[:sampling_batch_size]
            if sample_lengths is not None:
                # Generate random numbers in the range [num_atoms[i]-3, num_atoms[i]+6]
                min_atoms = torch.clamp(sample_lengths - 3, min=1)
                max_atoms = sample_lengths + 6
                sample_lengths = torch.randint_like(min_atoms, 0, 1).to(self.device)
                for i in range(len(min_atoms)):
                    sample_lengths[i] = torch.randint(
                        min_atoms[i], max_atoms[i] + 1, (1,), device=self.device
                    )
            out, batch, samples = self.sample_and_decode(
                num_nodes_bincount=self.num_nodes_bincount,
                sample_lengths=sample_lengths,
                batch_size=sampling_batch_size,
                cfg_scale=self.hparams.sampling.cfg_scale,
                cond=cond_batch,
                cond_mask=cond_mask_batch,
            )
            # Save predictions for metrics and visualisation
            start_idx = 0
            for idx_in_batch, num_atom in enumerate(batch["num_atoms"].tolist()):
                _atom_types = (
                    out["molecule_decoder"]["atom_types"]
                    .narrow(0, start_idx, num_atom)
                    .argmax(dim=1)
                )  # take argmax
                _pos = (
                    out["molecule_decoder"]["pos"].narrow(0, start_idx, num_atom) * 10.0
                )  # nm to A
                generation_evaluator.append_pred_array(
                    {
                        "atom_types": _atom_types.detach().cpu().numpy(),
                        "pos": _pos.detach().cpu().numpy(),
                        "sample_idx": samples_so_far
                        + self.global_rank * len(batch["num_atoms"])
                        + idx_in_batch,
                    }
                )
                start_idx = start_idx + num_atom

            # Log valid molecules for conditioning
            if self.hparams.denoiser.use_conditioning:
                validation_molecules = first_val_batch["molecules"]
                data_list = validation_molecules.to_data_list()
                # Only log up to sampling_batch_size molecules to match the generated samples
                max_molecules = min(len(data_list), sampling_batch_size)
                for idx_in_batch, _data in enumerate(data_list[:max_molecules]):
                    generation_evaluator.append_gt_array(
                        {
                            "atom_types": _data["atom_types"].detach().cpu().numpy(),
                            "pos": _data["pos"].detach().cpu().numpy(),
                            "sample_idx": samples_so_far
                            + self.global_rank * len(batch["num_atoms"])
                            + idx_in_batch,
                        }
                    )

            # Process synthesis outputs for each sample in the batch
            synthesis_outputs = out["synthesis_decoder"]

            for idx_in_batch in range(len(synthesis_outputs)):
                # Store synthesis data without target molecule for now
                # Target molecules will be extracted after molecule evaluator processes them
                pred_data = {
                    "synthesis_predictions": synthesis_outputs[idx_in_batch],
                    "target_molecule": None,  # Will be updated after molecule evaluation
                    "sample_idx": samples_so_far
                    + self.global_rank * len(batch["num_atoms"])
                    + idx_in_batch,
                }
                synthesis_evaluator.append_pred_data(pred_data)
        t_end = time.time()

        if stage == "val":
            # Compute generation metrics
            # For testing we don't log metrics here
            gen_metrics_dict = generation_evaluator.get_metrics(
                save=self.hparams.sampling.visualize,
                save_dir=self.hparams.sampling.save_dir + f"/molecules_{stage}_{self.global_rank}",
                do_pharmacophores=self.hparams.denoiser.use_conditioning,
                do_rocs=self.hparams.sampling.do_rocs_metric,
                do_pb=self.hparams.sampling.do_posebusters,
                compute_diversity=self.hparams.sampling.compute_diversity,
                n_samples_per_conditioning=1 if self.hparams.denoiser.use_conditioning else None,
                diversity_fingerprint_type=self.hparams.sampling.diversity_fingerprint_type,
            )
            gen_metrics_dict["sampling_time"] = t_end - t_start
            for k, v in gen_metrics_dict.items():
                metrics[k](v)
                self.log(
                    f"{stage}/{k}",
                    metrics[k],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False if k != "valid_rate" else True,
                    sync_dist=True,
                )

            # Update synthesis evaluator with target molecules from molecule evaluator
            if self.hparams.sampling.synthesis:
                synthesis_evaluator.update_target_molecules(generation_evaluator.pred_rdkit_list)

                # Compute synthesis metrics
                synthesis_metrics_dict = synthesis_evaluator.get_metrics()
                for k, v in synthesis_metrics_dict.items():
                    self.log(
                        f"{stage}/synthesis_{k}",
                        v,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True if k == "synthesis_molecule_match_rate" else False,
                        sync_dist=True,
                    )

            if self.hparams.sampling.visualize:
                pred_table = generation_evaluator.get_wandb_table(
                    current_epoch=self.current_epoch,
                    save_dir=self.hparams.sampling.save_dir
                    + f"/molecules_{stage}_{self.global_rank}",
                )
                self.logger.experiment.log(
                    {f"{stage}_samples_table_device{self.global_rank}": pred_table}
                )

                if self.hparams.sampling.synthesis:
                    # Log synthesis table
                    synthesis_table = synthesis_evaluator.get_wandb_table(
                        current_epoch=self.current_epoch
                    )
                    self.logger.experiment.log(
                        {f"{stage}_synthesis_table_device{self.global_rank}": synthesis_table}
                    )

    #####################################################################################################

    def sample_and_decode(
        self,
        batch_size,
        cfg_scale=4.0,
        num_nodes_bincount=None,
        sample_lengths=None,
        cond=None,
        cond_mask=None,
    ):
        # sample random lengths from distribution: (B, 1)
        if sample_lengths is None:
            assert num_nodes_bincount is not None, (
                "num_nodes_bincount must be provided if sample_lengths is not provided"
            )
            sample_lengths = torch.multinomial(
                num_nodes_bincount.float(),
                batch_size,
                replacement=True,
            ).to(self.device)

        # create token mask for visualization
        token_mask = torch.zeros(
            batch_size,
            max(sample_lengths),
            dtype=torch.bool,
            device=self.device,
        )
        for idx, length in enumerate(sample_lengths):
            token_mask[idx, :length] = True

        if self.hparams.sampling.use_cfg:
            samples = self.interpolant.sample_with_classifier_free_guidance(
                batch_size=batch_size,
                num_tokens=max(sample_lengths),
                emb_dim=self.denoiser.d_x,
                model=self.denoiser,
                cond=cond,
                cfg_scale=cfg_scale,
                token_mask=token_mask,
            )
        else:
            samples = self.interpolant.sample(
                batch_size=batch_size,
                num_tokens=max(sample_lengths),
                emb_dim=self.denoiser.d_x,
                model=self.denoiser,
                cond=cond,
                cond_mask=cond_mask,
                token_mask=token_mask,
            )

        # get final samples and remove padding (to PyG format)
        x = samples["clean_traj"][-1][token_mask]

        encoded_batch = {
            "x": x,
            "num_atoms": sample_lengths,
            "batch": torch.repeat_interleave(
                torch.arange(len(sample_lengths), device=self.device), sample_lengths
            ),
            "token_idx": (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask],
        }
        # decode samples using frozen decoder
        out, conditioning, conditioning_mask = self.autoencoder.decode_at_inference(encoded_batch)
        if self.hparams.sampling.synthesis:
            out["synthesis_decoder"] = self.generate_synthesis(conditioning, conditioning_mask)
        else:
            # Return empty synthesis outputs during validation
            batch_size = out["molecule_decoder"]["atom_types"].shape[0]
            out["synthesis_decoder"] = [[] for _ in range(batch_size)]
        return out, encoded_batch, samples

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

    # Sample given an input pharmacophore (this can come from the test set or from an external source)

    def _iter_pharmacophore_conditioning(self, loader, n_pharmacophores):
        """Yield up to n_pharmacophores per-pharmacophore conditioning tuples.

        Each item is (global_idx, c_dense_single, c_mask_single, molecule_data, num_atoms_i).
        c_dense_single/c_mask_single keep a leading batch dim of 1. molecule_data and
        num_atoms_i are None when the loader does not carry ground-truth molecules.
        """
        processed = 0
        for input_batch in loader:
            if processed >= n_pharmacophores:
                break

            pharmacophore_batch = input_batch["pharmacophores"].to(self.device)
            cond = self.ph4_encoder(pharmacophore_batch)
            cond_dense, cond_mask = to_dense_batch(cond["x"], cond["batch"])
            cond_dense = self.cond_projection(cond_dense)

            molecules_batch = input_batch["molecules"]
            if molecules_batch is not None:
                data_list = molecules_batch.to_data_list()
                num_atoms = molecules_batch["num_atoms"]
            else:
                data_list = None
                num_atoms = None

            for i in range(cond_dense.shape[0]):
                if processed >= n_pharmacophores:
                    break
                yield (
                    processed,
                    cond_dense[i : i + 1],
                    cond_mask[i : i + 1],
                    data_list[i] if data_list is not None else None,
                    num_atoms[i] if num_atoms is not None else None,
                )
                processed += 1

    @staticmethod
    def _apply_ph4_dropout(c_mask, min_kept=5):
        """Per-row random dropout of valid mask positions, keeping at least min_kept valid ones.

        Assumes every row of c_mask starts with the same set of valid positions (produced by
        repeating a single pharmacophore mask). Returns a new mask; original is not mutated.
        """
        valid_positions = torch.where(c_mask[0])[0]
        num_valid = int(valid_positions.numel())
        if num_valid <= min_kept:
            return c_mask

        chunk_size = c_mask.shape[0]
        # Sample a keep-count per row in [min_kept, num_valid]; equivalent to the original
        # "drop 0..num_valid-min_kept" formulation.
        num_kept = torch.randint(min_kept, num_valid + 1, (chunk_size,), device=c_mask.device)
        # Random priority per (row, valid_pos); keep the top num_kept by priority per row.
        priorities = torch.rand(chunk_size, num_valid, device=c_mask.device)
        ranks = priorities.argsort(dim=1).argsort(dim=1)
        keep_valid = ranks < num_kept.unsqueeze(1)

        out = c_mask.clone()
        out[:, valid_positions] = keep_valid
        return out

    def _sample_num_atoms(self, num_atoms_i, chunk_size, fix_num_atoms):
        """Return a [chunk_size] atom-count tensor, or None if no reference count is available."""
        if num_atoms_i is None:
            return None
        if fix_num_atoms:
            return num_atoms_i.repeat(chunk_size).to(self.device)
        center = int(num_atoms_i)
        min_atoms = max(1, center - 3)
        max_atoms = center + 3
        return torch.randint(min_atoms, max_atoms + 1, (chunk_size,), device=self.device)

    @staticmethod
    def _record_chunk_predictions(
        out,
        out_batch,
        molecule_data,
        base_sample_idx,
        global_pharmacophore_idx,
        generation_evaluator,
        synthesis_evaluator,
    ):
        """Append predictions, optional ground truth, and synthesis outputs for one decoded chunk."""
        num_atoms_list = out_batch["num_atoms"].tolist()
        atom_types_all = out["molecule_decoder"]["atom_types"]
        pos_all = out["molecule_decoder"]["pos"]
        synthesis_outputs = out["synthesis_decoder"]

        start_idx = 0
        for idx_in_chunk, num_atom in enumerate(num_atoms_list):
            global_sample_idx = base_sample_idx + idx_in_chunk
            _atom_types = atom_types_all.narrow(0, start_idx, num_atom).argmax(dim=1)
            _pos = pos_all.narrow(0, start_idx, num_atom) * 10.0  # nm to A
            generation_evaluator.append_pred_array(
                {
                    "atom_types": _atom_types.detach().cpu().numpy(),
                    "pos": _pos.detach().cpu().numpy(),
                    "sample_idx": global_sample_idx,
                }
            )
            if molecule_data is not None:
                generation_evaluator.append_gt_array(
                    {
                        "atom_types": molecule_data["atom_types"].detach().cpu().numpy(),
                        "pos": molecule_data["pos"].detach().cpu().numpy(),
                        "sample_idx": global_sample_idx,
                    }
                )
            synthesis_evaluator.append_pred_data(
                {
                    "synthesis_predictions": synthesis_outputs[idx_in_chunk],
                    "target_molecule": None,
                    "target_pharmacophore_id": None,
                    "sample_idx": global_sample_idx,
                    "pharmacophore_idx": global_pharmacophore_idx,
                }
            )
            start_idx += num_atom

    def sample_conditioned_on_pharmacophore(
        self,
        loader,
        n_samples_per_pharmacophore=100,
        n_pharmacophores=100,
        fix_num_atoms=False,
        ph4_dropout_enabled=False,
        max_batch_size=100,
    ):
        import logging

        generation_evaluator = self.test_generation_evaluator
        synthesis_evaluator = self.test_synthesis_evaluator

        pharmacophores_processed = 0
        pbar = tqdm(total=n_pharmacophores, desc="Generating samples for pharmacophores")

        for (
            global_ph_idx,
            c_dense_single,
            c_mask_single,
            molecule_data,
            num_atoms_i,
        ) in self._iter_pharmacophore_conditioning(loader, n_pharmacophores):
            samples_processed = 0
            while samples_processed < n_samples_per_pharmacophore:
                chunk_size = min(max_batch_size, n_samples_per_pharmacophore - samples_processed)

                c_dense = c_dense_single.repeat(chunk_size, 1, 1)
                c_mask = c_mask_single.repeat(chunk_size, 1)
                if ph4_dropout_enabled:
                    c_mask = self._apply_ph4_dropout(c_mask, min_kept=5)

                n_atoms = self._sample_num_atoms(num_atoms_i, chunk_size, fix_num_atoms)

                out, out_batch, _ = self.sample_and_decode(
                    num_nodes_bincount=self.num_nodes_bincount,
                    sample_lengths=n_atoms,
                    batch_size=chunk_size,
                    cfg_scale=self.hparams.sampling.cfg_scale,
                    cond=c_dense,
                    cond_mask=c_mask,
                )

                base_sample_idx = global_ph_idx * n_samples_per_pharmacophore + samples_processed
                self._record_chunk_predictions(
                    out,
                    out_batch,
                    molecule_data,
                    base_sample_idx=base_sample_idx,
                    global_pharmacophore_idx=global_ph_idx,
                    generation_evaluator=generation_evaluator,
                    synthesis_evaluator=synthesis_evaluator,
                )

                samples_processed += chunk_size

            pharmacophores_processed += 1
            pbar.update(1)

        pbar.close()

        logging.info(f"Completed sampling for {pharmacophores_processed} pharmacophores")
        logging.info(
            f"Total samples generated: {pharmacophores_processed * n_samples_per_pharmacophore}"
        )

        # Compute diversity metrics after sampling if enabled
        if hasattr(self.hparams, "diversity") and self.hparams.diversity.get("enabled", False):
            logging.info("Computing per-conditioning diversity metrics...")
            generation_evaluator = self.test_generation_evaluator

            # Ensure molecules are converted to RDKit format
            if not generation_evaluator.pred_rdkit_list:
                generation_evaluator._arrays_to_molecules(
                    save=self.hparams.sampling.visualize,
                    save_dir=self.hparams.sampling.save_dir + f"/molecules_test_{self.global_rank}",
                )

    #####################################################################################################

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)
            self.denoiser = torch.compile(self.denoiser)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_mp20/valid_rate",
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}
