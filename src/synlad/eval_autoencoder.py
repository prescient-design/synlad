"""Adapted from https://github.com/facebookresearch/all-atom-diffusion-transformer/blob/main/src/eval_autoencoder.py

Copyright (c) Meta Platforms, Inc. and affiliates.
This code is released under the CC-BY-NC License -- see https://github.com/facebookresearch/all-atom-diffusion-transformer for further details.
"""

import json
import math
import os
from datetime import datetime
from typing import Any

import hydra
import lightning as L
import numpy as np
import pandas as pd
import rootutils
from lightning import LightningDataModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict
from rdkit import Chem

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from synlad.models.vae_module import VariationalAutoencoderLitModule  # noqa: E402
from synlad.utils import (  # noqa: E402
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

SIMILARITY_REPORT_THRESHOLDS = (0.8, 0.95)
SYNTHESIS_SPECIAL_TOKENS = frozenset({"<EOS>", "<FWD_RXN>", "<FWD_RXN_RUN>", "<BB>"})


def extract_final_product(synthesis_sequence) -> str | None:
    """Return the last token in the sequence that parses as a valid SMILES."""
    for token in reversed(list(synthesis_sequence)):
        if not isinstance(token, str) or not token or token in SYNTHESIS_SPECIAL_TOKENS:
            continue
        if Chem.MolFromSmiles(token) is not None:
            return token
    return None


def _save_molecules_and_syntheses(
    reconstruction_evaluator,
    synthesis_evaluator,
    save_dir: str,
    logger,
) -> None:
    """Save reconstructed molecules as SDF and synthesis pathways as CSV."""
    rdkit_molecules = []
    logger.info(
        f"Processing {len(reconstruction_evaluator.pred_rdkit_list)} reconstructed molecules..."
    )
    for i, mol in enumerate(reconstruction_evaluator.pred_rdkit_list):
        if mol is not None:
            mol.SetProp("sample_idx", str(i))
            rdkit_molecules.append(mol)

    sdf_path = os.path.join(save_dir, "reconstructed_molecules.sdf")
    if rdkit_molecules:
        with Chem.SDWriter(sdf_path) as w:
            for mol in rdkit_molecules:
                w.write(mol)
        logger.info(f"Saved {len(rdkit_molecules)} valid molecules to: {sdf_path}")
    else:
        logger.warning("No valid molecules to save!")

    logger.info("Processing synthesis pathways...")
    synthesis_data = []
    for pred_data in synthesis_evaluator.pred_data_list:
        sample_idx = pred_data.get("sample_idx", -1)
        target_molecule_smiles = pred_data.get("target_molecule", "")
        for synth_pred in pred_data.get("synthesis_predictions", []):
            sequence = synth_pred.get("sequence", [])
            score = synth_pred.get("score")
            sequence_str = " → ".join(sequence) if sequence else ""
            final_product = extract_final_product(sequence) if sequence else None
            synthesis_data.append(
                {
                    "sample_idx": sample_idx,
                    "target_molecule_smiles": target_molecule_smiles,
                    "synthesis_sequence": sequence_str,
                    "synthesis_score": score,
                    "final_product_smiles": final_product,
                }
            )

    synthesis_csv_path = os.path.join(save_dir, "synthesis_pathways.csv")
    pd.DataFrame(synthesis_data).to_csv(synthesis_csv_path, index=False)
    logger.info(f"Saved {len(synthesis_data)} synthesis pathways to: {synthesis_csv_path}")


def _override_hparams(model, cfg: DictConfig) -> None:
    """Apply config overrides onto `model.hparams` in place."""
    log.info("Updating configuration...")
    for key, value in cfg.items():
        if hasattr(model.hparams, key):
            setattr(model.hparams, key, value)
        else:
            # Allow older checkpoints that pre-date this key
            setattr(model.hparams, key, value)
        # Some module state (e.g. vocab_json, rxn_server_url) is stored as plain
        # instance attributes in __init__, not just on hparams. Mirror the override
        # there so checkpoints that were trained without these values can still be
        # evaluated by providing them via the eval config.
        if key in model.__dict__:
            setattr(model, key, value)


def _process_metric_value(value):
    """Convert a torchmetrics / tensor / numpy value to a JSON-friendly primitive."""
    if hasattr(value, "compute"):
        value = value.compute()
    if hasattr(value, "numel") and value.numel() > 1:
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (int, float)):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _collect_max_similarities(synthesis_evaluator) -> tuple[list[float], list[int]]:
    max_similarities: list[float] = []
    sample_indices: list[int] = []
    for result in synthesis_evaluator.processed_results:
        max_sim = result.get("max_similarity_to_target")
        if max_sim is None:
            continue
        sample_indices.append(result.get("sample_idx", len(max_similarities)))
        max_similarities.append(float(max_sim))
    log.info(f"Collected {len(max_similarities)} max similarity scores (for valid mols only)")
    return max_similarities, sample_indices


def _save_results(
    cfg: DictConfig,
    processed_metrics: dict,
    reconstruction_evaluator,
    synthesis_evaluator,
) -> None:
    """Persist metrics, config, and sampled artifacts to a timestamped subdir of `results_save_dir`."""
    results_save_dir = cfg.get("results_save_dir")
    if not results_save_dir:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"{results_save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    log.info(f"Results will be saved to: {save_dir}")

    metrics_path = os.path.join(save_dir, "evaluator_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(processed_metrics, f, indent=2)
    log.info(f"Saved evaluator metrics to: {metrics_path}")

    config_path = os.path.join(save_dir, "evaluation_config.yaml")
    OmegaConf.save(cfg, config_path)
    log.info(f"Saved configuration to: {config_path}")

    log.info("Saving reconstructed molecules and synthesis pathways...")
    _save_molecules_and_syntheses(reconstruction_evaluator, synthesis_evaluator, save_dir, log)


def _format_metric_value(key: str, value) -> str:
    is_rate = "rate" in key and "time" not in key
    if isinstance(value, list):
        if len(value) == 1:
            value = value[0]
        else:
            mean_val = float(np.mean(value))
            suffix = f" (mean of {len(value)} values)"
            return (f"{mean_val:.1%}" if is_rate else f"{mean_val:.4f}") + suffix
    return f"{value:.1%}" if is_rate else f"{value:.4f}"


def _log_metrics_summary(processed_metrics: dict) -> None:
    log.info("=== EVALUATOR METRICS SUMMARY ===")
    for section, header in (
        ("molecule_reconstruction_metrics", "Molecule Reconstruction Metrics:"),
        ("synthesis_generation_metrics", "Synthesis Generation Metrics:"),
    ):
        log.info(f"\n{header}")
        for key, value in processed_metrics[section].items():
            log.info(f"  {key}: {_format_metric_value(key, value)}")

    max_sims = processed_metrics["max_similarities"]
    if not max_sims:
        return
    arr = np.array(max_sims)
    log.info("\nSimilarities to target molecules:")
    log.info(f"  Count: {len(max_sims)}")
    log.info(f"  Mean: {arr.mean():.3f}")
    log.info(f"  Std: {arr.std():.3f}")
    log.info(f"  Min/Max: {arr.min():.3f} / {arr.max():.3f}")
    for threshold in SIMILARITY_REPORT_THRESHOLDS:
        n_above = int(np.sum(arr > threshold))
        log.info(
            f"  >{threshold}: {n_above}/{len(max_sims)} ({100 * n_above / len(max_sims):.1f}%)"
        )


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a VAE checkpoint on a datamodule testset.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        A `(metric_dict, object_dict)` pair: trainer callback metrics and instantiated objects.
    """
    if not cfg.ckpt_path:
        raise ValueError("Must provide checkpoint path for evaluation")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    max_molecules = (
        cfg.get("evaluation", {}).get("max_molecules") if hasattr(cfg, "evaluation") else None
    )
    if max_molecules is not None:
        max_test_batches = math.ceil(max_molecules / cfg.data.batch_size)
        log.info(
            f"Limiting evaluation to {max_molecules} molecules "
            f"(limit_test_batches={max_test_batches} at batch_size={cfg.data.batch_size})"
        )
        with open_dict(cfg):
            cfg.trainer.limit_test_batches = max_test_batches

    log.info(f"Loading autoencoder model from checkpoint <{cfg.ckpt_path}>")
    model = VariationalAutoencoderLitModule.load_from_checkpoint(cfg.ckpt_path, weights_only=False)
    model.eval()

    _override_hparams(model, cfg)
    if hasattr(cfg.data, "batch_size"):
        log.info(f"Evaluation batch size: {cfg.data.batch_size}")

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path, weights_only=False)

    test_reconstruction_evaluator = model.reconstruction_evaluator
    test_synthesis_evaluator = model.test_synthesis_evaluator

    log.info("Computing evaluator metrics...")
    rec_metrics_dict = test_reconstruction_evaluator.get_metrics(
        save=model.hparams.visualization.visualize,
        save_dir=f"{model.hparams.visualization.save_dir}/molecules_test_{model.global_rank}",
    )

    pred_rdkit_list = [
        (None, smiles, None, True) for smiles in test_reconstruction_evaluator.pred_smiles_list
    ]
    test_synthesis_evaluator.update_target_molecules(pred_rdkit_list)
    synthesis_metrics_dict = test_synthesis_evaluator.get_metrics()

    log.info("Extracting max similarities for plotting...")
    max_similarities, sample_indices = _collect_max_similarities(test_synthesis_evaluator)

    processed_metrics: dict[str, Any] = {
        "molecule_reconstruction_metrics": {},
        "synthesis_generation_metrics": {},
        "max_similarities": max_similarities,
        "sample_indices": sample_indices,
        "all_evaluator_metrics": {},
    }
    for key, value in rec_metrics_dict.items():
        processed_value = _process_metric_value(value)
        processed_metrics["molecule_reconstruction_metrics"][key] = processed_value
        processed_metrics["all_evaluator_metrics"][key] = processed_value
    for key, value in synthesis_metrics_dict.items():
        processed_value = _process_metric_value(value)
        processed_metrics["synthesis_generation_metrics"][key] = processed_value
        processed_metrics["all_evaluator_metrics"][key] = processed_value

    _save_results(cfg, processed_metrics, test_reconstruction_evaluator, test_synthesis_evaluator)
    _log_metrics_summary(processed_metrics)

    return trainer.callback_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval_vae.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
