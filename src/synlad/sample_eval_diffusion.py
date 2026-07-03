import json
import os
import pickle
from typing import Any

import hydra
import lightning as L
import numpy as np
import pandas as pd
import rootutils
import torch
from lightning import LightningDataModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from synlad.data.components.molecule_dataset import MoleculeDataset  # noqa: E402
from synlad.eval.synthesis_generation import calculate_tanimoto_similarity  # noqa: E402
from synlad.models.ldm_module import LatentDiffusionLitModule  # noqa: E402
from synlad.utils import (  # noqa: E402
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

DEFAULT_MAX_ATOMS = 40
SIMILARITY_REPORT_THRESHOLDS = (0.8, 0.95)
SYNTHESIS_SPECIAL_TOKENS = frozenset({"<EOS>", "<FWD_RXN>", "<FWD_RXN_RUN>", "<BB>"})


def create_custom_sdf_loader(
    ligands_path: str, batch_size: int = 8, max_atoms: int = DEFAULT_MAX_ATOMS
) -> DataLoader:
    """Create a loader from a pickled list of molecules carrying ph4_coords / ph4_channels."""
    with open(ligands_path, "rb") as f:
        molecules = pickle.load(f)
    if len(molecules) == 0:
        raise ValueError(f"No molecules found in {ligands_path}")

    pharmacophores = []
    for mol in molecules:
        coords = (
            mol.ph4_coords
            if isinstance(mol.ph4_coords, torch.Tensor)
            else torch.tensor(mol.ph4_coords, dtype=torch.float32)
        )
        channels = (
            mol.ph4_channels
            if isinstance(mol.ph4_channels, torch.Tensor)
            else torch.tensor(mol.ph4_channels, dtype=torch.long)
        )
        pharmacophores.append({0: {"coords": coords, "channels": channels}})

    dataset = MoleculeDataset(
        data_source=molecules,
        pharmacophores=pharmacophores,
        use_all_conformers=False,
        max_atoms=max_atoms,
        coords_normalizer=1.0,
        removeHs=True,
    )

    def collate_fn(batch):
        if not all(isinstance(item, tuple) and len(item) == 2 for item in batch):
            raise TypeError("Expected MoleculeDataset to yield (molecule, pharmacophore) tuples")
        molecules_list, pharmacophores_list = zip(*batch, strict=False)
        return {
            "molecules": Batch.from_data_list(list(molecules_list)),
            "pharmacophores": Batch.from_data_list(list(pharmacophores_list)),
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def extract_final_product(synthesis_sequence) -> str | None:
    """Return the last token in the sequence that parses as a valid SMILES."""
    for token in reversed(list(synthesis_sequence)):
        if not isinstance(token, str) or not token or token in SYNTHESIS_SPECIAL_TOKENS:
            continue
        if Chem.MolFromSmiles(token) is not None:
            return token
    return None


def _save_molecules_and_syntheses(
    generation_evaluator,
    synthesis_evaluator,
    save_dir: str,
    logger,
    synthesis_top_k: int = 1,
) -> None:
    """Save sampled molecules as SDF and synthesis pathways as CSV."""
    rdkit_molecules = []
    logger.info(f"Processing {len(generation_evaluator.pred_rdkit_list)} sampled molecules...")

    for i, entry in enumerate(generation_evaluator.pred_rdkit_list):
        if len(entry) == 5:
            mol, _, _, valid, pharmacophore_id = entry
        elif len(entry) == 4:
            mol, _, _, valid = entry
            pharmacophore_id = None
        else:
            raise ValueError(f"pred_rdkit_list entry of unexpected length: {len(entry)}")

        if mol is not None and valid:
            mol.SetProp("sample_idx", str(i))
            if pharmacophore_id is not None:
                mol.SetProp("pharmacophore_id", str(pharmacophore_id))
            rdkit_molecules.append(mol)

    sdf_path = os.path.join(save_dir, "sampled_molecules.sdf")
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
        pharmacophore_idx = pred_data.get("pharmacophore_idx", -1)
        target_molecule_smiles = pred_data.get("target_molecule", "")
        target_pharmacophore_id = pred_data.get("target_pharmacophore_id", None)
        synthesis_predictions = pred_data.get("synthesis_predictions", [])

        for synth_pred in synthesis_predictions:
            sequence = synth_pred.get("sequence", [])
            score = synth_pred.get("score")
            sequence_str = " → ".join(sequence) if sequence else ""
            final_product = extract_final_product(sequence) if sequence else None
            similarity = calculate_tanimoto_similarity(final_product, target_molecule_smiles)

            synthesis_data.append(
                {
                    "sample_idx": sample_idx,
                    "pharmacophore_idx": pharmacophore_idx,
                    "target_molecule_smiles": target_molecule_smiles,
                    "target_pharmacophore_id": target_pharmacophore_id,
                    "synthesis_sequence": sequence_str,
                    "synthesis_score": score,
                    "final_product_smiles": final_product,
                    "similarity_to_target": similarity,
                }
            )

    df_synthesis = pd.DataFrame(synthesis_data)
    if synthesis_top_k > 1 and len(df_synthesis):
        print(df_synthesis)
        df_synthesis = _keep_top_per_chunk(df_synthesis, synthesis_top_k)
        print(df_synthesis)

    synthesis_csv_path = os.path.join(save_dir, "synthesis_pathways.csv")
    df_synthesis.to_csv(synthesis_csv_path, index=False)
    logger.info(f"Saved {len(df_synthesis)} synthesis pathways to: {synthesis_csv_path}")

    if len(df_synthesis):
        eval_csv_path = os.path.join(save_dir, "synthesis_pathways_for_eval.csv")
        df_synthesis[["pharmacophore_idx", "final_product_smiles"]].to_csv(
            eval_csv_path, index=False
        )
        logger.info(f"Saved {len(df_synthesis)} synthesis pathways to: {eval_csv_path}")


def _keep_top_per_chunk(df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    """Within consecutive chunks of `chunk_size` rows, keep the row with the highest similarity.

    Falls back to the first row of the chunk when every similarity in the chunk is NaN.
    """
    group = df.index // chunk_size
    keep = df.groupby(group)["similarity_to_target"].agg(
        lambda s: s.idxmax() if s.notna().any() else s.index[0]
    )
    return df.loc[keep.astype(int)].reset_index(drop=True)


def _override_sampling_cfg(model, cfg: DictConfig) -> None:
    """Apply sampling/interpolant overrides from `cfg` onto a loaded model in place."""
    if "sampling" in cfg:
        # Resolve first so interpolations like ${DIR_NAME} (defined at cfg's root)
        # are baked in before we detach the subtree by merging it under model.hparams.
        sampling_override = OmegaConf.create(OmegaConf.to_container(cfg.sampling, resolve=True))
        model.hparams.sampling = OmegaConf.merge(model.hparams.sampling, sampling_override)
        log.info(f"Merged sampling overrides: {OmegaConf.to_container(sampling_override)}")

    if OmegaConf.select(cfg, "interpolant.num_timesteps") is not None:
        model.interpolant.num_timesteps = cfg.interpolant.num_timesteps
        log.info(f"Set interpolant.num_timesteps = {cfg.interpolant.num_timesteps}")

    # Synthesis pipeline resources are stored as plain instance attributes in
    # __init__, not just on hparams — override them so eval-time configs can
    # supply values the training checkpoint left unset.
    for key in ("vocab_json", "rxn_server_url"):
        value = cfg.get(key)
        if value is None:
            continue
        setattr(model, key, value)
        if key in model.hparams:
            model.hparams[key] = value
        log.info(f"Set {key} = {value}")


def _build_pharmacophore_loader(cfg: DictConfig, datamodule: LightningDataModule | None):
    """Return (loader, n_pharmacophores) for conditioned sampling."""
    if cfg.sample_from_test_set:
        datamodule.setup("test")
        loader = datamodule.test_dataloader()
    else:
        ligands_path = cfg.get("ligands_path")
        if ligands_path is None:
            raise ValueError("ligands_path must be provided in config when datamodule is None")
        batch_size = cfg.get("batch_size", 8)
        max_atoms = cfg.data.get("max_atoms", DEFAULT_MAX_ATOMS)
        loader = create_custom_sdf_loader(ligands_path, batch_size, max_atoms)

    log.info(
        f"Will generate {cfg.n_samples_per_pharmacophore} samples per pharmacophore, "
        f"for {cfg.n_pharmacophores} pharmacophores..."
    )
    log.info(f"Total expected samples: {cfg.n_samples_per_pharmacophore * cfg.n_pharmacophores}")
    return loader


def _collect_similarity_scores(synthesis_evaluator) -> dict[str, list]:
    """Pull max similarity / ROCS scores out of an evaluator's processed_results."""
    out = {
        "sample_indices": [],
        "max_similarities": [],
        "combo_scores": [],
        "shape_scores": [],
        "colour_scores": [],
    }
    for result in synthesis_evaluator.processed_results:
        sample_idx = result.get("sample_idx", len(out["max_similarities"]))
        max_sim = result.get("max_similarity_to_target")
        if max_sim is not None:
            out["sample_indices"].append(sample_idx)
            out["max_similarities"].append(float(max_sim))

        for src_key, dst_key in (
            ("max_combo_score", "combo_scores"),
            ("max_shape_score", "shape_scores"),
            ("max_colour_score", "colour_scores"),
        ):
            value = result.get(src_key)
            if value is not None:
                out[dst_key].append(float(value))

    log.info(
        f"Collected {len(out['max_similarities'])} max similarity scores (for valid mols only)"
    )
    log.info(f"Collected {len(out['combo_scores'])} combo scores")
    log.info(f"Collected {len(out['shape_scores'])} shape scores")
    log.info(f"Collected {len(out['colour_scores'])} colour scores")
    return out


def _process_metric_value(value):
    """Convert a torchmetrics / tensor value to a JSON-friendly primitive."""
    if hasattr(value, "compute"):
        return value.compute().item()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (int, float)):
        return value
    return str(value)


def _save_results(
    cfg: DictConfig, processed_metrics: dict, generation_evaluator, synthesis_evaluator
) -> None:
    """Persist metrics, config, and sampled artifacts to `cfg.results_save_dir`."""
    save_dir = cfg.get("results_save_dir")
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)

    metrics_path = os.path.join(save_dir, "evaluator_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(processed_metrics, f, indent=2)
    log.info(f"Saved evaluator metrics to: {metrics_path}")

    config_path = os.path.join(save_dir, "sampling_config.yaml")
    OmegaConf.save(cfg, config_path)
    log.info(f"Saved configuration to: {config_path}")

    log.info("Saving sampled molecules and synthesis pathways...")
    _save_molecules_and_syntheses(
        generation_evaluator,
        synthesis_evaluator,
        save_dir,
        log,
        synthesis_top_k=cfg.sampling.synthesis_top_k,
    )


def _log_metrics_summary(processed_metrics: dict) -> None:
    log.info("=== EVALUATOR METRICS SUMMARY ===")
    for section, header in (
        ("molecule_generation_metrics", "Molecule Generation Metrics:"),
        ("synthesis_generation_metrics", "Synthesis Generation Metrics:"),
    ):
        log.info(f"\n{header}")
        for key, value in processed_metrics[section].items():
            if "rate" in key and "time" not in key:
                log.info(f"  {key}: {value:.1%}")
            else:
                log.info(f"  {key}: {value:.4f}")

    max_sims = processed_metrics["max_similarities"]
    if not max_sims:
        return
    arr = np.array(max_sims)
    log.info("\nSimilarities to target molecules (valid mols only):")
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
def sample_and_evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample from a trained diffusion model and run the evaluator suite.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        A `(metrics, objects)` pair: the processed metrics dict and the instantiated objects.
    """
    if not cfg.ckpt_path:
        raise ValueError("Must provide checkpoint path for sampling evaluation")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    if cfg.get("sample_from_test_set", True):
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    else:
        datamodule = None

    log.info(f"Loading model from checkpoint <{cfg.ckpt_path}>")
    model = LatentDiffusionLitModule.load_from_checkpoint(cfg.ckpt_path, weights_only=False)
    model.eval()

    _override_sampling_cfg(model, cfg)

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

    log.info("Starting sampling and evaluation!")
    if cfg.get("do_pharmacophores", False):
        loader = _build_pharmacophore_loader(cfg, datamodule)
        model.sample_conditioned_on_pharmacophore(
            loader,
            n_samples_per_pharmacophore=cfg.n_samples_per_pharmacophore,
            n_pharmacophores=cfg.n_pharmacophores,
            fix_num_atoms=cfg.get("fix_num_atoms", False),
            ph4_dropout_enabled=cfg.get("ph4_dropout_enabled", False),
        )
    else:
        log.info(f"Will generate {model.hparams.sampling.num_samples} samples...")
        trainer.test(
            model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path, weights_only=False
        )

    test_generation_evaluator = model.test_generation_evaluator
    test_synthesis_evaluator = model.test_synthesis_evaluator

    log.info("Computing evaluator metrics...")
    gen_metrics_dict = test_generation_evaluator.get_metrics(
        save=model.hparams.sampling.visualize,
        save_dir=model.hparams.sampling.save_dir,
        do_pharmacophores=model.hparams.denoiser.use_conditioning,
        do_rocs=model.hparams.sampling.do_rocs_metric,
        compute_diversity=cfg.diversity.enabled,
        n_samples_per_conditioning=cfg.n_samples_per_pharmacophore
        if cfg.get("do_pharmacophores", False)
        else None,
        diversity_fingerprint_type=cfg.diversity.fingerprint_type,
        do_pb=cfg.get("do_posebusters", False),
    )

    if model.hparams.sampling.synthesis:
        test_synthesis_evaluator.update_target_molecules(test_generation_evaluator.pred_rdkit_list)
        synthesis_metrics_dict = test_synthesis_evaluator.get_metrics(
            save_dir=model.hparams.sampling.save_dir,
            compute_rocs_for_syn=model.hparams.sampling.do_rocs_for_syn,
        )
    else:
        synthesis_metrics_dict = {}

    log.info("Extracting max similarities for plotting...")
    scores = _collect_similarity_scores(test_synthesis_evaluator)

    processed_metrics: dict[str, Any] = {
        "molecule_generation_metrics": {},
        "synthesis_generation_metrics": {},
        "max_similarities": scores["max_similarities"],
        "combo_scores": scores["combo_scores"],
        "shape_scores": scores["shape_scores"],
        "colour_scores": scores["colour_scores"],
        "sample_indices": scores["sample_indices"],
        "all_evaluator_metrics": {},
    }
    for key, value in gen_metrics_dict.items():
        processed_value = _process_metric_value(value)
        processed_metrics["molecule_generation_metrics"][key] = processed_value
        processed_metrics["all_evaluator_metrics"][key] = processed_value
    for key, value in synthesis_metrics_dict.items():
        processed_value = _process_metric_value(value)
        processed_metrics["synthesis_generation_metrics"][key] = processed_value
        processed_metrics["all_evaluator_metrics"][key] = processed_value

    _save_results(cfg, processed_metrics, test_generation_evaluator, test_synthesis_evaluator)
    _log_metrics_summary(processed_metrics)

    return processed_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sample_uncond.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for sampling evaluation.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    metrics, _ = sample_and_evaluate(cfg)
    return metrics


if __name__ == "__main__":
    main()
