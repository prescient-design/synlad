"""Per-pharmacophore ROCS hit evaluation.

For each pharmacophore target, scores generated molecules against the ground
truth reference ligand using ROCS shape/colour overlap and counts how many
unique SMILES exceed the hit threshold. Supports several generator methods
(synlad, synformer, shepherd, reinvent) and a dataset baseline.

Usage:
    python evaluate_pharmacophore_hits.py --samples_dir /path/to/sampling/outputs
"""

import argparse
import glob
import json
import os
import pickle
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from openeye import oechem, oeomega, oeshape
from synlad.utils.metrics_utils import rdmol_to_oemol

RDLogger.DisableLog("rdApp.*")

HIT_THRESHOLD = 1.2
MAX_COMBO_SENTINEL = 2.0


def load_molecules_from_pathways(pathways_path: str) -> List[Chem.Mol]:
    """Load molecules from pickle file of pathway data."""
    with open(pathways_path, "rb") as f:
        rxn_data_test = pickle.load(f)
    molecules = [task.molecule_conformations for task in rxn_data_test["dataset"]
                 if task.molecule_conformations is not None]
    print(f"Loaded {len(molecules)} molecules")
    return molecules


def score_smiles_to_query(smiles, query, metric_name="tanimoto"):
    assert metric_name in ["tanimoto", "tversky"], f"Invalid metric: {metric_name}"

    overlay_opts = oeshape.OEOverlayOptions()
    overlay_opts.SetShapeFuncType(oeshape.OEShapeType_Exact)
    overlay_opts.SetColorFuncType(oeshape.OEColorType_Exact)

    rocs_opts = oeshape.OEROCSOptions()
    rocs_opts.SetOverlayOptions(overlay_opts)
    rocs_opts.SetNumBestHits(1)
    rocs_opts.SetRankPredicate(oeshape.OEHighestTanimotoCombo())

    omega_opts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_ROCS)
    omega_opts.SetMaxConfs(100)
    omega_opts.SetStrictStereo(False)
    builder = oeomega.OEOmega(omega_opts)

    combo_scores, shape_scores, colour_scores, successful_smiles = [], [], [], []

    for smi in tqdm(smiles, desc="ROCS scoring"):
        mol = oechem.OEMol()
        if not oechem.OESmilesToMol(mol, smi):
            print(f"Failed to parse SMILES: {smi}")
            continue

        ret_code = builder.Build(mol)
        if ret_code != oeomega.OEOmegaReturnCode_Success:
            print(f"Failed to build {smi}")
            oechem.OEThrow.Warning("%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code)))
            continue

        rocs = oeshape.OEROCS(rocs_opts)
        rocs.AddMolecule(mol)

        best_res = next(iter(rocs.Overlay(query)), None)
        if best_res is None:
            continue

        if metric_name == "tanimoto":
            combo_scores.append(best_res.GetTanimotoCombo())
        else:
            combo_scores.append(best_res.GetFitTverskyCombo())
        shape_scores.append(best_res.GetTanimoto())
        colour_scores.append(best_res.GetColorTanimoto())
        successful_smiles.append(smi)

    return combo_scores, shape_scores, colour_scores, successful_smiles


def get_scaffold(smiles: str) -> str:
    """Get the scaffold SMILES of a molecule from its SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is not None:
        return Chem.MolToSmiles(scaffold)
    return None


def compute_diversity(smiles: List[str]) -> float:
    """Compute the diversity of a list of SMILES (1 - mean pairwise Tanimoto)."""
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [fpgen.GetFingerprint(mol) for mol in (Chem.MolFromSmiles(s) for s in smiles) if mol is not None]
    if len(fps) < 2:
        return 0.0
    similarities = [
        DataStructs.TanimotoSimilarity(fps[i], fps[j])
        for i in range(len(fps)) for j in range(i + 1, len(fps))
    ]
    return 1.0 - np.mean(similarities) if similarities else 0.0


def extract_smiles_for_synlad(generated_mols_path: str, pharmacophore_id: int):
    """Extract the 3D and synthesis prediction SMILES for a given pharmacophore ID."""
    synth_csv = os.path.join(generated_mols_path, "synthesis_pathways_for_eval.csv")
    if os.path.exists(synth_csv):
        synthesis_df = pd.read_csv(synth_csv)
        synthesis_smiles = synthesis_df[synthesis_df["pharmacophore_idx"] == pharmacophore_id]["final_product_smiles"].tolist()
    else:
        synthesis_smiles = []

    pred_dir = os.path.join(generated_mols_path, "pred")
    pred_smiles = []
    if os.path.exists(pred_dir):
        pred_sdf_files = glob.glob(os.path.join(pred_dir, f"*ph_{pharmacophore_id}.sdf"))
        if not pred_sdf_files:
            print(f"No predicted molecules found for pharmacophore ID {pharmacophore_id}")
        else:
            pred_mols = []
            for sdf_path in pred_sdf_files:
                pred_mols.extend(mol for mol in Chem.SDMolSupplier(sdf_path) if mol is not None)
            pred_smiles = [Chem.MolToSmiles(mol) for mol in pred_mols]
    return pred_smiles, synthesis_smiles


def extract_reference_ligand(generated_mols_path: str, pharmacophore_id: int):
    """Extract the reference ligand for a given pharmacophore ID."""
    gt_sdf_path = os.path.join(generated_mols_path, "gt", f"molecule_ph_{pharmacophore_id}.sdf")
    if not os.path.exists(gt_sdf_path):
        raise ValueError(f"Ground truth ligand not found: {gt_sdf_path}")
    return Chem.SDMolSupplier(gt_sdf_path)[0]


def _summarize_smiles(smiles: Sequence[str], ref_oemol, do_diversity: bool) -> Dict[str, Any]:
    """Score a set of SMILES against a reference and summarize hits/scaffolds/diversity."""
    if not smiles:
        return {
            "combo_scores": [], "shape_scores": [], "colour_scores": [],
            "num_hits": 0, "num_unique_scaffold_hits": 0, "diversity": 0.0, "hits": [],
        }

    combo_scores, shape_scores, colour_scores, successful_smiles = score_smiles_to_query(
        smiles, ref_oemol, metric_name="tanimoto"
    )

    hits = [smi for smi, score in zip(successful_smiles, combo_scores) if score >= HIT_THRESHOLD]
    scaffolds = {get_scaffold(smi) for smi in hits}

    return {
        "combo_scores": combo_scores,
        "shape_scores": shape_scores,
        "colour_scores": colour_scores,
        "num_hits": len(hits),
        "num_unique_scaffold_hits": len(scaffolds),
        "diversity": compute_diversity(successful_smiles) if do_diversity else 0.0,
        "hits": hits,
    }


def analysis_per_pharmacophore_dataset_baseline(generated_mols_path, smiles, pharmacophore_id, do_diversity=False):
    gt_mol = extract_reference_ligand(generated_mols_path, pharmacophore_id)
    ref_oemol = rdmol_to_oemol(gt_mol)
    return _summarize_smiles(smiles, ref_oemol, do_diversity)


def analysis_per_pharmacophore_synformer(generated_mols_path, synformer_df_path, pharmacophore_id,
                                         do_diversity=False, num_samples_per_pharmacophore=100):
    synformer_df = pd.read_csv(synformer_df_path)
    if pharmacophore_id not in synformer_df["pharmacophore_id"].unique():
        print(f"Pharmacophore ID {pharmacophore_id} not found in dataframe")
        return None

    gt_mol = extract_reference_ligand(generated_mols_path, pharmacophore_id)
    ref_oemol = rdmol_to_oemol(gt_mol)

    smiles_list = synformer_df[synformer_df["pharmacophore_id"] == pharmacophore_id]["smiles"].tolist()
    valid_rate = len(smiles_list) / num_samples_per_pharmacophore
    smiles_list = list(set(smiles_list))
    unique_smiles_rate = len(smiles_list) / num_samples_per_pharmacophore

    summary = _summarize_smiles(smiles_list, ref_oemol, do_diversity)
    return {
        "valid_rate": valid_rate,
        "unique_smiles_rate": unique_smiles_rate,
        "smiles": smiles_list,
        **summary,
    }


def analysis_per_pharmacophore_synlad(generated_mols_path, pharmacophore_id, do_diversity=False,
                                      num_samples_per_pharmacophore=100):
    pred_smiles, synthesis_smiles = extract_smiles_for_synlad(generated_mols_path, pharmacophore_id)
    gt_mol = extract_reference_ligand(generated_mols_path, pharmacophore_id)
    ref_oemol = rdmol_to_oemol(gt_mol)

    valid_3d_rate = len(pred_smiles) / num_samples_per_pharmacophore
    valid_synthesis_rate = len(synthesis_smiles) / num_samples_per_pharmacophore

    pred_smiles = list(set(pred_smiles))
    synthesis_smiles = list(set(synthesis_smiles))
    unique_3d_smiles_rate = len(pred_smiles) / num_samples_per_pharmacophore
    unique_synthesis_smiles_rate = len(synthesis_smiles) / num_samples_per_pharmacophore

    s3d = _summarize_smiles(pred_smiles, ref_oemol, do_diversity)
    ssy = _summarize_smiles(synthesis_smiles, ref_oemol, do_diversity)

    return {
        "valid_3d_rate": valid_3d_rate,
        "valid_synthesis_rate": valid_synthesis_rate,
        "unique_3d_smiles_rate": unique_3d_smiles_rate,
        "unique_synthesis_smiles_rate": unique_synthesis_smiles_rate,
        "num_hits_3d": s3d["num_hits"],
        "num_hits_synthesis": ssy["num_hits"],
        "num_unique_scaffold_hits_3d": s3d["num_unique_scaffold_hits"],
        "num_unique_scaffold_hits_synthesis": ssy["num_unique_scaffold_hits"],
        "shape_scores_3d": s3d["shape_scores"],
        "shape_scores_synthesis": ssy["shape_scores"],
        "colour_scores_3d": s3d["colour_scores"],
        "colour_scores_synthesis": ssy["colour_scores"],
        "combo_scores_3d": s3d["combo_scores"],
        "combo_scores_synthesis": ssy["combo_scores"],
        "diversity_3d": s3d["diversity"],
        "diversity_synthesis": ssy["diversity"],
        "pred_smiles": pred_smiles,
        "synthesis_smiles": synthesis_smiles,
        "hits_3d": s3d["hits"],
        "hits_synthesis": ssy["hits"],
    }


def _max_combo(combo_scores: Sequence[float]) -> float:
    """Max combo score, ignoring the sentinel value; 0.0 if empty."""
    filtered = [s for s in combo_scores if s != MAX_COMBO_SENTINEL]
    return float(np.max(filtered)) if filtered else 0.0


def _write_json(path: str, obj: Dict[str, Any], label: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {label} to {path}")


def _write_smiles_csv(path: str, smiles: Sequence[str], label: str) -> None:
    with open(path, "w") as f:
        for smi in smiles:
            f.write(f"{smi}\n")
    print(f"Saved {label} to {path}")


def _write_indexed_csv(path: str, values: Sequence, label: str) -> None:
    with open(path, "w") as f:
        for i, v in enumerate(values):
            f.write(f"{i},{v}\n")
    print(f"Saved {label} to {path}")


def _run_synlad(args):
    keys = [
        "valid_3d_rate", "valid_synthesis_rate",
        "unique_3d_smiles_rate", "unique_synthesis_smiles_rate",
        "diversity_3d", "diversity_synthesis",
        "num_hits_3d", "num_hits_synthesis",
        "num_unique_scaffold_hits_3d", "num_unique_scaffold_hits_synthesis",
        "shape_scores_3d", "shape_scores_synthesis",
        "colour_scores_3d", "colour_scores_synthesis",
        "combo_scores_3d", "combo_scores_synthesis",
    ]
    agg = {k: [] for k in keys}
    max_combo_3d, max_combo_synth = [], []
    all_3d_smiles, all_synthesis_smiles, hits_3d, hits_synthesis = [], [], [], []

    for ph_id in range(args.num_pharmacophores):
        r = analysis_per_pharmacophore_synlad(
            args.samples_dir, ph_id, args.do_diversity, args.num_samples_per_pharmacophore
        )
        for k in keys:
            agg[k].append(r[k])
        max_combo_3d.append(_max_combo(r["combo_scores_3d"]))
        max_combo_synth.append(_max_combo(r["combo_scores_synthesis"]))
        all_3d_smiles.extend(r["pred_smiles"])
        all_synthesis_smiles.extend(r["synthesis_smiles"])
        hits_3d.extend(r["hits_3d"])
        hits_synthesis.extend(r["hits_synthesis"])

    results = {
        "validity_3d_avg": np.mean(agg["valid_3d_rate"]),
        "validity_synthesis_avg": np.mean(agg["valid_synthesis_rate"]),
        "uniqueness_3d_avg": np.mean(agg["unique_3d_smiles_rate"]),
        "uniqueness_synthesis_avg": np.mean(agg["unique_synthesis_smiles_rate"]),
        "diversity_3d_avg": np.mean(agg["diversity_3d"]),
        "diversity_synthesis_avg": np.mean(agg["diversity_synthesis"]),
        "num_hits_3d_median": np.median(agg["num_hits_3d"]),
        "num_hits_3d_avg": np.mean(agg["num_hits_3d"]),
        "num_hits_synthesis_median": np.median(agg["num_hits_synthesis"]),
        "num_hits_synthesis_avg": np.mean(agg["num_hits_synthesis"]),
        "num_unique_scaffold_hits_3d_median": np.median(agg["num_unique_scaffold_hits_3d"]),
        "num_unique_scaffold_hits_3d_avg": np.mean(agg["num_unique_scaffold_hits_3d"]),
        "num_unique_scaffold_hits_synthesis_median": np.median(agg["num_unique_scaffold_hits_synthesis"]),
        "num_unique_scaffold_hits_synthesis_avg": np.mean(agg["num_unique_scaffold_hits_synthesis"]),
        "max_combo_scores_3d_median": np.median(max_combo_3d),
        "max_combo_scores_synthesis_median": np.median(max_combo_synth),
        "max_combo_scores_synthesis_mean": np.mean(max_combo_synth),
    }
    print(results)

    out = args.samples_dir
    _write_json(os.path.join(out, "results_synlad.json"), results, "results")
    _write_json(os.path.join(out, "scores_synlad.json"), {
        "shape_scores_3d": agg["shape_scores_3d"],
        "shape_scores_synthesis": agg["shape_scores_synthesis"],
        "colour_scores_3d": agg["colour_scores_3d"],
        "colour_scores_synthesis": agg["colour_scores_synthesis"],
        "combo_scores_3d": agg["combo_scores_3d"],
        "combo_scores_synthesis": agg["combo_scores_synthesis"],
    }, "scores")

    _write_smiles_csv(os.path.join(out, "all_3d_smiles.csv"), list(set(all_3d_smiles)), "all 3D smiles")
    _write_smiles_csv(os.path.join(out, "all_synthesis_smiles.csv"), list(set(all_synthesis_smiles)), "all synthesis smiles")
    _write_smiles_csv(os.path.join(out, "hits_3d.csv"), hits_3d, "hits_3d")
    _write_smiles_csv(os.path.join(out, "hits_synthesis.csv"), hits_synthesis, "hits_synthesis")

    _write_indexed_csv(os.path.join(out, "num_hits_3d.csv"), agg["num_hits_3d"], "num_hits_3d")
    _write_indexed_csv(os.path.join(out, "num_hits_synthesis.csv"), agg["num_hits_synthesis"], "num_hits_synthesis")
    _write_indexed_csv(os.path.join(out, "num_unique_scaffold_hits_3d.csv"),
                       agg["num_unique_scaffold_hits_3d"], "num_unique_scaffold_hits_3d")
    _write_indexed_csv(os.path.join(out, "num_unique_scaffold_hits_synthesis.csv"),
                       agg["num_unique_scaffold_hits_synthesis"], "num_unique_scaffold_hits_synthesis")


def _run_dataframe_method(args):
    """Handles synformer / shepherd / reinvent (all share the same df-based path)."""
    validity, uniqueness, diversity = [], [], []
    num_hits, num_unique_scaffold_hits = [], []
    shape_scores, colour_scores, combo_scores = [], [], []
    max_combo_scores, all_smiles, hits = [], [], []

    for ph_id in range(args.num_pharmacophores):
        r = analysis_per_pharmacophore_synformer(
            args.samples_dir, args.df_path, ph_id, args.do_diversity, args.num_samples_per_pharmacophore
        )
        if r is None:
            continue
        validity.append(r["valid_rate"])
        uniqueness.append(r["unique_smiles_rate"])
        diversity.append(r["diversity"])
        num_hits.append(r["num_hits"])
        num_unique_scaffold_hits.append(r["num_unique_scaffold_hits"])
        shape_scores.append(r["shape_scores"])
        colour_scores.append(r["colour_scores"])
        combo_scores.append(r["combo_scores"])
        max_combo_scores.append(_max_combo(r["combo_scores"]))
        all_smiles.extend(r["smiles"])
        hits.extend(r["hits"])

    results = {
        "validity_avg": np.mean(validity),
        "uniqueness_avg": np.mean(uniqueness),
        "diversity_avg": np.mean(diversity),
        "num_hits_median": np.median(num_hits),
        "num_hits_avg": np.mean(num_hits),
        "num_unique_scaffold_hits_median": np.median(num_unique_scaffold_hits),
        "num_unique_scaffold_hits_avg": np.mean(num_unique_scaffold_hits),
        "max_combo_scores_median": np.median(max_combo_scores),
    }
    print(results)

    out, method = args.samples_dir, args.method
    _write_json(os.path.join(out, f"results_{method}.json"), results, "results")
    _write_json(os.path.join(out, f"scores_{method}.json"), {
        "shape_scores": shape_scores,
        "colour_scores": colour_scores,
        "combo_scores": combo_scores,
    }, "scores")
    _write_smiles_csv(os.path.join(out, f"all_{method}_smiles.csv"), list(set(all_smiles)), "all smiles")
    _write_indexed_csv(os.path.join(out, f"num_hits_{method}.csv"), num_hits, "num_hits")
    _write_indexed_csv(os.path.join(out, f"num_unique_scaffold_hits_{method}.csv"),
                       num_unique_scaffold_hits, f"num_unique_scaffold_hits_{method}")
    _write_smiles_csv(os.path.join(out, f"hits_{method}.csv"), hits, "hits")


def _run_dataset_baseline(args):
    """Score a fixed pool of molecules against each pharmacophore's reference ligand.

    The molecule pool comes from EITHER:
      - ``--dataset_mols_path``: a plain-text file with one SMILES per line, OR
      - ``--train_data_path``: a pickle of the pathway dataset (see
        ``load_molecules_from_pathways``); molecules are shuffled before use.
    In both cases the pool is truncated to ``--max_molecules``.

    ``--samples_dir`` must still point at a directory containing reference ligands
    in ``<samples_dir>/gt/molecule_ph_<i>.sdf`` for every ``i`` in
    ``range(num_pharmacophores)`` — these are the queries each pool molecule is
    scored against. No ``pred/`` or ``synthesis_pathways_for_eval.csv`` is needed
    for this method; only ``gt/`` is read.
    """
    if args.dataset_mols_path:
        with open(args.dataset_mols_path, "r") as f:
            smiles = [line.strip() for line in f if line.strip()]
        if args.max_molecules:
            smiles = smiles[:args.max_molecules]
        print(f"Using {len(smiles)} molecules from {args.dataset_mols_path}")
    else:
        molecules = load_molecules_from_pathways(args.train_data_path)
        import random
        random.shuffle(molecules)
        if args.max_molecules:
            molecules = molecules[:args.max_molecules]
        smiles = [Chem.MolToSmiles(mol) for mol in molecules]
        print(f"Using {len(smiles)} molecules from {args.train_data_path}")

    num_hits, num_unique_scaffold_hits = [], []
    shape_scores, colour_scores, combo_scores = [], [], []
    max_combo_scores, hits = [], []

    for ph_id in range(args.num_pharmacophores):
        r = analysis_per_pharmacophore_dataset_baseline(args.samples_dir, smiles, ph_id, args.do_diversity)
        num_hits.append(r["num_hits"])
        num_unique_scaffold_hits.append(r["num_unique_scaffold_hits"])
        shape_scores.append(r["shape_scores"])
        colour_scores.append(r["colour_scores"])
        combo_scores.append(r["combo_scores"])
        max_combo_scores.append(_max_combo(r["combo_scores"]))
        hits.extend(r["hits"])

    results = {
        "num_hits_median": np.median(num_hits),
        "num_hits_avg": np.mean(num_hits),
        "num_unique_scaffold_hits_median": np.median(num_unique_scaffold_hits),
        "num_unique_scaffold_hits_avg": np.mean(num_unique_scaffold_hits),
        "max_combo_scores_median": np.median(max_combo_scores),
    }
    print(results)

    out = args.samples_dir
    tag = args.dataset_mols_path.split("/")[-1]
    _write_json(os.path.join(out, f"results_dataset_baseline_{tag}.json"), results, "results")
    _write_json(os.path.join(out, f"scores_dataset_baseline_{tag}.json"), {
        "shape_scores": shape_scores,
        "colour_scores": colour_scores,
        "combo_scores": combo_scores,
    }, "scores")
    _write_smiles_csv(os.path.join(out, f"random_smiles_{tag}.csv"), smiles, "random smiles")
    _write_smiles_csv(os.path.join(out, f"hits_{tag}.csv"), list(set(hits)), "hits")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the ROCS scores for the 3D and synthesis predictions")
    parser.add_argument("--samples_dir", "-d", required=True,
                        help="Directory containing pred/ and gt/ subdirectories with SDF files")
    parser.add_argument("--num_pharmacophores", "-n", type=int, default=50,
                        help="Number of pharmacophores to analyze (default: 50)")
    parser.add_argument("--do_diversity", action="store_true", default=False,
                        help="Compute diversity scores")
    parser.add_argument("--num_samples_per_pharmacophore", "-ns", type=int, default=100,
                        help="Number of samples per pharmacophore (default: 100)")
    parser.add_argument("--method", "-m",
                        choices=["synlad", "dataset_baseline", "synformer", "shepherd", "reinvent"],
                        default="synlad", help="Method to compute the metrics for.")
    parser.add_argument("--train_data_path", "-train", required=False,
                        help="Path to the training data pickle file")
    parser.add_argument("--dataset_mols_path", "-dm", default="",
                        help="Optional path to the dataset molecules txt file.")
    parser.add_argument("--max_molecules", "-mm", type=int, default=500,
                        help="Maximum number of molecules to use from the training data")
    parser.add_argument("--df_path", "-df", default="",
                        help="Path to the SynFormer/Shepherd dataframe")
    args = parser.parse_args()

    print("=== ROCS Score Analysis ===")
    print(f"Samples directory: {args.samples_dir}")

    if args.method == "synlad":
        _run_synlad(args)
    elif args.method in ("synformer", "shepherd", "reinvent"):
        _run_dataframe_method(args)
    elif args.method == "dataset_baseline":
        _run_dataset_baseline(args)
    else:
        raise ValueError(f"Invalid method: {args.method}")


if __name__ == "__main__":
    main()
