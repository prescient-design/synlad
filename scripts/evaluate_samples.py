"""Adapted from https://github.com/andreirekesh/SynCoGen/blob/main/syncogen/eval/metrics.py.
"""

import argparse
import collections
import json
import pickle
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, AllChem, rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer
from rdkit.SimDivFilters import rdSimDivPickers
from tqdm import tqdm

from aizynthfinder.aizynthfinder import AiZynthFinder


Entry = Union[str, Path, Chem.Mol]


# -----------------------------------------------------------------------------
# Small numeric / chemistry helpers
# -----------------------------------------------------------------------------
def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _compute_mmff_energy(mol: Chem.Mol) -> Optional[float]:
    if mol.GetNumConformers() == 0:
        return None
    mol_copy = Chem.Mol(mol)
    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_copy, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mmff_props, confId=0)
        return ff.CalcEnergy()
    except Exception:
        return None


def _sphere_exclusion_diversity(fps, thresh: float = 0.65) -> float:
    if not fps:
        return 0.0
    picks = rdSimDivPickers.LeaderPicker().LazyBitVectorPick(fps, len(fps), thresh)
    return len(picks) / len(fps)


def _internal_diversity(fps) -> float:
    tot, pairs = 0.0, 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        tot += sum(sims)
        pairs += len(sims)
    return 1.0 - tot / pairs if pairs else 0.0


_BH2_QUERY = Chem.MolFromSmarts("[B;H2]")
_BORONIC_REPL = Chem.MolFromSmiles("B(O)O")


def _convert_bh2_to_boronic(mol: Chem.Mol) -> Chem.Mol:
    if not mol.HasSubstructMatch(_BH2_QUERY):
        return mol
    new_mol = Chem.ReplaceSubstructs(mol, _BH2_QUERY, _BORONIC_REPL, replaceAll=True)[0]
    Chem.SanitizeMol(new_mol)
    return new_mol


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------
def _expand_sdf_entries(entries: Sequence[Entry]) -> tuple[list, int]:
    """Expand SDF file paths into Mol records; count *all* records, including failures."""
    expanded: list = []
    total = 0
    for e in entries:
        if isinstance(e, Chem.Mol):
            total += 1
            expanded.append(e)
            continue
        supp = Chem.SDMolSupplier(str(e), removeHs=False)
        total += len(supp)
        expanded.extend(supp)
    return expanded, total


def _to_valid_mols(
    entries: Sequence[Entry], input_mode: str
) -> tuple[list[Chem.Mol], list[str]]:
    valid_mols: list[Chem.Mol] = []
    valid_smiles: list[str] = []
    for entry in tqdm(entries, desc="Validating"):
        if entry is None:
            continue
        if input_mode in ("smiles", "df_from_sample"):
            mol = Chem.MolFromSmiles(str(entry))
        elif isinstance(entry, Chem.Mol):
            mol = Chem.Mol(entry)
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
        else:
            mol = None
        if mol is None:
            continue
        if input_mode == "sdf":
            mol = Chem.AddHs(mol, addCoords=True)
        valid_mols.append(mol)
        valid_smiles.append(Chem.MolToSmiles(mol, canonical=True))
    return valid_mols, valid_smiles


def _prepare_inputs(
    entries: Sequence[Entry],
    input_mode: str,
    subsample: Optional[int],
) -> tuple[list[Chem.Mol], list[str], int]:
    """Returns (valid_mols, valid_smiles, total_raw_count) — total includes invalids."""
    if input_mode == "sdf":
        entries, total_mols = _expand_sdf_entries(entries)
    else:
        total_mols = len(entries)

    if subsample is not None and subsample < len(entries):
        random.seed(10)
        entries = random.sample(list(entries), subsample)
        total_mols = len(entries)
        print(f"subsampled raw entries to {len(entries)} molecules")

    entries = [m for m in entries if m]
    valid_mols, valid_smiles = _to_valid_mols(entries, input_mode)
    return valid_mols, valid_smiles, total_mols


# -----------------------------------------------------------------------------
# Diversity (global or per-pharmacophore)
# -----------------------------------------------------------------------------
def _compute_diversity(
    valid_mols: Sequence[Chem.Mol],
    valid_smiles: Sequence[str],
    pharm_ids: Optional[Sequence[int]],
    diversity_thresh: float,
) -> tuple[float, float, float]:
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    if pharm_ids is None:
        fps = [fpgen.GetFingerprint(m) for m in valid_mols]
        intdiv = _internal_diversity(fps)
        clust_div = _sphere_exclusion_diversity(fps, diversity_thresh)
        uniqueness = 100.0 * len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0.0
        print(f"diversity cluster {clust_div} and intdiv {intdiv}")
        return intdiv, clust_div, uniqueness

    groups_mols: dict = collections.defaultdict(list)
    groups_smiles: dict = collections.defaultdict(list)
    for m, pid, smi in zip(valid_mols, pharm_ids, valid_smiles):
        groups_mols[pid].append(m)
        groups_smiles[pid].append(smi)

    int_divs, cluster_divs, unique_by_pharm = [], [], []
    for pid, mols in groups_mols.items():
        fps = [fpgen.GetFingerprint(m) for m in mols]
        if not fps:
            continue
        int_divs.append(_internal_diversity(fps))
        cluster_divs.append(_sphere_exclusion_diversity(fps, diversity_thresh))
        smis = groups_smiles[pid]
        unique_by_pharm.append(100.0 * len(set(smis)) / len(smis) if smis else 0.0)

    intdiv = float(mean(int_divs)) if int_divs else 0.0
    clust_div = float(mean(cluster_divs)) if cluster_divs else 0.0
    uniqueness = float(mean(unique_by_pharm)) if unique_by_pharm else 0.0
    print(
        f"[per-pharmacophore] diversity cluster avg {clust_div} and "
        f"intdiv avg {intdiv}, uniqueness avg {uniqueness}"
    )
    return intdiv, clust_div, uniqueness


# -----------------------------------------------------------------------------
# Synthesizability (SA + optional SC + optional AiZynth)
# -----------------------------------------------------------------------------
class Synthesizability:
    """Evaluate synthesizability via SA-score, optional SC-score, and optional AiZynthFinder."""

    def __init__(
        self,
        *,
        enable_aizynth: bool = True,
        aizynth_config: Optional[Union[str, Path]] = None,
        aizynth_limit: int = 100,
        scscore_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._enable_aizynth = enable_aizynth
        self._aizynth_limit = aizynth_limit
        if enable_aizynth:
            if aizynth_config is None:
                raise ValueError("AiZynthFinder enabled but no config path provided.")
            self._finder = AiZynthFinder(configfile=str(aizynth_config))
            self._finder.stock.select("zinc")
            self._finder.expansion_policy.select("uspto")
            self._finder.filter_policy.select("uspto")
            print("Finished initializing aizynthfinder")

        RDLogger.DisableLog("rdApp.*")

    def _aizynth_one(self, smi: str) -> bool:
        self._finder.target_smiles = smi
        self._finder.tree_search()
        self._finder.build_routes()
        stats = self._finder.extract_statistics()
        print(f"aizynth stats: {stats}")
        return bool(stats["is_solved"])

    def __call__(self, mols: Iterable[Chem.Mol]) -> Dict[str, Any]:
        mols = [_convert_bh2_to_boronic(Chem.RemoveHs(m)) for m in mols]
        random.shuffle(mols)

        print("computing sa...")
        sa_scores: List[float] = [
            float(sascorer.calculateScore(m))
            for m in mols
        ]

        aizynth_solved_rate: Optional[float] = None
        aizynth_attempted: Optional[int] = None
        if self._enable_aizynth:
            print("running aizynthfinder...")
            targets = mols[: self._aizynth_limit]
            solved = 0
            for mol in tqdm(targets, desc="AiZynthFinder"):
                smi = Chem.MolToSmiles(mol, canonical=True)
                if self._aizynth_one(smi):
                    solved += 1
            aizynth_attempted = len(targets)
            aizynth_solved_rate = solved / aizynth_attempted if aizynth_attempted else None

        return {
            "sa_scores": sa_scores,
            "aizynth_solved_rate": aizynth_solved_rate,
        }


# -----------------------------------------------------------------------------
# Full evaluation
# -----------------------------------------------------------------------------
def evaluate(
    entries: Sequence[Entry],
    pharm_ids: Optional[Sequence[int]] = None,
    *,
    input_mode: str = "smiles",
    diversity_thresh: float = 0.65,
    training_smiles: Optional[Sequence[str]] = None,
    calc_fcd: bool = False,
    aizynth: bool = False,
    aizynth_config: Optional[str] = None,
    aizynth_limit: int = 100,
    subsample: Optional[int] = None,
) -> Dict[str, Any]:
    valid_mols, valid_smiles, total_mols = _prepare_inputs(entries, input_mode, subsample)
    validity = 100.0 * len(valid_mols) / total_mols if total_mols else 0.0
    print(f"validity {validity}")

    print("computing diversity...")
    intdiv, clust_div, uniqueness = _compute_diversity(
        valid_mols, valid_smiles, pharm_ids, diversity_thresh
    )

    mols_with_confs = [m for m in valid_mols if m.GetNumConformers() > 0]
    if mols_with_confs:
        print("computing energy...")
        mmff_energies = [_compute_mmff_energy(m) for m in tqdm(mols_with_confs, desc="MMFF energy")]
    else:
        mmff_energies = []

    exact_nov: Optional[float] = None
    num_novel: Optional[int] = None
    fcd: Optional[float] = None
    if training_smiles:
        print("computing novelty...")
        canon_train = []
        for smi in tqdm(training_smiles, desc="Canonicalizing training SMILES"):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canon_train.append(Chem.MolToSmiles(mol, canonical=True))
        canon_set = set(canon_train)
        novel = [s for s in valid_smiles if s not in canon_set]
        exact_nov = 100.0 * len(novel) / len(valid_smiles) if valid_smiles else 0.0
        num_novel = len(novel)
        if calc_fcd:
            import torch
            from fcd_torch import FCD
            print("computing FCD...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            fcd = FCD(device=device, n_jobs=1)(ref=canon_train, gen=valid_smiles)

    print("computing qed...")
    qed_vals = [QED.qed(m) for m in valid_mols]

    synth_results = Synthesizability(
        enable_aizynth=aizynth,
        aizynth_config=aizynth_config,
        aizynth_limit=aizynth_limit,
    )(valid_mols)

    return {
        "validity_%": validity,
        "uniqueness_%": uniqueness,
        "intdiv": intdiv,
        "cluster_diversity_%": 100.0 * clust_div,
        "avg_sa_score": _mean(synth_results["sa_scores"]),
        "avg_mmff_energy": _mean(mmff_energies),
        "aizynth_solved_rate": synth_results["aizynth_solved_rate"],
        "avg_qed": float(np.mean(qed_vals)) if qed_vals else None,
        "exact_novelty_%": exact_nov,
        "fcd": fcd,
        "num_input": total_mols,
        "num_valid": len(valid_mols),
        "num_unique": len(set(valid_smiles)),
        "num_novel": num_novel,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _load_entries(input_mode: str, path: str) -> list:
    if input_mode == "smiles":
        with open(path, "r", encoding="utf-8") as f:
            entries = [line.strip() for line in f]
        random.shuffle(entries)
        return entries
    if input_mode == "sdf":
        p = Path(path)
        return [p] if p.is_file() else list(p.glob("*.sdf"))
    if input_mode == "pkl":
        with open(path, "rb") as pkl:
            return pickle.load(pkl)
    if input_mode == "df_from_sample":
        return pd.read_csv(path)["final_product_smiles"].tolist()
    raise ValueError(f"Unknown input_mode: {input_mode}")


def _load_training_smiles(args) -> Optional[list[str]]:
    if not args.training_smi:
        return None
    with open(args.training_smi, "r", encoding="utf-8") as f:
        return [
            line.split()[0]
            for line in f
            if line.strip() and not line.startswith("#")
        ]


def main():  # pragma: no cover -- thin CLI
    p = argparse.ArgumentParser(description="Evaluate generated molecules (2D/3D)")
    p.add_argument("--input", required=True, help="CSV with SMILES, or SDF/PKL file/dir")
    p.add_argument("--input_mode", choices=["smiles", "sdf", "pkl", "df_from_sample"], default="smiles")
    p.add_argument("--training_smi", help="training .smi: one SMILES per line")
    p.add_argument("--aizynth", action="store_true")
    p.add_argument("--aizynth_config", help="Path to the AiZynthFinder YAML config (stock/expansion/filter policies); required if --aizynth is set")
    p.add_argument("--aizynth_limit", type=int, default=100, help="Limit AiZynth to first N molecules")
    p.add_argument("--calc_fcd", action="store_true")
    p.add_argument("--out_json", default="results.json")
    p.add_argument("--subsample", type=int, default=None, help="optional max number of molecules to process")
    args = p.parse_args()

    entries = _load_entries(args.input_mode, args.input)
    training_smiles = _load_training_smiles(args)

    eval_kwargs = dict(
        input_mode=args.input_mode,
        aizynth=args.aizynth,
        aizynth_config=args.aizynth_config,
        aizynth_limit=args.aizynth_limit,
        subsample=args.subsample,
    )


    res = evaluate(
        entries,
        training_smiles=training_smiles,
        calc_fcd=args.calc_fcd,
        **eval_kwargs,
    )
    with open(args.out_json, "w", encoding="utf-8") as out:
        json.dump(res, out, indent=1)

    print("[done] results saved →", args.out_json)


if __name__ == "__main__":
    main()
