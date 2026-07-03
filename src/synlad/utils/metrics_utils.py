"""ROCS shape/colour scoring helpers built on the OpenEye toolkits."""

import logging
import pickle
from dataclasses import dataclass

# Hydra imports
from openeye import oechem, oeomega, oeshape
from rdkit import Chem, Geometry
from tqdm import tqdm

logger = logging.getLogger(__name__)


def score_smiles_to_query(smiles, query, metric_name="tanimoto"):
    assert metric_name in ["tanimoto", "tversky"], f"Invalid metric: {metric_name}"
    metric = oeshape.OEHighestTanimotoCombo()
    overlay_opts = oeshape.OEOverlayOptions()
    overlay_opts.SetShapeFuncType(oeshape.OEShapeType_Exact)
    overlay_opts.SetColorFuncType(oeshape.OEColorType_Exact)

    rocs_opts = oeshape.OEROCSOptions()
    rocs_opts.SetOverlayOptions(overlay_opts)
    rocs_opts.SetNumBestHits(1_000_000)
    rocs_opts.SetRankPredicate(metric)
    rocs = oeshape.OEROCS(rocs_opts)

    for smi in tqdm(smiles):
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smi)
        builder = oeomega.OEOmega(oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_ROCS))
        ret_code = builder.Build(mol)
        if ret_code != oeomega.OEOmegaReturnCode_Success:
            logger.warning("Conformer build failed")
            continue
        rocs.AddMolecule(mol)

    combo_scores = list()
    shape_scores = list()
    colour_scores = list()
    mols = list()
    for res in rocs.Overlay(query):
        if metric_name == "tanimoto":
            combo_scores.append(res.GetTanimotoCombo())
            shape_scores.append(res.GetTanimoto())
            colour_scores.append(res.GetColorTanimoto())
        elif metric_name == "tversky":
            combo_scores.append(res.GetFitTverskyCombo())
            shape_scores.append(res.GetTanimoto())
            colour_scores.append(res.GetColorTanimoto())
        else:
            raise AssertionError(f"Invalid metric: {metric_name}")
        mols.append(res.GetOverlayConf())
    return combo_scores, shape_scores, colour_scores, mols


def rdmol_to_oemol(rdmol: Chem.Mol) -> oechem.OEGraphMol:
    """Convert RDKit mol to OpenEye mol."""
    mol_block = Chem.MolToMolBlock(rdmol)
    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SDF)
    ims.openstring(mol_block)
    oemol = oechem.OEGraphMol()
    if not oechem.OEReadMolecule(ims, oemol):
        raise ValueError("Could not read molecule from string")
    return oemol


def get_rocs_score(pred_mol: Chem.Mol, ref_mol: Chem.Mol):
    """
    Compute ROCS scores between predicted and ground truth molecules.
    """
    pred_oemol = rdmol_to_oemol(pred_mol)
    ref_oemol = rdmol_to_oemol(ref_mol)

    res = oeshape.OEROCSResult()
    oeshape.OEROCSOverlay(res, pred_oemol, ref_oemol)
    combo_score = res.GetTanimotoCombo()
    shape_score = res.GetTanimoto()
    colour_score = res.GetColorTanimoto()
    return combo_score, shape_score, colour_score


def get_rocs_score_oemols(pred_oemol: oechem.OEGraphMol, ref_oemol: oechem.OEGraphMol):
    """
    Compute ROCS scores between predicted and ground truth molecules.
    """

    res = oeshape.OEROCSResult()
    oeshape.OEROCSOverlay(res, pred_oemol, ref_oemol)
    combo_score = res.GetTanimotoCombo()
    shape_score = res.GetTanimoto()
    colour_score = res.GetColorTanimoto()
    return combo_score, shape_score, colour_score


@dataclass
class ConformerConfig:
    """Configuration for conformer generation."""

    rxn_network: str = None  # Input reaction network file path (pickle format)
    output_pickle: str = None  # Output pickle file path for conformers
    single: bool = False  # Generate only single lowest-energy conformer
    check_pb: bool = False  # Check if conformers are valid using PoseBusters
    max_conformers: int = 10  # Maximum number of conformers to generate
    strict_stereo: bool = False  # Use strict stereochemistry in Omega

    # Size filtering options
    min_heavy_atoms: int = 10
    max_heavy_atoms: int = 30
    check_size: bool = False  # Whether to apply size filtering


def create_molecule_from_smiles(smiles_string):
    """
    Create an OEMol object from a SMILES string.

    Args:
        smiles_string (str): The SMILES string

    Returns:
        OEMol: The molecule object, or None if invalid SMILES
    """
    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, smiles_string):
        return None

    # Set the molecule title to the SMILES string for identification
    mol.SetTitle(smiles_string)
    return mol


def check_size(mol, min_atoms=10, max_atoms=27):
    """
    Check if molecule size is within specified heavy atom range.

    Args:
        mol (OEMol): The molecule to check
        min_atoms (int): Minimum number of heavy atoms
        max_atoms (int): Maximum number of heavy atoms

    Returns:
        bool: True if molecule is within size range, False otherwise
    """
    num_atoms = mol.NumAtoms()
    return min_atoms <= num_atoms <= max_atoms


def setup_omega_options(cfg: ConformerConfig):
    """
    Set up OEOmega options for conformer generation.

    Args:
        cfg (ConformerConfig): Configuration object

    Returns:
        OEOmegaOptions: Configured options object
    """

    omega_opts = oeomega.OEOmegaOptions()

    omega_opts.GetTorDriveOptions().SetUseGPU(True)

    # Set maximum number of conformers
    if cfg.single:
        omega_opts.SetMaxConfs(1)  # Single lowest-energy conformer
    else:
        omega_opts.SetMaxConfs(cfg.max_conformers)  # Up to specified number of conformers

    # Set stereochemistry strictness
    omega_opts.SetStrictStereo(cfg.strict_stereo)

    return omega_opts


def generate_conformers(mol, cfg: ConformerConfig):
    """
    Generate conformers for a molecule using OEOmega.

    Args:
        mol (OEMol): The input molecule
        cfg (ConformerConfig): Configuration object

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    # Set up Omega options
    omega_opts = setup_omega_options(cfg)
    omega = oeomega.OEOmega(omega_opts)

    # Generate conformers
    ret_code = omega.Build(mol)

    if ret_code == oeomega.OEOmegaReturnCode_Success:
        return True, None
    else:
        error_msg = oeomega.OEGetOmegaError(ret_code)
        return False, error_msg


def oemol_to_rdkit(oemol):
    """
    Convert an OpenEye molecule with conformers to RDKit molecule with conformers.
    Based on conversion utilities found in the codebase.

    Args:
        oemol (OEMol): OpenEye molecule with conformers

    Returns:
        Chem.Mol: RDKit molecule with conformers, or None if conversion fails
    """
    try:
        # Start with an empty RDKit molecule
        rdmol = Chem.RWMol()

        # Bond type mapping from OpenEye to RDKit
        _bondtypes = {
            1: Chem.BondType.SINGLE,
            1.5: Chem.BondType.AROMATIC,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.QUADRUPLE,
            5: Chem.BondType.QUINTUPLE,
            6: Chem.BondType.HEXTUPLE,
            7: Chem.BondType.ONEANDAHALF,
        }

        # Create atom mapping
        map_atoms = {}  # {oe_idx: rd_idx}
        for oea in oemol.GetAtoms():
            oe_idx = oea.GetIdx()
            rda = Chem.Atom(oea.GetAtomicNum())
            rda.SetFormalCharge(oea.GetFormalCharge())
            rda.SetIsAromatic(oea.IsAromatic())

            # Set chirality
            cip = oechem.OEPerceiveCIPStereo(oemol, oea)
            if cip == oechem.OECIPAtomStereo_S:
                rda.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
            elif cip == oechem.OECIPAtomStereo_R:
                rda.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)

            map_atoms[oe_idx] = rdmol.AddAtom(rda)

        # Add bonds
        for oeb in oemol.GetBonds():
            rd_a1 = map_atoms[oeb.GetBgnIdx()]
            rd_a2 = map_atoms[oeb.GetEndIdx()]

            rdmol.AddBond(rd_a1, rd_a2)
            rdbond = rdmol.GetBondBetweenAtoms(rd_a1, rd_a2)

            # Set bond type
            order = oeb.GetOrder()
            if oeb.IsAromatic():
                rdbond.SetBondType(_bondtypes[1.5])
                rdbond.SetIsAromatic(True)
            else:
                rdbond.SetBondType(_bondtypes[order])
                rdbond.SetIsAromatic(False)

        # Add conformers from OpenEye molecule
        for conf in oemol.GetConfs():
            conformer = Chem.Conformer(rdmol.GetNumAtoms())
            for oe_idx, rd_idx in map_atoms.items():
                pos = conf.GetCoords()[oe_idx]
                conformer.SetAtomPosition(rd_idx, Geometry.Point3D(pos[0], pos[1], pos[2]))
            rdmol.AddConformer(conformer, assignId=True)

        # Set molecule title
        rdmol.SetProp("_Name", oemol.GetTitle())

        # Cleanup and finalize
        rdmol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(rdmol)
        Chem.AssignStereochemistry(rdmol, force=False)

        return rdmol.GetMol()

    except Exception as e:
        logger.error(f"Error converting OpenEye molecule to RDKit: {e}")
        return None


def write_oemol_conformers_to_sdf(mol, output_path):
    """
    Write molecule conformers to an SDF file.

    Args:
        mol (OEMol): The molecule with conformers
        output_path (str): Path to output SDF file

    Returns:
        bool: True if successful, False otherwise
    """
    ofs = oechem.oemolostream()
    if not ofs.open(output_path):
        return False

    # Verify output format supports 3D coordinates
    if not oechem.OEIs3DFormat(ofs.GetFormat()):
        logger.error("Output format does not support 3D coordinates")
        return False

    # Write the molecule (with all its conformers) to the file
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()
    return True


def write_rdkit_mol_conformers_to_pickle(mol, output_path):
    """
    Write RDKit molecule with all conformers to a pickle file.

    Args:
        mol (Chem.Mol): RDKit molecule with conformers
        output_path (str): Path to output pickle file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(output_path, "wb") as f:
            pickle.dump(mol, f)
        return True
    except Exception as e:
        logger.error(f"Error writing conformers to pickle file: {e}")
        return False


def load_rdkit_mol_conformers_from_pickle(input_path):
    """
    Load RDKit molecule with all conformers from a pickle file.

    Args:
        input_path (str): Path to input pickle file

    Returns:
        Chem.Mol: RDKit molecule with conformers, or None if failed
    """
    try:
        with open(input_path, "rb") as f:
            mol = pickle.load(f)
        return mol
    except Exception as e:
        logger.error(f"Error loading conformers from pickle file: {e}")
        return None


def write_rdkit_mol_conformers_to_sdf(mol, output_path):
    """
    Write RDKit molecule with conformers to an SDF file.

    NOTE: SDF format writes ALL conformers as separate molecule entries.
    Use write_rdkit_mol_conformers_to_pickle() to preserve multi-conformer structure.
    """
    try:
        # For multi-conformer molecules, write each conformer as a separate entry
        writer = Chem.SDWriter(output_path)
        for i in range(mol.GetNumConformers()):
            # Create a copy with only this conformer
            mol_copy = Chem.Mol(mol)
            conf_ids = [conf.GetId() for conf in mol_copy.GetConformers()]
            for conf_id in conf_ids:
                if conf_id != i:
                    mol_copy.RemoveConformer(conf_id)

            # Set conformer properties
            mol_copy.SetProp(
                "_Name", f"{mol.GetProp('_Name') if mol.HasProp('_Name') else 'mol'}_conf_{i}"
            )
            mol_copy.SetProp("ConformerID", str(i))

            writer.write(mol_copy)
        writer.close()
        return True
    except Exception as e:
        logger.error(f"Error writing conformers to SDF file: {e}")
        return False
