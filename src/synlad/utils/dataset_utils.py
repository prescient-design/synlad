"""Dataset preprocessing utilities for converting pathway pickles to graph batches."""

import logging
from collections import defaultdict

import torch
from rdkit import Chem
from tqdm import tqdm

logger = logging.getLogger(__name__)

PADDING_INDEX = 999.0


def count_atoms(smiles_list):
    atom_count = defaultdict(int)

    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.RemoveAllHs(mol)
        for atom in mol.GetAtoms():
            atom_count[atom.GetSymbol()] += 1

    return atom_count


def count_molecules_per_atom_type(smiles_list):
    molecule_count = defaultdict(int)

    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        atom_types = {atom.GetSymbol() for atom in mol.GetAtoms()}

        for atom_type in atom_types:
            molecule_count[atom_type] += 1

    return molecule_count


def parseSDF(SDFFile):
    fil = SDFFile
    totcoords = []
    totaname = []
    coords = []
    atomNames = []
    for line in open(fil).readlines():
        a = line.strip().split()
        if len(a) == 16:  ## atom
            element = a[3]
            x = float(a[0])
            y = float(a[1])
            z = float(a[2])
            coords += [[x, y, z]]
            aname = "MOL" + "_" + "0" + "_" + element + "_" + "A"

            atomNames += [aname]
        elif "$$$$" in line:
            totcoords += [torch.tensor(coords)]
            totaname += [atomNames]
            coords = []
            atomNames = []
    return totcoords, totaname


def parsePharmacophoreSDFOnly(SDFFile, ph4_list):
    """Parse SDF file and extract only pharmacophore features."""
    fil = SDFFile
    totcoords = []
    totaname = []
    coords = []
    atomNames = []
    for line in open(fil).readlines():
        a = line.strip().split()
        if len(a) == 16:  ## atom
            element = a[3]
            # Only keep pharmacophore atom types
            if element in ph4_list:
                x = float(a[0])
                y = float(a[1])
                z = float(a[2])
                coords += [[x, y, z]]
                atomNames += [element]  # Direct pharmacophore type
        elif "$$$$" in line:
            if coords:  # Only add if we found pharmacophore atoms
                totcoords += [torch.tensor(coords)]
                totaname += [atomNames]
            coords = []
            atomNames = []
    return totcoords, totaname


def atomlistToChannels(atomNames, hashing, device="cpu"):
    assert type(hashing) is dict
    if isinstance(hashing[list(hashing.keys())[0]], dict):
        useResName = True
    else:
        useResName = False
        assert isinstance(hashing[list(hashing.keys())[0]], int)
    channels = []
    for singleAtomList in atomNames:
        haTMP = []
        for i in singleAtomList:
            resname = i.split("_")[0]
            atName = i.split("_")[2]
            if useResName:
                if resname in hashing and atName in hashing[resname]:
                    haTMP += [hashing[resname][atName]]
                else:
                    haTMP += [PADDING_INDEX]
                    logger.warning(f"missing {resname} {atName}")
            else:
                if atName in hashing:
                    haTMP += [hashing[atName]]
                elif atName[0] in hashing:
                    haTMP += [hashing[atName[0]]]
                elif hashing == "Element_Hashing":
                    haTMP += [6]
                else:
                    haTMP += [PADDING_INDEX]
                    logger.warning(f"missing {resname} {atName}")

        channels += [torch.tensor(haTMP, dtype=torch.float, device=device)]
    return channels


def pharmacophoreToChannels(atomNames, hashing, device="cpu"):
    """Convert pharmacophore atom names directly to channels."""
    channels = []
    for singleAtomList in atomNames:
        haTMP = []
        for atom_type in singleAtomList:
            if atom_type in hashing:
                haTMP += [hashing[atom_type]]
            else:
                haTMP += [PADDING_INDEX]
                logger.warning(f"missing pharmacophore type: {atom_type}")
        channels += [torch.tensor(haTMP, dtype=torch.float, device=device)]
    return channels


def atomlistToRadius(atomList, hashing, device="cpu"):
    assert type(hashing) is dict

    radius = []
    for singleAtomList in atomList:
        haTMP = []
        for i in singleAtomList:
            resname = i.split("_")[0]
            atName = i.split("_")[2]
            if resname in hashing and atName in hashing[resname]:
                haTMP += [hashing[resname][atName]]
            else:
                haTMP += [1.0]
                logger.warning(f"missing {resname} {atName}")
        radius += [torch.tensor(haTMP, dtype=torch.float, device=device)]
    return radius
