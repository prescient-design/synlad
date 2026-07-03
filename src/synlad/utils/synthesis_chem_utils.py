"""A set of methods for manipulating and obtaining properties from SMILES strings/RDKit molecules.

from: https://github.com/john-bradshaw/rxn-splits/blob/main/rxn_splits/chem_utils.py and then adapted
"""

import collections
import functools
import itertools
import re
import typing
import warnings

from rdkit import Chem
from rdkit.Chem import Lipinski, rdqueries

smiles_tokenizer_pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|\%\([0-9]{3}\)|[0-9])"
# ^ added the 3 digit bond connection number
# from Schwaller, P., Gaudin, T., Lanyi, D., Bekas, C. and Laino, T. (2018) ‘“Found in Translation”: Predicting
# Outcomes of Complex Organic Chemistry Reactions using Neural Sequence-to-Sequence Models’, Chemical science , 9(28),
# pp. 6091–6098.
regex_smiles_tokenizer = re.compile(smiles_tokenizer_pattern)

isCAtomsQuerier = rdqueries.AtomNumEqualsQueryAtom(6)


def is_smiles(token: str) -> bool:
    """Check if a token is a valid SMILES string.

    Tokens beginning with ``<`` (e.g. ``<BOS>``, ``<PAD>``) are treated as
    non-SMILES special tokens and rejected without attempting an RDKit parse.
    """
    if token.startswith("<"):
        # not a valid SMILES string, added this check as we are often distinguishing between
        # SMILES and other tokens
        return False
    try:
        mol = Chem.MolFromSmiles(token)
        return mol is not None
    except Exception:
        return False


def num_heavy_atoms(iter_of_smiles: typing.Iterable[str]) -> int:
    """Sum the heavy-atom count across a collection of SMILES strings."""
    return sum([Lipinski.HeavyAtomCount(Chem.MolFromSmiles(smi)) for smi in iter_of_smiles])


def at_least_one_carbon(iter_of_smiles: typing.Iterable[str]) -> bool:
    """Return True if any SMILES in the collection contains at least one carbon atom."""
    return any(
        [
            len(Chem.MolFromSmiles(smi).GetAtomsMatchingQuery(isCAtomsQuerier)) >= 1
            for smi in iter_of_smiles
        ]
    )


def max_number_of_bonds_per_mol(iter_of_smiles: typing.Iterable[str]) -> int:
    """Return the largest bond count across a collection of SMILES strings."""
    return max([len(list(Chem.MolFromSmiles(smi).GetBonds())) for smi in iter_of_smiles])


def at_least_one_larger_product_different(
    reactant_frozenset, product_frozenset, canonicalize_flag=True, num_heavy_atom_required=2
):
    """
    Check that at least one of the products is not in the reactant frozen set, *and* that this given product has a
    certain number of heavy atoms.

    nb this function is currently multiset naive
    """

    def canon_or_orig(smi):
        # ^ tries to canonicalize, if None then will return the original SMILES string though.
        return canonicalize(smi) or smi

    canon_op = canon_or_orig if canonicalize_flag else lambda x: x
    reactants_set = {canon_op(el[0]) for el in reactant_frozenset}
    for prod, _ in product_frozenset:
        # ^ ignore count, i.e., second element -- see function docstring
        if prod not in reactants_set:
            mol = Chem.MolFromSmiles(prod)
            num_heavy_atoms = mol.GetNumHeavyAtoms()
            if num_heavy_atoms >= num_heavy_atom_required:
                return True
    return False


def not_deprotonation(reactant_frozenset, product_frozenset):
    """
    Checks the reaction is not a simple (de)protonation

    Do this by putting each molecule into a canonical neutralized form and then

    Note currently naive wrt multisets. Also note that have not exhaustively checked this function,
    we expect to have a decent number of false negatives, i.e., low recall, due to the limited set of hardcoded rules
    we check against.
    """
    neutralized_reactants = {try_neutralize_smi(smi) for smi, count in reactant_frozenset}
    neutralized_products = {try_neutralize_smi(smi) for smi, count in product_frozenset}
    if (len(neutralized_products) == 0) or (len(neutralized_products - neutralized_reactants) == 0):
        return False
    else:
        return True


CHARGED_ATOM_PATTERN = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
# i.e. +1 charge, at least one hydrogen, and not linked to negative charged atom; or -1 charge and not linked to
# positive atom


def try_neutralize_smi(smi: str, canonical: bool = True, log=None) -> str:
    """Return a neutralized SMILES, or the original string if neutralization fails.

    Failures emit a warning and the input is returned unchanged so callers can
    treat the result as best-effort rather than authoritative.
    """
    mol = Chem.MolFromSmiles(smi)
    try:
        mol = neutralize_atoms(mol)
    except Exception:
        err_str = f"Failed to neutralize {smi}"
        warnings.warn(err_str, stacklevel=2)
    else:
        smi = Chem.MolToSmiles(mol, canonical=canonical)
    return smi


def neutralize_atoms(mol):
    """
    from http://www.rdkit.org/docs/Cookbook.html
    note changed so that returns a RWCopy
    """
    mol = Chem.RWMol(mol)
    at_matches = mol.GetSubstructMatches(CHARGED_ATOM_PATTERN)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def remove_isotope_info_from_mol_in_place(mol):
    """
    adapted from https://www.rdkit.org/docs/Cookbook.html#isomeric-smiles-without-isotopes
    see limitations at link about needing to canonicalize _after_.
    """
    atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in atom_data:
        if isotope:
            atom.SetIsotope(0)
    return mol


def clean_smiles_reaction(smiles_in, fail_on_canon=True, remove_reagents=False):
    """
    Creates a 2-element tuple of frozen sets of counters -- reactants and products.

    Does the following "cleaning" operations:
    - remove atom mapping
    - canonicalizes the molecules
    - puts the already classified reagents with the reactants *if* `remove_reagents` is `True`
    - splits the individual molecules up and puts them in counters
    """
    smiles_in = smiles_in.split()[0]  # some will have an end part, which we will ignore for now.
    reactants, reagents, products = smiles_in.split(">")
    reactants_cc = create_canon_counter(reactants, fail_on_canon)
    if not remove_reagents:
        reactants_cc = reactants_cc + create_canon_counter(reagents, fail_on_canon)
    products = create_canon_counter(products, fail_on_canon)
    return frozenset(reactants_cc.items()), frozenset(products.items())


def create_canon_counter(smiles_in: str, fail_on_canon: bool = False) -> "collections.Counter[str]":
    """Split a dot-joined SMILES string and return a Counter of canonical fragments.

    With ``fail_on_canon=True``, any fragment that fails to canonicalize raises
    ``ValueError``; otherwise the original (non-canonical) SMILES is kept.
    """
    smiles_of_each_mol = smiles_in.split(".")
    if fail_on_canon:

        def can_(smi):
            can = canonicalize(smi)
            if can is None:
                raise ValueError(f"Failed to canonicalize {smi}")
            return can
    else:

        def can_(smi):
            return canonicalize(smi) or smi

    canon_smiles = filter(len, map(can_, smiles_of_each_mol))
    counter = collections.Counter(canon_smiles)
    return counter


@functools.lru_cache(maxsize=100_000)
def canonicalize(
    smi: str,
    remove_hs: bool = True,
    remove_atm_map: bool = True,
    kekuleSmiles: bool = False,
    remove_isotope_info=True,
    rounds: int = 2,
    **otherargs,
) -> str | None:
    """
    Canonicalizes molecule (returns `None` if invalid SMILES).
    Rounds is the number of times to canonicalize -- sometimes RDKit takes a while to converge (particularly if
    atom maps are used).
    """
    # v put this check at the top so happens even if rounds is 0
    if ("<-" in smi) or ("->" in smi):
        # ^ for some reason, RDKIT 2023+ seems to put these symbols in the SMILES string. Eg with
        # O=C1O[Cu]2(OC(=O)c3ccc4ccccc4n->23)<-n2c1ccc1ccccc12. Not entirely sure what they are so removing for now.
        return None
    if rounds == 0:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if remove_hs:
        mol = Chem.RemoveHs(mol)
    if remove_isotope_info and mol is not None:
        mol = remove_isotope_info_from_mol_in_place(mol)
    if remove_atm_map:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    smi = Chem.MolToSmiles(
        mol, canonical=True, ignoreAtomMapNumbers=remove_atm_map, kekuleSmiles=kekuleSmiles
    )
    return canonicalize(
        smi, remove_hs, remove_atm_map, kekuleSmiles, remove_isotope_info, rounds - 1
    )


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction

    from: https://github.com/pschwllr/MolecularTransformer
    """
    tokens = [token for token in regex_smiles_tokenizer.findall(smi)]
    and_back = "".join(tokens)
    if smi != and_back:
        raise RuntimeError(f"{smi} was tokenized incorrectly to {tokens}")
    return " ".join(tokens)


class InconsistentSMILES(RuntimeError):
    """Raised when a SMILES string fails an internal consistency check (e.g. atom-mapping)."""

    pass


def get_atom_map_nums(mol_str, accept_invalid=False) -> typing.Iterator[int]:
    """
    :return: iterator of the atom mapping numbers of the atoms in the reaction string
    """
    mol = Chem.MolFromSmiles(mol_str)
    if accept_invalid and mol is None:
        warnings.warn(f"Invalid molecule passed to get_atom_map_nums {mol_str}", stacklevel=2)
        return iter([])
    return (
        int(a.GetProp("molAtomMapNumber")) for a in mol.GetAtoms() if a.HasProp("molAtomMapNumber")
    )


def _not_unique_elements(itr_in) -> bool:
    lst_ = list(itr_in)
    return True if len(lst_) != len(set(lst_)) else False


def get_changed_bonds(rxn_smi, strict_mode=False) -> list[tuple[int, int, float]]:
    """
    Note unless `strict_mode` is `True`, this method does not check that the reagents have been seperated correctly
    (i.e., reagent atoms are not in the products) or that atom map numbers have been repeated in the reactants or
    products, which would break the rest of the implementation in a silent manner. If `strict_mode` is `True` and these
    conditions are violated then an `InconsistentSMILES` is raised.
    API should match the alternative:
    https://github.com/connorcoley/rexgen_direct/blob/master/rexgen_direct/scripts/prep_data.py

    Caution: Experimental!

    :return: list of tuples of atom map numbers for each bond and new bond order
    """
    # 1. Split into reactants and products.
    reactants_smi, reagents_smi, products_smi = rxn_smi.split(">")

    # 2. If necessary check for repeated atom map numbers in the SMILES and products.
    if strict_mode:
        if _not_unique_elements(get_atom_map_nums(reactants_smi)):
            raise InconsistentSMILES(f"Repeated atom map numbers in reactants: {rxn_smi}")
        if _not_unique_elements(get_atom_map_nums(products_smi)):
            raise InconsistentSMILES(f"Repeated atom maps numbers in products {rxn_smi}")

    # 3. Get the bonds and their types in reactants and products
    bonds_prev = {}
    bonds_new = {}
    for bond_dict, bonds in [
        (bonds_prev, Chem.MolFromSmiles(reactants_smi).GetBonds()),
        (bonds_new, Chem.MolFromSmiles(products_smi).GetBonds()),
    ]:
        for bond in bonds:
            try:
                bond_atmmap = frozenset(
                    (
                        int(bond.GetBeginAtom().GetProp("molAtomMapNumber")),
                        int(bond.GetEndAtom().GetProp("molAtomMapNumber")),
                    )
                )
            except KeyError:
                continue
            bond_dict[bond_atmmap] = float(bond.GetBondTypeAsDouble())

    # 4. Go through the bonds before and after...
    bond_changes: list[tuple[int, int, float]] = []
    product_atmmap_nums = set(get_atom_map_nums(products_smi))
    if strict_mode and (len(set(get_atom_map_nums(reagents_smi)) & product_atmmap_nums) > 0):
        raise InconsistentSMILES(f"Reagent atoms end up in products: {rxn_smi}")
    for bnd in {*bonds_prev, *bonds_new}:
        bnd_different_btwn_reacts_and_products = not (
            bonds_prev.get(bnd, None) == bonds_new.get(bnd, None)
        )
        bnd_missing_in_products = len(bnd & product_atmmap_nums) == 0

        # ... and if a bond has (a) changed or (b) is half missing in the products then it must have changed!
        if bnd_different_btwn_reacts_and_products and (not bnd_missing_in_products):
            bond_changes.append((*sorted(list(bnd)), bonds_new.get(bnd, 0.0)))
            # ^ note if no longer in products then new order is 0.

    return bond_changes


def split_reagents_out_from_reactants_and_products(
    rxn_smi, strict_mode=False
) -> tuple[str, str, str]:
    """
    Splits reaction into reactants, reagents, and products. Can deal with reagents in reactants part of SMILES string.
    Note that this method expects relatively well done atom mapping.
    Reagent defined as either:
    1. in the middle part of reaction SMILES string, i.e. inbetween the `>` tokens.
    2. in the reactants part of the SMILES string and all of these are true:
            a. no atoms in the product(s).
            b. not involved in the reaction center (atoms for which bonds change before and after) -- depending on the
                center identification code this will be covered by a, but is also checked to allow for cases where
                center can include information about changes in say a reactant that results in two undocumented minor
                products.
            c. reaction has been atom mapped (i.e., can we accurately check conditions a and b) -- currently judged by
                a center being able to be identified.
    3. in the reactants and products part of the SMILES string and both:
            a. not involved in reaction center
            b. unchanged before and after the reaction (comparing with canonicalized, atom-map removed strings)

    Caution: Experimental!

    :param rxn_smi: the reaction SMILES string
    :param strict_mode: whether to run `get_changed_bonds` in strict mode when determining atom map numbers involved in
                        center and whether to allow `get_atom_map_nums` to deal with invalid molecules.
    :return: tuple of reactants, reagents, and products
    """

    # 1. Split up reaction and get involved atom counts.
    reactant_all_str, reagents_all_str, product_all_str = rxn_smi.split(">")
    atoms_involved_in_reaction = set(
        itertools.chain(
            *[(int(el[0]), int(el[1])) for el in get_changed_bonds(rxn_smi, strict_mode)]
        )
    )
    reactants_str = reactant_all_str.split(".")
    products_str = product_all_str.split(".")
    products_to_keep = collections.Counter(products_str)
    product_atom_map_nums = functools.reduce(
        lambda x, y: x | y, (set(get_atom_map_nums(prod, not strict_mode)) for prod in products_str)
    )
    reaction_been_atom_mapped = len(atoms_involved_in_reaction) > 0

    # 2. Store map from canonical products to multiset of their SMILES in the products --> we will class
    canon_products_to_orig_prods = collections.defaultdict(collections.Counter)
    for smi in products_str:
        canon_products_to_orig_prods[canonicalize(smi)].update([smi])

    # 3. Go through the remaining reactants and check for conditions 2 or 3.
    reactants = []
    reagents = reagents_all_str.split(".") if reagents_all_str else []
    for candidate_reactant in reactants_str:
        atom_map_nums_in_candidate_reactant = set(
            get_atom_map_nums(candidate_reactant, not strict_mode)
        )

        # compute some flags useful for checks 2 and 3
        # 2a any atoms in products
        not_in_product = len(list(product_atom_map_nums & atom_map_nums_in_candidate_reactant)) == 0
        # 2b/3a any atoms in reaction center
        not_in_center = (
            len(list(set(atoms_involved_in_reaction & atom_map_nums_in_candidate_reactant))) == 0
        )

        # Check 2.
        if reaction_been_atom_mapped and not_in_product and not_in_center:
            reagents.append(candidate_reactant)
            continue

        # Check 3.
        canonical_reactant = canonicalize(candidate_reactant)
        reactant_possibly_unchanged_in_products = canonical_reactant in canon_products_to_orig_prods
        # ^ possibly as it could be different when we include atom maps -- we will check for this later.
        if not_in_center and reactant_possibly_unchanged_in_products:
            # We also need to match this reactant up with the appropriate product SMILES string and remove this from
            # the product.  To do this we shall go through the possible product SMILES strings.
            possible_prod = None
            for prod in canon_products_to_orig_prods[canonical_reactant]:
                # if the atom mapped numbers intersect then this must be the product we are after and can break!
                if (
                    len(
                        set(get_atom_map_nums(prod, not strict_mode))
                        & set(get_atom_map_nums(candidate_reactant, not strict_mode))
                    )
                    > 0
                ):
                    break

                # if the product in the reaction SMILES has no atom map numbers it _could_ match but check other
                # possibilities first to see if we get an atom map match.
                if len(list(get_atom_map_nums(prod, not strict_mode))) == 0:
                    possible_prod = prod
            else:
                prod = (
                    possible_prod  # <-- if we are here then we did not get an exact atom map match
                )

            if prod is not None:
                # ^ if it is still None then a false alarm and not the same molecule due to atom map numbers.
                # (we're going to defer to atom map numbers and assume they're right!)
                reagents.append(candidate_reactant)
                products_to_keep.subtract([prod])  # remove it from the products too

                # we also need to do some book keeping on our datastructure mapping canonical SMILES to product strings
                # to indicate that we have removed one.
                canon_products_to_orig_prods[canonical_reactant].subtract([prod])
                canon_products_to_orig_prods[canonical_reactant] += collections.Counter()
                # ^ remove zero and negative values
                if len(canon_products_to_orig_prods[canonical_reactant]) == 0:
                    del canon_products_to_orig_prods[canonical_reactant]

                continue

        # if passed check 2 and 3 then it is a reactant!
        reactants.append(candidate_reactant)

    product_all_str = ".".join(products_to_keep.elements())
    return ".".join(reactants), ".".join(reagents), product_all_str
