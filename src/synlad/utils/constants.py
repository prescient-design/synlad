"""Constants and utilities for the repo"""

ATOM_TYPES = [
    "C",
    "N",
    "O",
    "F",
    "S",
    "Cl",
    "Br",
    "I",
    "*",  # dummy/unknown atom
]

# Mapping from atomic numbers to atom type indices
ATOMIC_NUMBER_TO_INDEX = {
    6: 0,  # Carbon
    7: 1,  # Nitrogen
    8: 2,  # Oxygen
    9: 3,  # Fluorine
    16: 4,  # Sulfur
    17: 5,  # Chlorine
    35: 6,  # Bromine
    53: 7,  # Iodine
    # All other elements map to index 8 (wildcard "*")
}

PH4_TYPES_TO_INDEX = {
    21: 0,
    22: 1,
    23: 2,
    24: 3,
    25: 4,
    26: 5,
}

# Reverse mapping from indices to element symbols
INDEX_TO_ELEMENT_SYMBOL = {i: symbol for i, symbol in enumerate(ATOM_TYPES)}


def atomic_number_to_atom_type_index(atomic_number: int) -> int:
    """
    Map atomic numbers to atom type indices for supported elements.

    Args:
        atomic_number: Standard atomic number (e.g., 6 for Carbon)

    Returns:
        Atom type index (0-8) for the limited vocabulary
    """
    return ATOMIC_NUMBER_TO_INDEX.get(atomic_number, 8)  # 8 is wildcard for unknown


def ph4_type_to_index(ph4_type: int) -> int:
    """
    Map pharmacophore types to indices.
    """
    return PH4_TYPES_TO_INDEX.get(ph4_type, 6)  # 6 is wildcard for unknown


def atom_type_index_to_element_symbol(atom_type_index: int) -> str:
    """
    Convert atom type index back to element symbol.

    Args:
        atom_type_index: Index in the ATOM_TYPES list (0-8)

    Returns:
        Element symbol string (e.g., "C", "N", "O", etc.)
    """
    if 0 <= atom_type_index < len(ATOM_TYPES):
        return ATOM_TYPES[atom_type_index]
    else:
        return "*"  # Unknown/invalid index
