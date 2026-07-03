import copy
import enum
import functools

import numpy as np
import numpy.typing as npt


class SPECIAL_TOKENS(enum.StrEnum):
    """Special tokens used in the tokenizer, usual stuff (eg pad, BOS, etc), special tokens for the reaction network
    (e.g., building block, etc), and special task prompt tokens.
    """

    PAD = "<PAD>"
    BOS = "<BOS>"  # Beginning of sequence
    EOS = "<EOS>"  # End of sequence, used as a stopping token for the decoder between prompt and task solution.

    BB = "<BB>"  # Building block
    FWD_RXN = "<FWD_RXN>"  # Forward reaction
    FWD_RXN_RUN = (
        "<FWD_RXN_RUN>"  # Forward reaction, deliminator between the reactants and products.
    )
    REV_RXN = "<REV_RXN>"  # Reverse reaction, i.e., retrosynthesis.
    REV_RXN_RUN = (
        "<REV_RXN_RUN>"  # Reverse reaction, deliminator between the product and reactants.
    )

    RETRO = "<RETRO>"  # Retrosynthesis task
    OPTIMIZE = "<OPTIMIZE>"  # Optimization task
    PROJECT = "<PROJECT>"  # Project task


PREFIX_TOKENS = frozenset(
    map(lambda x: x.value, [SPECIAL_TOKENS.RETRO, SPECIAL_TOKENS.OPTIMIZE, SPECIAL_TOKENS.PROJECT])
)
RXNNET_TOKENS = frozenset(
    map(lambda x: x.value, [SPECIAL_TOKENS.BB, SPECIAL_TOKENS.FWD_RXN, SPECIAL_TOKENS.REV_RXN])
)
MAIN_ACTION_TOKEN = PREFIX_TOKENS | RXNNET_TOKENS | frozenset([SPECIAL_TOKENS.EOS.value])
REACTION_START_TOKENS = frozenset([SPECIAL_TOKENS.FWD_RXN.value, SPECIAL_TOKENS.REV_RXN.value])
REACTION_DELIM_TOKENS = frozenset(
    [SPECIAL_TOKENS.FWD_RXN_RUN.value, SPECIAL_TOKENS.REV_RXN_RUN.value]
)


class TokenLibrary:
    """Holds a group of tokens along with their integer indices."""

    def __init__(self, tokens: list[str], start_idx: int = 0, molecule_tokens: bool = False):
        """
        Args:
            tokens (list[str]): List of tokens in the library.
            start_idx (int): Index of the first token in the library. This is used to determine the index of a token in
                the library.
            molecule_tokens (bool): If True, the tokens are molecule tokens.
        """

        # Keep both list (for O(1) index->token) and dict (for O(1) token->index)
        self.tokens = tokens
        self._token_to_idx = {token: start_idx + i for i, token in enumerate(tokens)}
        # Check for duplicates
        if len(tokens) != len(self._token_to_idx):
            raise ValueError("Tokens are not unique!")

        self.start_idx = (
            start_idx  # note at moment you should not change this after initialization!
        )
        self.molecule_tokens = molecule_tokens

    def __copy__(self):
        # copy the initial dict
        return TokenLibrary(
            tokens=self.tokens.copy(),
            start_idx=self.start_idx,
            molecule_tokens=self.molecule_tokens,
        )

    def to_dict(self):
        return {
            "tokens": self.tokens,
            "start_idx": self.start_idx,
            "molecule_tokens": self.molecule_tokens,
        }

    @classmethod
    def from_dict(cls, json_dict):
        return cls(
            tokens=json_dict["tokens"],
            start_idx=json_dict["start_idx"],
            molecule_tokens=json_dict["molecule_tokens"],
        )

    def idx_frm_token(self, token: str) -> int:
        if token not in self._token_to_idx:
            raise ValueError(f"Token '{token}' not found in library")
        return self._token_to_idx[token]

    def token_frm_idx(self, idx: int) -> str:
        if idx < self.start_idx or idx > self.end_idx:
            raise IndexError(f"Index {idx} not in library.")
        return self.tokens[idx - self.start_idx]

    def __len__(self):
        return len(self.tokens)

    @property
    def token_set(self):
        return set(self._token_to_idx.keys())

    @property
    def end_idx(self):
        return self.start_idx + len(self) - 1

    def idx_in_library(self, idx: int) -> bool:
        return self.start_idx <= idx <= self.end_idx

    def token_in_library(self, token: str) -> bool:
        return token in self._token_to_idx

    @functools.lru_cache(maxsize=10_000)  # noqa: B019
    def token_indcs(self, frozenset_of_tokens_to_indx) -> npt.NDArray[np.int_]:
        return np.array(sorted([self.idx_frm_token(t) for t in frozenset_of_tokens_to_indx]))

    def tokenidx_intersect(self, other: "TokenLibrary") -> tuple[bool, bool]:
        o1 = self.idx_in_library(other.start_idx) or self.idx_in_library(other.end_idx)
        o2 = self.molecule_tokens == other.molecule_tokens
        return o1, o2

    def add_token_to_end(self, token: str):
        if token in self._token_to_idx:
            raise ValueError(f"Token '{token}' already in library!")
        self._token_to_idx[token] = self.end_idx + 1
        self.tokens.append(token)
        return self


class TokenLibraryCollection:
    """Holds a collection of three token libraries: special tokens, building block tokens, and molecule tokens.

    Building blocks and molecules tokens are assumed to be disjoint. Both represent "molecules", but building blocks are
    those which are readily available.

    (Should be consecutive in the indices used.)
    """

    def __init__(self, token_libraries: list[TokenLibrary]):
        self.token_libraries = token_libraries
        assert len(self.token_libraries) == 3, "TokenLibraryCollection must have 3 token libraries."
        self.special_tkn_library = self.token_libraries[0]
        self.bb_tkn_library = self.token_libraries[1]
        self.mol_tkn_library = self.token_libraries[2]
        assert not any(
            any(self.special_tkn_library.tokenidx_intersect(other_tkn_lib))
            for other_tkn_lib in [self.bb_tkn_library, self.mol_tkn_library]
        ), "Token libraries must not overlap!"

    def __copy__(self):
        """Create a shallow copy of the TokenLibraryCollection.

        Special tokens (index 0) and building blocks (index 1) are always the same across
        instances and never modified, so we can safely share references to them.
        Only the molecule library (index 2) is dynamic and gets modified during inference,
        so we need to copy it to avoid shared state between instances.
        """
        copied_token_libraries = [
            self.token_libraries[0],  # special tokens - shared reference
            self.token_libraries[1],  # building blocks - shared reference
            copy.copy(self.token_libraries[2]),  # molecules - copy needed
        ]

        return TokenLibraryCollection(copied_token_libraries)

    def to_dict(self):
        return {"token_libraries": [t.to_dict() for t in self.token_libraries]}

    @classmethod
    def from_dict(cls, json_dict):
        return cls(
            token_libraries=[TokenLibrary.from_dict(t) for t in json_dict["token_libraries"]]
        )

    @property
    def start_idx(self):
        return min([t.start_idx for t in self.token_libraries])

    @property
    def start_of_molecular_indices(self):
        return self.bb_tkn_library.start_idx

    @property
    def end_idx(self):
        return max([t.end_idx for t in self.token_libraries])

    def __len__(self):
        return self.end_idx - self.start_idx + 1

    @property
    def tokens(self):
        return [t for lib in self.token_libraries for t in lib.tokens]

    def token_in_collection(self, token: str) -> bool:
        return any(t.token_in_library(token) for t in self.token_libraries)

    def token_frm_idx(self, idx: int) -> str:
        for t in self.token_libraries:
            if t.idx_in_library(idx):
                return t.token_frm_idx(idx)
        raise ValueError(f"Index {idx} not in any token library.")

    def idx_frm_token(self, token: str, fail_on_not_found: bool = True) -> int:
        for t in self.token_libraries:
            if t.token_in_library(token):
                return t.idx_frm_token(token)
        if fail_on_not_found:
            raise ValueError(f"Token {token} not in any token library.")
        else:
            return self.add_mol_token_to_end_and_get_idx(token)

    def add_mol_token_to_end_and_get_idx(self, token: str):
        if token in self.bb_tkn_library.token_set:
            raise ValueError(f"Token '{token}' is already a building block!")
        if token in self.mol_tkn_library.token_set:
            raise ValueError(f"Token '{token}' is already a molecule in mol tkn library!")
        self.mol_tkn_library.add_token_to_end(token)
        return self.mol_tkn_library.idx_frm_token(token)


SPECIAL_TOKENS_LIBRARY = TokenLibrary(
    [t.value for t in SPECIAL_TOKENS], start_idx=0, molecule_tokens=False
)
PAD_VALUE = int(SPECIAL_TOKENS_LIBRARY.idx_frm_token(SPECIAL_TOKENS.PAD))
EOS_VALUE = int(SPECIAL_TOKENS_LIBRARY.idx_frm_token(SPECIAL_TOKENS.EOS))
PREFIX_TOKEN_VALUES = {int(SPECIAL_TOKENS_LIBRARY.idx_frm_token(t)) for t in PREFIX_TOKENS}
BB_VALUE = int(SPECIAL_TOKENS_LIBRARY.idx_frm_token(SPECIAL_TOKENS.BB))

FWD_RXN_RUN_VALUE = int(SPECIAL_TOKENS_LIBRARY.idx_frm_token(SPECIAL_TOKENS.FWD_RXN_RUN))
REV_RXN_RUN_VALUE = int(SPECIAL_TOKENS_LIBRARY.idx_frm_token(SPECIAL_TOKENS.REV_RXN_RUN))
