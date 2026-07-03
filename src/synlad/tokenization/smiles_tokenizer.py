"""Regex-based SMILES tokenizer (Schwaller et al. 2019)."""

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import torch

# Schwaller-style atom-level SMILES regex. Matches bracketed atoms first, then
# two-letter element symbols, single-letter atoms, bonds/branches/ring-digits.
SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|Br|Cl|Si|Se|Li|Na|Mg|Al|Ca|Fe|Zn"
    r"|[BCNOSPFI]|[bcnosp]"
    r"|=|#|-|\+|\\|\/|\(|\)|\.|:|@|%\d{2}|\d)"
)


PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def tokenize_smiles(smiles: str) -> list[str]:
    return SMILES_REGEX.findall(smiles)


class SmilesTokenizer:
    """Maps SMILES strings to integer token ids and back.

    Token layout: special tokens first (pad=0, sos=1, eos=2, unk=3), then the
    learned vocabulary. `<pad>` is index 0 so `nn.Embedding(..., padding_idx=0)`
    works without extra remapping.
    """

    def __init__(self, vocab: list[str]):
        # vocab must begin with the special tokens in canonical order
        assert vocab[: len(SPECIAL_TOKENS)] == SPECIAL_TOKENS, (
            "vocab must begin with special tokens: " + str(SPECIAL_TOKENS)
        )
        self.vocab: list[str] = list(vocab)
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.sos_id = self.token_to_id[SOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(
        self,
        smiles: str,
        max_len: int | None = None,
        add_special: bool = True,
    ) -> torch.LongTensor:
        """Encode a single SMILES string into a 1-D LongTensor of token ids."""
        toks = tokenize_smiles(smiles)
        ids = [self.token_to_id.get(t, self.unk_id) for t in toks]
        if add_special:
            ids = [self.sos_id] + ids + [self.eos_id]
        if max_len is not None:
            # Truncate; caller is responsible for filtering overlong inputs upstream
            ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Iterable[int], strip_special: bool = True) -> str:
        special_ids = {self.pad_id, self.sos_id, self.eos_id}
        pieces = []
        for i in ids:
            i = int(i)
            if strip_special and i in special_ids:
                if i == self.eos_id:
                    break
                continue
            if i < 0 or i >= self.vocab_size:
                continue
            tok = self.vocab[i]
            if strip_special and tok == UNK_TOKEN:
                # keep an explicit marker so invalid SMILES show up downstream
                pieces.append("?")
            else:
                pieces.append(tok)
        return "".join(pieces)

    def to_dict(self) -> dict:
        return {"vocab": list(self.vocab)}

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_json(cls, path: str | Path) -> "SmilesTokenizer":
        with open(path) as f:
            d = json.load(f)
        return cls(d["vocab"])

    @classmethod
    def build_from_smiles(
        cls,
        smiles_list: Iterable[str],
        min_count: int = 1,
    ) -> "SmilesTokenizer":
        counter: Counter = Counter()
        for smi in smiles_list:
            if not smi:
                continue
            counter.update(tokenize_smiles(smi))
        learned = sorted([tok for tok, c in counter.items() if c >= min_count])
        vocab = list(SPECIAL_TOKENS) + learned
        return cls(vocab)

    def count_tokens(self, smiles: str, add_special: bool = True) -> int:
        n = len(tokenize_smiles(smiles))
        return n + (2 if add_special else 0)
