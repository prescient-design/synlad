"""Tests for the synthesis vocab primitives.

Covers the public surface of :class:`TokenLibrary` plus the round-trip
serialisation contract used to persist vocabs alongside model checkpoints.
"""

from __future__ import annotations

import pytest

from synlad.tokenization.synthesis_vocab import (
    SPECIAL_TOKENS,
    SPECIAL_TOKENS_LIBRARY,
    TokenLibrary,
)


def test_special_tokens_library_round_trip() -> None:
    restored = TokenLibrary.from_dict(SPECIAL_TOKENS_LIBRARY.to_dict())
    assert restored.tokens == SPECIAL_TOKENS_LIBRARY.tokens
    assert restored.start_idx == SPECIAL_TOKENS_LIBRARY.start_idx
    assert restored.molecule_tokens == SPECIAL_TOKENS_LIBRARY.molecule_tokens
    assert len(restored) == len(SPECIAL_TOKENS_LIBRARY)
    for token in SPECIAL_TOKENS_LIBRARY.tokens:
        assert restored.idx_frm_token(token) == SPECIAL_TOKENS_LIBRARY.idx_frm_token(token)


def test_special_tokens_library_contains_required_tokens() -> None:
    for token in (SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.BOS, SPECIAL_TOKENS.EOS):
        assert token.value in SPECIAL_TOKENS_LIBRARY.token_set


def test_token_library_indexing_and_bounds() -> None:
    lib = TokenLibrary(tokens=["a", "b", "c"], start_idx=10)

    assert len(lib) == 3
    assert lib.start_idx == 10
    assert lib.end_idx == 12

    assert lib.idx_frm_token("a") == 10
    assert lib.idx_frm_token("c") == 12
    assert lib.token_frm_idx(11) == "b"

    with pytest.raises(ValueError):
        lib.idx_frm_token("missing")

    with pytest.raises(IndexError):
        lib.token_frm_idx(9)
    with pytest.raises(IndexError):
        lib.token_frm_idx(13)


def test_token_library_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        TokenLibrary(tokens=["a", "a"])
