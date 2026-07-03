import collections
import concurrent.futures
import copy
import hashlib
import json
import logging
import pathlib
import sys
from collections import abc as c_abc
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from numpy import random as np_random


class TwoWayDict(c_abc.MutableMapping):
    """Dict for bidirectional one to one mapping of keys and values. Eg between canonical SMILES and molecule"""

    def __init__(self, *args, **kwargs):
        self._forward = {}
        self._backward = {}
        self.update(dict(*args, **kwargs))

    @classmethod
    def from_forward_mapping(cls, forward_mapping):
        twd = cls()
        twd._forward = forward_mapping
        assert len(set(forward_mapping.values())) == len(forward_mapping), (
            "Values in forward mapping must be unique."
        )
        twd._backward = {v: k for k, v in forward_mapping.items()}
        return twd

    def __getitem__(self, key):
        return self._forward[key]

    def __iter__(self):
        return iter(self._forward)

    def __len__(self):
        return len(self._forward)

    def __setitem__(self, key, value):
        if value in self._backward and self._backward[value] != key:
            raise KeyError(f"Value {value} is already mapped to key {self._backward[value]}")
        if key in self._forward:
            old_value = self._forward[key]
            del self._backward[old_value]
        self._forward.__setitem__(key, value)
        self._backward.__setitem__(value, key)

    def __delitem__(self, key):
        value = self._forward[key]
        self._forward.__delitem__(key)
        self._backward.__delitem__(value)

    def __getstate__(self):
        return self._forward

    def __setstate__(self, state):
        self._forward = state
        self._backward = {v: k for k, v in state.items()}

    @property
    def inversed(self):
        nw = TwoWayDict()
        nw._forward = self._backward
        nw._backward = self._forward
        return nw

    def inverse_in(self, value):
        return value in self._backward

    def inverse_get(self, value):
        return self._backward[value]

    @property
    def forward_mapping_only(self):
        return self._forward


class ManyToManyMapping(c_abc.MutableMapping):
    """Stores a many-to-many relationship between keys and values. eg between molecule and reactions it particpates in.

    Both keys and values should be hashable and unique. Stores the mapping in both directions.

    Note if all the mappings from an individual item are removed then so it the item itself.
    """

    def __init__(self, *args, **kwargs):
        self._forward = collections.defaultdict(set)
        self._backward = collections.defaultdict(set)
        self.update(dict(*args, **kwargs))

    @classmethod
    def from_forward_mapping(cls, forward_mapping):
        mtm = cls()
        mtm._forward.update(forward_mapping)
        for k, v_items in forward_mapping.items():
            for item in v_items:
                mtm._backward[item].add(k)
        return mtm

    def __getitem__(self, key):
        if key not in self._forward:
            raise KeyError(f"{key} not found in mapping.")
        return self._forward[key]

    def __iter__(self):
        return iter(self._forward)

    def __len__(self):
        return len(self._forward)

    def __setitem__(self, key, value):
        assert isinstance(value, set)
        if len(value) == 0:
            raise ValueError("Cannot set an empty set as a value -- use del instead.")
        old_values = self._forward[key]
        for item in old_values:
            self._backward[item].discard(key)
            if len(self._backward[item]) == 0:
                del self._backward[item]
        self._forward[key] = value
        for item in value:
            self._backward[item].add(key)

    def __delitem__(self, key):
        old_values = self._forward[key]
        for item in old_values:
            self._backward[item].discard(key)
            if len(self._backward[item]) == 0:
                del self._backward[item]
        del self._forward[key]

    def __contains__(self, key):
        return key in self._forward

    def __getstate__(self):
        return self._forward

    def __setstate__(self, state):
        self._forward = state
        self._backward = collections.defaultdict(set)
        for k, v_items in state.items():
            for item in v_items:
                self._backward[item].add(k)

    def add_single_relationship(self, key, value):
        self._forward[key].add(value)
        self._backward[value].add(key)

    def remove_single_relationship(self, key, value):
        self._forward[key].remove(value)
        if len(self._forward[key]) == 0:
            del self._forward[key]
        self._backward[value].remove(key)
        if len(self._backward[value]) == 0:
            del self._backward[value]

    def get_inverse_relationships(self, value):
        if value not in self._backward:
            raise KeyError(f"No inverse relationships found for {value}")
        return self._backward[value]

    @property
    def inversed(self):
        nw = ManyToManyMapping()
        nw._forward = self._backward
        nw._backward = self._forward
        return nw

    @property
    def forward_mapping_only(self):
        return self._forward


def random_top_idx(x: list | np.ndarray, rng: np_random.RandomState, reverse: bool = False) -> int:
    rng.shuffle(copy.copy(x))
    top_idx = np.argmin(x) if reverse else np.argmax(x)
    return int(top_idx)


def load_smiles_csv(csv_fpath, smiles_col="SMILES"):
    """
    Load a CSV file with a column of SMILES strings and return a list of the SMILES strings. CSV path can be in s3.
    (as use pandas to do the read in).
    """
    df = pd.read_csv(csv_fpath)
    return df[smiles_col].tolist()


def multiprocess_func_with_tqdm(pickleable_func, data, num_processes):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(pickleable_func, d) for d in data]
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="running in ||"
        ):
            out = future.result()
            yield out


def get_date_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_logger(*, internal_log_name=None, log_file_name="run_log.log", capture_warnings=True):
    """
    from: https://github.com/john-bradshaw/rxn-splits/blob/main/rxn_splits/utils.py
    """

    internal_log_name = internal_log_name or __name__
    logger = logging.getLogger(internal_log_name)

    # std out handler
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        f"{internal_log_name} - %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File Handler
    fh = logging.FileHandler(log_file_name)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.setLevel(logging.DEBUG)
    logger.debug(f"Internal log name: {internal_log_name}")
    logger.debug(f"Log file name: {log_file_name}")

    if capture_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        se = logging.StreamHandler(sys.stderr)
        warnings_logger.addHandler(se)
        warnings_logger.addHandler(fh)
        logger.debug("Warnings logger also directed to file handler.")

    return logger


def save_json(data, fpath, indent=4):
    with open(fpath, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(fpath):
    with open(fpath) as f:
        o = json.load(f)
    return o


def compute_file_sha256(file_path: str | pathlib.Path, chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of a file's contents.  Returns Hexadecimal string representation of the SHA256 hash
    """
    hash_obj = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def extract_new_tokens(
    last_pad_pos_before: torch.Tensor,  # [m]
    last_pad_pos_after: torch.Tensor,  # [m]
    seqs_to_index_from: torch.Tensor,  # [m, n]
    pad_value=0,
) -> torch.Tensor:
    """
    Extract elements added between before and after positions. If no elements are to be extracted,
    then raises a RuntimeError.

    For example if the sequences are:
    [[1, 2, 3, 4, 5],
     [6, 7, 8, 9, 10]]

    The first pad positions are: [2,3]
    And the last pad positions are:[3,6]

    Then the extracted elements are:
    [[3, 0, 0],
     [9, 10, 0]]

    Args:
        last_pad_pos_before: Index where extraction starts for each sequence [m]
        last_pad_pos_after: Index where extraction ends for each sequence (exclusive) [m]
        seqs_to_index_from: Sequences to extract from (i.e., each row is a sequence) [m, n]
        pad_value: Value to use for padding invalid positions

    Returns:
        Tensor of shape [m, n'] containing only the new, extracted elements.
        (non used positions are padded with pad_value)
    """
    num_elements_to_extract_per_row = last_pad_pos_after - last_pad_pos_before  # [m]
    n_prime = num_elements_to_extract_per_row.max().item()  # max number of elements to extract

    if n_prime <= 0:
        raise RuntimeError("No elements are to be extracted!")

    m = seqs_to_index_from.shape[0]
    ranges = torch.arange(n_prime, device=seqs_to_index_from.device)[None, :].repeat(
        m, 1
    )  # [m, n_prime]
    start_indices = last_pad_pos_before[:, None]  # [m, 1] (inclusive start)
    sequence_indices = ranges + start_indices  # [m, n_prime]

    # Add pad value to end of sequences to handle if action index goes past the end
    padded_sequences = torch.cat(
        [
            seqs_to_index_from,
            torch.full(
                (m, 1), pad_value, device=seqs_to_index_from.device, dtype=seqs_to_index_from.dtype
            ),
        ],
        dim=1,
    )
    sequence_indices = torch.clamp(sequence_indices, 0, padded_sequences.shape[1] - 1)

    extracted = torch.gather(padded_sequences, 1, sequence_indices)  # [m, n_prime]

    # Apply masking to set positions beyond actual extracted length to pad_value
    valid_mask = ranges < num_elements_to_extract_per_row[:, None]  # [m, n_prime]
    extracted = torch.where(valid_mask, extracted, pad_value)

    return extracted


def masked_cross_entropy_efficient(logits, targets, mask):
    """
    Efficient cross-entropy that skips padded tokens entirely.

    Args:
        logits: (B, T, C)
        targets: (B, T)
        mask: (B, T) — 1 for valid tokens, 0 for padding, or bool tensor

    Returns:
        loss: (B, T) — per-token loss, 0.0 where masked
    """
    B, T, C = logits.shape  # also checks shape size is as expected
    assert targets.shape == (B, T), "Targets must have same shape for first two dims as logits."
    assert mask.shape == (B, T), "Mask must have same shape as targets."

    logits_flat = logits.view(-1, C)  # (B*T, C)
    targets_flat = targets.view(-1)  # (B*T)
    mask_flat = mask.view(-1).bool()  # (B*T)

    # Only compute on valid positions
    valid_logits = logits_flat[mask_flat]  # (N, C)
    valid_targets = targets_flat[mask_flat]  # (N)

    valid_loss = F.cross_entropy(valid_logits, target=valid_targets, reduction="none")  # (N,)

    # Build full-size loss tensor
    full_loss = torch.zeros(B * T, device=logits.device, dtype=logits.dtype)
    full_loss[mask_flat] = valid_loss
    return full_loss.view(B, T)


def pad_then_stack(
    list_of_tensors: list[torch.Tensor],
    pad_value,
    pad_dim: int = -1,
    padded_final_length: int | None = None,
    _RUN_CHECKS: bool = True,
):
    """
    Pad a specified dimension of tensors in a list to a common length and stack them.

    Args:
        list_of_tensors: List of tensors to pad. All tensors must have the same number of dimensions.
        pad_value: Value to use for padding
        pad_dim: Dimension to pad (default: -1 for last dimension). Can be negative.
        padded_final_length: Target length for the specified dimension. If None, uses max length in the list.
        _RUN_CHECKS: Whether to run validation checks (default: True)

    Returns:
        Stacked tensor with shape [len(list_of_tensors), ..., padded_final_length, ...]
        where padded_final_length is in the position specified by pad_dim.

    Raises:
        ValueError: If padded_final_length is 0 or if tensors have different numbers of dimensions
    """
    if not list_of_tensors:
        raise ValueError("Cannot pad empty list of tensors")

    max_pad_dim_length = max(o.shape[pad_dim] for o in list_of_tensors)
    padded_final_length = padded_final_length or max_pad_dim_length
    if padded_final_length == 0:
        raise ValueError("Padded final length cannot be 0")
    if padded_final_length < max_pad_dim_length:
        raise ValueError(
            "Padded final length cannot be less than the maximum length of the dimension being padded"
        )

    num_dims = list_of_tensors[0].ndim

    # Verify all tensors have the same number of dimensions
    if _RUN_CHECKS and not all(t.ndim == num_dims for t in list_of_tensors):
        raise ValueError("All tensors must have the same number of dimensions")

    # Create padding specification
    # F.pad expects pairs (pad_left, pad_right) for each dimension, starting from the last dimension
    # Convert pad_dim to positive index and calculate how many dimensions from the end
    pad_dim_positive = pad_dim if pad_dim >= 0 else (num_dims + pad_dim)

    if _RUN_CHECKS:
        all_shapes_before_pad_dim = set(t.shape[:pad_dim_positive] for t in list_of_tensors)
        all_shapes_after_pad_dim = set(t.shape[pad_dim_positive + 1 :] for t in list_of_tensors)
        if len(all_shapes_before_pad_dim) > 1 or len(all_shapes_after_pad_dim) > 1:
            raise ValueError(
                "All tensors must have the same shape in the dimensions that are not being padded"
            )

    pad_dim_backwards = num_dims - pad_dim_positive
    first_pad_vals = [0] * (2 * pad_dim_backwards - 1)

    padded_tensors = [
        F.pad(o, pad=first_pad_vals + [padded_final_length - o.shape[pad_dim]], value=pad_value)
        for o in list_of_tensors
    ]
    stacked_padded_tensors = torch.stack(padded_tensors)
    return stacked_padded_tensors
