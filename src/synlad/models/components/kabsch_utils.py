"""Adapted from https://github.com/facebookresearch/all-atom-diffusion-transformer/blob/main/src/models/components/kabsch_utils.py

Copyright (c) Meta Platforms, Inc. and affiliates.
This code is released under the CC-BY-NC License -- see https://github.com/facebookresearch/all-atom-diffusion-transformer for further details.
"""

import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ["differentiable_kabsch", "rototranslate", "random_rotation_matrix"]


def differentiable_kabsch(
    p1: torch.Tensor, p2: torch.Tensor, max_iter: int = 10, verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies the differentiable Kabsch algorithm to align two sets of points in 3D space.

    Args:
        p1 (torch.Tensor): A tensor of shape (N, 3) representing the first set of points.
        p2 (torch.Tensor): A tensor of shape (N, 3) representing the second set of points.
        max_iter (int): Maximum number of iterations to perform if SVD is numerically unstable.
        verbose (bool): Whether to print additional information.

    Returns:
        A tuple containing two tensors:
        - rot_mat (torch.Tensor): A tensor of shape (3, 3) representing the rotation matrix that aligns p2 with p1.
        - trans_vec (torch.Tensor): A tensor of shape (3,) representing the translation vector that aligns p2 with p1.
    """
    num_points, num_coords = p1.shape
    device = p1.device
    assert p1.shape == p2.shape

    # Compute centers of mass
    p1_com = p1.mean(dim=0)  # (num_coords)
    p2_com = p2.mean(dim=0)  # (num_coords)

    # Compute (spatial) covariance matrix
    cov = torch.einsum("ni,nj->ij", p1 - p1_com, p2 - p2_com)  # (num_coords, num_coords)
    assert not torch.isnan(cov).any()

    # Compute SVD
    U, S, Vt = torch.linalg.svd(
        cov, full_matrices=True
    )  # (num_coords, num_coords), (num_coords), (num_coords, num_coords)

    # Ensure numerical stability:
    #  Sometimes SVD is poorly conditioned and has zero or degenerate singular values for a square covariance matrix.
    #  This leads to unstable gradients in the backward pass. To avoid this we add a small amount of noise to the
    #  covariance matrix and recompute the SVD until all singular values are non-zero and non-degenerate.
    def _has_zero_singular_value(S):
        return S.min() < 1e-3

    # Generate all (sigma_i)^2 - (sigma_j)^2 and check if any are close to zero
    def _has_degenerate_singular_values(S):
        return (
            S**2 - (S**2).view(num_coords, 1) + torch.eye(num_coords, device=device)
        ).abs().min() < 1e-2

    num_it = 0
    while _has_zero_singular_value(S) or _has_degenerate_singular_values(S):
        # Add a small amount of noise to the covariance matrix and recompute SVD
        noise_scale = (
            cov.abs().max() * 5e-2
        )  # set noise scale to be 5% of the largest value in the covariance matrix
        cov = cov + (torch.rand(3, device=device) * noise_scale).diag()
        U, S, Vt = torch.linalg.svd(
            cov, full_matrices=True
        )  # (num_coords, num_coords), (num_coords), (num_coords, num_coords)
        num_it += 1

        if num_it > max_iter:
            raise ValueError("SVD is consistently numerically unstable.")
    if num_it > 0 and verbose:
        logger.warning(f"Kabsch: SVD was numerically unstable for {num_it} iterations.")

    # Compute rotation matrix and translation vector
    flip_mat = torch.tensor(
        [1, 1, cov.det().sign()], device=device
    ).diag()  # (num_coords, num_coords)
    rot_mat = U @ flip_mat @ Vt  # (num_coords, num_coords)
    trans_vec = p1_com - (rot_mat @ p2_com)  # (num_coords)

    return rot_mat, trans_vec


def rototranslate(
    p: torch.Tensor,
    rot_mat: torch.Tensor,
    trans_vec: torch.Tensor,
    inverse: bool = False,
    validate: bool = False,
) -> torch.Tensor:
    """Apply rototranslation to a set of 3D points. p' = R @ p + t (First rotate, then translate.)

    Args:
        p (torch.Tensor): A tensor of shape (N, 3) representing the set of points.
        rot_mat (torch.Tensor): A tensor of shape (3, 3) representing the rotation matrix.
        trans_vec (torch.Tensor): A tensor of shape (3,) representing the translation vector.
        validate (bool): Whether to validate the input.

    Returns:
        A tensor of shape (N, 3) representing the set of points after rototranslation.
    """
    if validate:
        device = p.device
        _, num_coords = p.shape
        assert rot_mat.shape == (num_coords, num_coords)
        assert trans_vec.shape == (num_coords,)
        assert torch.allclose(
            rot_mat @ rot_mat.T, torch.eye(num_coords, device=device), atol=1e-3, rtol=1e-3
        )

    if inverse:
        return (p - trans_vec) @ rot_mat

    return p @ rot_mat.T + trans_vec


def random_rotation_matrix(validate: bool = False, **tensor_kwargs) -> torch.Tensor:
    """Generates a random (3,3) rotation matrix.

    Args:
        tensor_kwargs: Keyword arguments to pass to the tensor constructor. E.g. `device`, `dtype`.

    Returns:
        A tensor of shape (3, 3) representing the rotation matrix.
    """

    # Generate a random quaternion
    q = torch.rand(4, **tensor_kwargs)
    q /= torch.linalg.norm(q)

    # Compute the rotation matrix from the quaternion
    rot_mat = torch.tensor(
        [
            [
                1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
                2 * q[1] * q[2] - 2 * q[0] * q[3],
                2 * q[1] * q[3] + 2 * q[0] * q[2],
            ],
            [
                2 * q[1] * q[2] + 2 * q[0] * q[3],
                1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
                2 * q[2] * q[3] - 2 * q[0] * q[1],
            ],
            [
                2 * q[1] * q[3] - 2 * q[0] * q[2],
                2 * q[2] * q[3] + 2 * q[0] * q[1],
                1 - 2 * q[1] ** 2 - 2 * q[2] ** 2,
            ],
        ],
        **tensor_kwargs,
    )

    if validate:
        assert torch.allclose(
            rot_mat @ rot_mat.T, torch.eye(3, device=rot_mat.device), atol=1e-5, rtol=1e-5
        ), "Not a rotation matrix."

    return rot_mat
