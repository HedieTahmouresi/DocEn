"""
Unit tests for homography, quad validation, quad sampling, and coordinate scaling.
"""

import pytest
import numpy as np
import cv2
from src.geometry.homography import (
    get_target_corners,
    compute_homography,
    invert_homography,
    validate_quad,
    sample_target_quad,
    scale_corners,
    rectify_document
)


def test_validate_quad_valid():
    # Standard clockwise square
    corners = np.array([
        [10.0, 10.0],
        [500.0, 10.0],
        [500.0, 500.0],
        [10.0, 500.0]
    ], dtype=np.float32)
    assert validate_quad(corners, canvas_size=(512, 512))


def test_validate_quad_invalid_order():
    # Counter-clockwise square (bowtie/mis-ordered)
    corners = np.array([
        [10.0, 10.0],
        [10.0, 500.0],
        [500.0, 500.0],
        [500.0, 10.0]
    ], dtype=np.float32)
    assert not validate_quad(corners, canvas_size=(512, 512))


def test_validate_quad_out_of_bounds():
    # Corner outside 512x512 canvas
    corners = np.array([
        [-5.0, 10.0],
        [500.0, 10.0],
        [500.0, 500.0],
        [10.0, 500.0]
    ], dtype=np.float32)
    assert not validate_quad(corners, canvas_size=(512, 512))


def test_sample_target_quad_1000_samples():
    rng = np.random.RandomState(42)
    for _ in range(1000):
        corners, params = sample_target_quad(canvas_size=(512, 512), rng=rng)
        assert corners.shape == (4, 2)
        assert validate_quad(corners, min_angle_deg=20.0, canvas_size=(512, 512))


def test_homography_inversion_roundtrip():
    rng = np.random.RandomState(42)
    src_corners = get_target_corners(512, 512)
    dst_corners, _ = sample_target_quad(canvas_size=(512, 512), rng=rng)

    H = compute_homography(src_corners, dst_corners)
    H_inv = invert_homography(H)

    # Test round-trip mapping of points: H @ (H_inv @ pt) == pt
    pts_hom = np.hstack([src_corners, np.ones((4, 1))]).T  # (3, 4)
    mapped_hom = H @ pts_hom
    mapped_pts = (mapped_hom[:2] / mapped_hom[2]).T

    np.testing.assert_allclose(mapped_pts, dst_corners, atol=1e-3)

    inv_hom = H_inv @ mapped_hom
    inv_pts = (inv_hom[:2] / inv_hom[2]).T

    np.testing.assert_allclose(inv_pts, src_corners, atol=1e-3)


def test_scale_corners_roundtrip():
    corners = np.array([
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 200.0],
        [0.0, 200.0]
    ], dtype=np.float32)

    src_size = (101, 201)
    dst_size = (512, 512)

    scaled = scale_corners(corners, src_size, dst_size)
    roundtripped = scale_corners(scaled, dst_size, src_size)

    np.testing.assert_allclose(roundtripped, corners, atol=1e-5)
