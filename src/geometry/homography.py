"""
Homography, Perspective Rectification, and Quad Sampling Utilities.

Conventions:
- Source & Target corners: (4, 2) float32 in [TL, TR, BR, BL] order (clockwise).
- Image & Canvas space coordinates: (x, y) float32 in pixels.
- Matrix H: (3, 3) float64 transform mapping source points to destination points.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def get_target_corners(width: int, height: int) -> np.ndarray:
    """
    Get standard destination corners for perspective warp:
    0: (0, 0) - TL
    1: (w-1, 0) - TR
    2: (w-1, h-1) - BR
    3: (0, h-1) - BL
    """
    return np.array([
        [0.0, 0.0],
        [float(width - 1), 0.0],
        [float(width - 1), float(height - 1)],
        [0.0, float(height - 1)]
    ], dtype=np.float32)


def compute_homography(src_corners: np.ndarray, dst_corners: np.ndarray) -> np.ndarray:
    """
    Compute 3x3 homography matrix H mapping src_corners to dst_corners.
    Both src_corners and dst_corners must be (4, 2) float32 in TL, TR, BR, BL order.
    """
    src_pts = src_corners.astype(np.float32)
    dst_pts = dst_corners.astype(np.float32)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return H.astype(np.float64)


def invert_homography(H: np.ndarray) -> np.ndarray:
    """
    Invert homography matrix exactly using matrix inversion (REQ-35).
    Do NOT re-derive from corners.
    """
    retval, H_inv = cv2.invert(H.astype(np.float64))
    if not retval:
        raise ValueError("Homography matrix H is singular and cannot be inverted.")
    return H_inv.astype(np.float64)


def validate_quad(
    corners: np.ndarray,
    min_angle_deg: float = 20.0,
    canvas_size: Tuple[int, int] = (512, 512),
    margin: float = 0.0
) -> bool:
    """
    Validate quadrilateral shape, convexity, corner ordering, interior angles, and canvas boundaries.

    Args:
        corners: (4, 2) float array [TL, TR, BR, BL]
        min_angle_deg: minimum interior angle floor in degrees (default 20.0)
        canvas_size: (width, height) of bounding canvas
        margin: minimum distance corners must be inside canvas (default 0.0)

    Returns:
        bool: True if quad is valid, False otherwise.
    """
    if corners.shape != (4, 2):
        return False

    w, h = canvas_size

    # 1. Bounds check (all 4 corners inside canvas)
    if np.any(corners[:, 0] < margin) or np.any(corners[:, 0] > (w - 1.0 - margin)):
        return False
    if np.any(corners[:, 1] < margin) or np.any(corners[:, 1] > (h - 1.0 - margin)):
        return False

    # 2. Convexity & Clockwise ordering check via cross products
    # Edges E_0 (TL->TR), E_1 (TR->BR), E_2 (BR->BL), E_3 (BL->TL)
    edges = np.roll(corners, -1, axis=0) - corners  # E_i = V_{i+1} - V_i
    next_edges = np.roll(edges, -1, axis=0)        # E_{i+1}

    # Cross product in 2D: E_i.x * E_{i+1}.y - E_i.y * E_{i+1}.x
    # In screen coordinates (y-down), clockwise vertices produce POSITIVE cross products
    cross_products = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if np.any(cross_products <= 1e-4):
        return False

    # 3. Interior angles check
    # At vertex V_i, incoming vector is -E_{i-1}, outgoing vector is E_i
    inc_edges = -np.roll(edges, 1, axis=0)  # V_{i-1} -> V_i
    out_edges = edges                       # V_i -> V_{i+1}

    inc_norms = np.linalg.norm(inc_edges, axis=1)
    out_norms = np.linalg.norm(out_edges, axis=1)
    if np.any(inc_norms < 1e-4) or np.any(out_norms < 1e-4):
        return False

    # Cosine of interior angle
    dot_prods = np.sum(inc_edges * out_edges, axis=1)
    cos_angles = np.clip(dot_prods / (inc_norms * out_norms), -1.0, 1.0)
    angles_rad = np.arccos(cos_angles)
    angles_deg = np.degrees(angles_rad)

    if np.any(angles_deg < min_angle_deg):
        return False

    return True


def sample_target_quad(
    canvas_size: Tuple[int, int] = (512, 512),
    area_fraction_range: Tuple[float, float] = (0.15, 0.95),
    rotation_range_deg: Tuple[float, float] = (-25.0, 25.0),
    perspective_strength_range: Tuple[float, float] = (0.0, 0.35),
    aspect_jitter_range: Tuple[float, float] = (-0.15, 0.15),
    min_angle_deg: float = 20.0,
    rng: Optional[np.random.RandomState] = None,
    max_retries: int = 200
) -> Tuple[np.ndarray, dict]:
    """
    Sample a realistic quad by shape-then-place algorithm (synthetic-generator-spec.md §5).

    Returns:
        corners: (4, 2) float32 in [TL, TR, BR, BL] order
        params: dict of sampled parameters
    """
    if rng is None:
        rng = np.random.RandomState()

    w_canvas, h_canvas = canvas_size
    canvas_area = float(w_canvas * h_canvas)

    for _ in range(max_retries):
        area_frac = rng.uniform(*area_fraction_range)
        target_area = canvas_area * area_frac

        aspect_jitter = rng.uniform(*aspect_jitter_range)
        aspect_ratio = 1.0 + aspect_jitter

        w0 = np.sqrt(target_area * aspect_ratio)
        h0 = np.sqrt(target_area / aspect_ratio)

        # Base rectangle centered at (0, 0) in TL, TR, BR, BL order
        base_corners = np.array([
            [-w0 / 2.0, -h0 / 2.0],
            [ w0 / 2.0, -h0 / 2.0],
            [ w0 / 2.0,  h0 / 2.0],
            [-w0 / 2.0,  h0 / 2.0]
        ], dtype=np.float64)

        # Perspective displacement (tilt)
        p_strength = rng.uniform(*perspective_strength_range)
        disp_scale = p_strength * min(w0, h0) * 0.5
        displacements = rng.uniform(-disp_scale, disp_scale, size=(4, 2))
        corners_tilted = base_corners + displacements

        # In-plane rotation
        rot_deg = rng.uniform(*rotation_range_deg)
        rot_rad = np.radians(rot_deg)
        cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        corners_rotated = corners_tilted @ R.T

        # Calculate bounding box of quad
        min_x, min_y = np.min(corners_rotated, axis=0)
        max_x, max_y = np.max(corners_rotated, axis=0)
        quad_w = max_x - min_x
        quad_h = max_y - min_y

        if quad_w >= (w_canvas - 1.0) or quad_h >= (h_canvas - 1.0):
            continue

        # Place quad within canvas with random shift
        slack_x = w_canvas - 1.0 - quad_w
        slack_y = h_canvas - 1.0 - quad_h

        shift_x = -min_x + rng.uniform(0.0, slack_x)
        shift_y = -min_y + rng.uniform(0.0, slack_y)

        placed_corners = corners_rotated + np.array([shift_x, shift_y])

        if validate_quad(placed_corners, min_angle_deg=min_angle_deg, canvas_size=canvas_size):
            params = {
                "area_fraction": area_frac,
                "aspect_jitter": aspect_jitter,
                "perspective_strength": p_strength,
                "rotation_deg": rot_deg,
                "center_x": float(np.mean(placed_corners[:, 0])),
                "center_y": float(np.mean(placed_corners[:, 1])),
            }
            return placed_corners.astype(np.float32), params

    raise RuntimeError(
        f"Failed to sample valid quad after {max_retries} retries with area_range={area_fraction_range}."
    )


def scale_corners(
    corners: np.ndarray,
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int]
) -> np.ndarray:
    """
    Scale corner coordinates from src_size (w_src, h_src) to dst_size (w_dst, h_dst).

    Args:
        corners: (N, 2) float array of (x, y) coordinates
        src_size: (w_src, h_src)
        dst_size: (w_dst, h_dst)

    Returns:
        scaled_corners: (N, 2) float32 array
    """
    w_src, h_src = src_size
    w_dst, h_dst = dst_size

    scale_x = float(w_dst - 1) / float(w_src - 1) if w_src > 1 else 1.0
    scale_y = float(h_dst - 1) / float(h_src - 1) if h_src > 1 else 1.0

    scaled = corners.copy().astype(np.float32)
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled


def rectify_document(
    img_rgb: np.ndarray,
    corners: np.ndarray,
    target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Warp an unrectified document image into a perspective-rectified crop.

    Args:
        img_rgb: (H, W, 3) uint8 RGB image
        corners: (4, 2) float32 corners in [TL, TR, BR, BL] order
        target_size: (target_w, target_h) output dimensions

    Returns:
        rectified_rgb: (target_h, target_w, 3) uint8 RGB image
    """
    target_w, target_h = target_size
    dst_corners = get_target_corners(target_w, target_h)

    # Compute homography
    H_matrix = compute_homography(corners, dst_corners)

    # INTER_CUBIC + BORDER_REPLICATE deliberately match the generator's inverse warp
    # (src/data/generator.py). Rectifying real photos with a different resampler than the
    # one that produced every training pair adds an avoidable sim2real gap: bilinear is
    # visibly softer on 1-3 px text strokes than bicubic.
    rectified_bgr = cv2.warpPerspective(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        H_matrix,
        (target_w, target_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)

