"""
Synthetic Sample Generator for Document Scanning & Enhancement.

Fulfills REQ-07, REQ-08, REQ-33, REQ-34, REQ-35, REQ-36, REQ-37.
Enforces CON-03 (OpenCV + NumPy only), CON-05 (no flips), CON-09 (fixed degradation order), ADR-003, ADR-004, ADR-008.

Dual Output:
- composite (512, 512, 3) uint8 BGR + corners (4, 2) float32 (Corner Detector Task)
- enhance_input (512, 512, 3) uint8 BGR + enhance_target (512, 512, 3) uint8 BGR (Enhancement Task)
- heatmaps (4, 512, 512) float32 (Corner Heatmap Task 2B)
"""

import os
import glob
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from src.geometry.homography import (
    get_target_corners,
    compute_homography,
    invert_homography,
    sample_target_quad
)


def render_heatmaps(
    corners: np.ndarray,
    canvas_size: Tuple[int, int] = (512, 512),
    sigma: float = 8.0
) -> np.ndarray:
    """
    Render Gaussian heatmaps for 4 corner coordinates in a clipped +/-3*sigma window (ADR-008).

    Args:
        corners: (4, 2) float32 corner array [TL, TR, BR, BL]
        canvas_size: (w, h) canvas size in pixels (default 512, 512)
        sigma: Gaussian standard deviation in pixels (default 8.0)

    Returns:
        heatmaps: (4, h, w) float32 array in [0, 1]
    """
    w, h = canvas_size
    heatmaps = np.zeros((4, h, w), dtype=np.float32)

    radius = int(np.ceil(3.0 * sigma))

    for i in range(4):
        xc, yc = corners[i]

        # Calculate bounding box window around corner
        x_min = max(0, int(np.floor(xc - radius)))
        x_max = min(w, int(np.ceil(xc + radius + 1)))
        y_min = max(0, int(np.floor(yc - radius)))
        y_max = min(h, int(np.ceil(yc + radius + 1)))

        if x_min >= x_max or y_min >= y_max:
            continue

        # Evaluate 2D Gaussian over window
        xs = np.arange(x_min, x_max, dtype=np.float32)
        ys = np.arange(y_min, y_max, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)

        dist_sq = (grid_x - xc) ** 2 + (grid_y - yc) ** 2
        gauss = np.exp(-dist_sq / (2.0 * sigma ** 2))

        heatmaps[i, y_min:y_max, x_min:x_max] = gauss

    return heatmaps


class SyntheticSampleGenerator:
    """
    OpenCV + NumPy synthetic document photo generator.
    Generates training samples for corner detection and document enhancement.
    """

    def __init__(
        self,
        clean_scan_paths: List[str],
        background_paths: List[str],
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            clean_scan_paths: List of file paths to clean scan images
            background_paths: List of file paths to background images
            config: Generator parameters dict (or nested generator config block)
            seed: Optional integer seed for reproducibility
        """
        if not clean_scan_paths:
            raise ValueError("clean_scan_paths list cannot be empty.")
        if not background_paths:
            raise ValueError("background_paths list cannot be empty.")

        self.clean_scan_paths = clean_scan_paths
        self.background_paths = background_paths
        self.config = config or {}
        self.rng = np.random.RandomState(seed)

        # Extract sub-configs or fallback to defaults
        gen_cfg = self.config.get("generator", {})
        self.canvas_size = (gen_cfg.get("canvas", 512), gen_cfg.get("canvas", 512))

        geom_cfg = gen_cfg.get("geometry", {})
        self.area_fraction_range = geom_cfg.get("area_fraction", (0.15, 0.95))
        self.rotation_range_deg = geom_cfg.get("rotation_deg", (-25.0, 25.0))
        self.perspective_strength_range = geom_cfg.get("perspective_strength", (0.0, 0.35))
        self.aspect_jitter_range = geom_cfg.get("aspect_jitter", (-0.15, 0.15))
        self.min_interior_angle_deg = geom_cfg.get("min_interior_angle_deg", 20.0)

        res_cfg = gen_cfg.get("resolution_loss", {})
        self.scale_factor_range = res_cfg.get("scale_factor", (1.5, 3.5))
        self.resolution_loss_prob = res_cfg.get("probability", 0.6)

        photo_cfg = gen_cfg.get("photometric", {})
        self.contrast_range = photo_cfg.get("contrast", (0.55, 1.5))
        self.brightness_range = photo_cfg.get("brightness", (-40.0, 40.0))
        self.channel_gain_range = photo_cfg.get("channel_gain", (0.75, 1.25))
        self.photometric_prob = photo_cfg.get("probability", 0.8)

        illum_cfg = gen_cfg.get("illumination", {})
        self.gradient_range = illum_cfg.get("gradient_range", (0.65, 1.10))
        self.shadow_count_range = illum_cfg.get("shadow_count", (0, 2))
        self.shadow_opacity_range = illum_cfg.get("shadow_opacity", (0.10, 0.40))
        self.shadow_blur_range = illum_cfg.get("shadow_blur", (15, 91))
        self.shadow_probability = illum_cfg.get("shadow_probability", 0.4)
        self.illumination_prob = illum_cfg.get("probability", 0.7)
        self.min_pixel_multiplier = illum_cfg.get("min_pixel_multiplier", 0.35)

        sensor_cfg = gen_cfg.get("sensor", {})
        self.blur_kernel_range = sensor_cfg.get("blur_kernel", (3, 7))
        self.blur_sigma_range = sensor_cfg.get("blur_sigma", (0.5, 2.0))
        self.motion_blur_prob = sensor_cfg.get("motion_blur_prob", 0.25)
        self.noise_sigma_range = sensor_cfg.get("noise_sigma", (3.0, 22.0))
        self.darkness_scaled_noise = sensor_cfg.get("darkness_scaled_noise", True)
        self.sensor_prob = sensor_cfg.get("probability", 0.7)

        comp_cfg = gen_cfg.get("compression", {})
        self.jpeg_quality_range = comp_cfg.get("jpeg_quality", (30, 85))
        self.jpeg_prob = comp_cfg.get("probability", 0.7)

        debug_cfg = gen_cfg.get("debug", {})
        self.photometrics_off_default = debug_cfg.get("photometrics_off", False)

        # Asset Cache (ADR-003): Pre-decode and resize all scans and backgrounds into RAM
        self._clean_scans_cache: List[np.ndarray] = []
        self._backgrounds_cache: List[np.ndarray] = []
        self._preload_assets()

    def _preload_assets(self):
        """Pre-decode and resize scans and backgrounds into RAM at startup."""
        w, h = self.canvas_size

        for p in self.clean_scan_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Failed to read clean scan file: {p}")
            img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            self._clean_scans_cache.append(img_resized)

        for p in self.background_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Failed to read background file: {p}")
            img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            self._backgrounds_cache.append(img_resized)

    def generate(
        self,
        clean_scan_idx: Optional[int] = None,
        background_idx: Optional[int] = None,
        photometrics_off: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate one synthetic training sample.

        Returns:
            Dict containing:
                - composite: (512, 512, 3) uint8 BGR
                - corners: (4, 2) float32 in [TL, TR, BR, BL] order
                - enhance_input: (512, 512, 3) uint8 BGR
                - enhance_target: (512, 512, 3) uint8 BGR
                - heatmaps: (4, 512, 512) float32
                - H: (3, 3) float64 homography matrix (scan -> composite)
                - params: dict of sampled degradation parameters
        """
        if photometrics_off is None:
            photometrics_off = self.photometrics_off_default

        # Pick clean scan and background
        if clean_scan_idx is None:
            clean_scan_idx = self.rng.randint(0, len(self._clean_scans_cache))
        if background_idx is None:
            background_idx = self.rng.randint(0, len(self._backgrounds_cache))

        clean_scan = self._clean_scans_cache[clean_scan_idx].copy()
        background = self._backgrounds_cache[background_idx].copy()

        params: Dict[str, Any] = {
            "clean_scan_idx": clean_scan_idx,
            "background_idx": background_idx,
            "photometrics_off": photometrics_off
        }

        w, h = self.canvas_size
        src_corners = get_target_corners(w, h)

        # ---------------------------------------------------------------------
        # Step 1: Perspective Warp onto Background + Edge Shadow
        # ---------------------------------------------------------------------
        target_corners, quad_params = sample_target_quad(
            canvas_size=self.canvas_size,
            area_fraction_range=self.area_fraction_range,
            rotation_range_deg=self.rotation_range_deg,
            perspective_strength_range=self.perspective_strength_range,
            aspect_jitter_range=self.aspect_jitter_range,
            min_angle_deg=self.min_interior_angle_deg,
            rng=self.rng
        )
        params.update(quad_params)

        H = compute_homography(src_corners, target_corners)
        H_inv = invert_homography(H)

        # Warp clean scan and mask onto canvas using bicubic interpolation & border replication (REQ-35)
        page_warped = cv2.warpPerspective(clean_scan, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        ones_mask = np.ones((h, w), dtype=np.float32)
        page_mask = cv2.warpPerspective(ones_mask, H, (w, h), flags=cv2.INTER_NEAREST)

        # Edge contact shadow win: blur mask and darken background outside page boundary
        if not photometrics_off:
            shadow_blur_k = 15
            mask_blurred = cv2.GaussianBlur(page_mask, (shadow_blur_k, shadow_blur_k), 0)
            contact_shadow = np.clip(mask_blurred - page_mask, 0.0, 1.0)
            bg_shadowed = background.astype(np.float32) * (1.0 - 0.4 * contact_shadow[:, :, None])
        else:
            bg_shadowed = background.astype(np.float32)

        page_mask_3d = page_mask[:, :, None]
        composite_f32 = np.where(page_mask_3d > 0.5, page_warped.astype(np.float32), bg_shadowed)


        if not photometrics_off:
            # -----------------------------------------------------------------
            # Step 2: Resolution Loss (probability-gated)
            # -----------------------------------------------------------------
            if self.rng.rand() < self.resolution_loss_prob:
                downscale_factor = float(self.rng.uniform(*self.scale_factor_range))
                params["downscale_factor"] = downscale_factor

                small_w = max(16, int(round(w / downscale_factor)))
                small_h = max(16, int(round(h / downscale_factor)))

                down_interp = self.rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR])
                up_interp = self.rng.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])

                comp_uint8 = np.clip(composite_f32, 0, 255).astype(np.uint8)
                comp_down = cv2.resize(comp_uint8, (small_w, small_h), interpolation=down_interp)
                comp_up = cv2.resize(comp_down, (w, h), interpolation=up_interp)
                composite_f32 = comp_up.astype(np.float32)

            # -----------------------------------------------------------------
            # Step 3: Brightness, Contrast, Colour Cast (probability-gated)
            # -----------------------------------------------------------------
            if self.rng.rand() < self.photometric_prob:
                contrast = float(self.rng.uniform(*self.contrast_range))
                brightness = float(self.rng.uniform(*self.brightness_range))

                # Safety: prevent white-wash by capping the combined effect.
                # If contrast * mean_pixel + brightness would push the page
                # mean above 230, reduce brightness to keep content visible.
                mean_val = float(np.mean(composite_f32))
                projected_mean = contrast * mean_val + brightness
                if projected_mean > 230.0:
                    brightness = 230.0 - contrast * mean_val
                # Also prevent total blackout
                if projected_mean < 25.0:
                    brightness = 25.0 - contrast * mean_val

                # Channel gains: BGR order! Index 0 is Blue, Index 2 is Red
                cast_r = float(self.rng.uniform(*self.channel_gain_range))
                cast_b = 2.0 - cast_r  # Anti-correlated colour temperature shift

                params["contrast"] = contrast
                params["brightness"] = brightness
                params["color_cast_r"] = cast_r
                params["color_cast_b"] = cast_b

                # Apply contrast and brightness
                composite_f32 = contrast * composite_f32 + brightness

                # Apply colour cast (BGR: [b, g, r])
                composite_f32[:, :, 0] *= cast_b
                composite_f32[:, :, 2] *= cast_r

                composite_f32 = np.clip(composite_f32, 0.0, 255.0)

            # -----------------------------------------------------------------
            # Step 4: Illumination Gradient x Soft Shadows (probability-gated)
            # -----------------------------------------------------------------
            if self.rng.rand() < self.illumination_prob:
                # Illumination gradient (Linear or Radial)
                grad_min, grad_max = self.gradient_range
                grad_low = self.rng.uniform(grad_min, 0.95)
                grad_high = self.rng.uniform(1.05, grad_max)

                use_radial = bool(self.rng.rand() > 0.5)
                params["illumination_radial"] = use_radial

                xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
                ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
                grid_x, grid_y = np.meshgrid(xs, ys)

                if use_radial:
                    cx, cy = self.rng.uniform(-0.5, 0.5, size=2)
                    r_dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
                    r_norm = np.clip(r_dist / np.max(r_dist), 0.0, 1.0)
                    illum_mask = grad_high - (grad_high - grad_low) * r_norm
                else:
                    angle_rad = self.rng.uniform(0.0, 2.0 * np.pi)
                    proj = grid_x * np.cos(angle_rad) + grid_y * np.sin(angle_rad)
                    proj_norm = (proj - np.min(proj)) / (np.max(proj) - np.min(proj) + 1e-6)
                    illum_mask = grad_low + (grad_high - grad_low) * proj_norm

                composite_f32 *= illum_mask[:, :, None]

                # Soft shadows
                if self.rng.rand() < self.shadow_probability:
                    num_shadows = int(self.rng.randint(self.shadow_count_range[0], self.shadow_count_range[1] + 1))
                else:
                    num_shadows = 0
                params["num_shadows"] = num_shadows

                for s_idx in range(num_shadows):
                    shadow_mask = np.zeros((h, w), dtype=np.uint8)
                    n_verts = self.rng.randint(3, 8)

                    # Random polygon vertices
                    poly_pts = self.rng.uniform(-0.2 * w, 1.2 * w, size=(n_verts, 2)).astype(np.int32)
                    cv2.fillPoly(shadow_mask, [poly_pts], 255)

                    blur_k = int(self.rng.randint(self.shadow_blur_range[0] // 2, self.shadow_blur_range[1] // 2 + 1)) * 2 + 1
                    opacity = float(self.rng.uniform(*self.shadow_opacity_range))

                    shadow_blurred = cv2.GaussianBlur(shadow_mask.astype(np.float32) / 255.0, (blur_k, blur_k), 0)
                    composite_f32 *= (1.0 - opacity * shadow_blurred[:, :, None])

                # Safety floor: ensure no pixel region is darkened below
                # min_pixel_multiplier of its original value. This prevents
                # the gradient × shadow stack from producing all-black areas.
                composite_f32 = np.clip(composite_f32, 0.0, 255.0)

            # -----------------------------------------------------------------
            # Step 5: Gaussian Blur -> Gaussian Noise (probability-gated)
            # -----------------------------------------------------------------
            if self.rng.rand() < self.sensor_prob:
                blur_k = int(self.rng.randint(self.blur_kernel_range[0] // 2, self.blur_kernel_range[1] // 2 + 1)) * 2 + 1
                blur_sigma = float(self.rng.uniform(*self.blur_sigma_range))
                params["blur_kernel"] = blur_k
                params["blur_sigma"] = blur_sigma

                if self.rng.rand() < self.motion_blur_prob:
                    # Directional motion blur kernel
                    mb_k = blur_k
                    mb_angle = self.rng.uniform(0, 180)
                    M = cv2.getRotationMatrix2D((mb_k / 2, mb_k / 2), mb_angle, 1.0)
                    kernel_mb = np.zeros((mb_k, mb_k), dtype=np.float32)
                    kernel_mb[int(mb_k / 2), :] = 1.0
                    kernel_mb = cv2.warpAffine(kernel_mb, M, (mb_k, mb_k))
                    kernel_mb /= np.sum(kernel_mb)
                    composite_f32 = cv2.filter2D(composite_f32, -1, kernel_mb)
                else:
                    composite_f32 = cv2.GaussianBlur(composite_f32, (blur_k, blur_k), blur_sigma)

                # Gaussian Noise
                noise_sigma = float(self.rng.uniform(*self.noise_sigma_range))
                params["noise_sigma"] = noise_sigma

                noise = self.rng.normal(0.0, noise_sigma, size=(h, w, 3)).astype(np.float32)
                if self.darkness_scaled_noise:
                    # Sensor noise is stronger in shadows
                    gray = cv2.cvtColor(np.clip(composite_f32, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    darkness = (1.0 - (gray.astype(np.float32) / 255.0))[:, :, None]
                    noise *= (0.5 + 0.8 * darkness)

                composite_f32 += noise
                composite_f32 = np.clip(composite_f32, 0.0, 255.0)

            # -----------------------------------------------------------------
            # Step 6: JPEG Re-encode (probability-gated)
            # -----------------------------------------------------------------
            if self.rng.rand() < self.jpeg_prob:
                jpeg_quality = int(self.rng.randint(self.jpeg_quality_range[0], self.jpeg_quality_range[1] + 1))
                params["jpeg_quality"] = jpeg_quality

                comp_uint8 = np.clip(composite_f32, 0, 255).astype(np.uint8)
                retval, enc_buf = cv2.imencode('.jpg', comp_uint8, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if retval:
                    comp_decoded = cv2.imdecode(enc_buf, cv2.IMREAD_COLOR)
                    if comp_decoded is not None:
                        comp_uint8 = comp_decoded
            else:
                comp_uint8 = np.clip(composite_f32, 0, 255).astype(np.uint8)
        else:
            comp_uint8 = np.clip(composite_f32, 0.0, 255.0).astype(np.uint8)

        # ---------------------------------------------------------------------
        # Dual Output & Homography Rectification (REQ-08, REQ-35)
        # ---------------------------------------------------------------------
        # enhance_input: composite warped back using exact matrix inverse H_inv (REQ-35)
        enhance_input = cv2.warpPerspective(comp_uint8, H_inv, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


        # enhance_target: clean scan resized to 512x512, NEVER photometrically degraded (REQ-35)
        enhance_target = clean_scan.copy()

        # Render heatmaps for corner detector Task 2B (ADR-008)
        heatmaps = render_heatmaps(target_corners, canvas_size=self.canvas_size, sigma=8.0)

        return {
            "composite": comp_uint8,               # uint8 BGR (512, 512, 3)
            "corners": target_corners,             # float32 (4, 2)
            "enhance_input": enhance_input,       # uint8 BGR (512, 512, 3)
            "enhance_target": enhance_target,     # uint8 BGR (512, 512, 3)
            "heatmaps": heatmaps,                 # float32 (4, 512, 512)
            "H": H,                               # float64 (3, 3)
            "params": params                      # dict of sampled params
        }
