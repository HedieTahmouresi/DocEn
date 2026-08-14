"""OCR evaluation metrics: Character Error Rate (CER) and Tesseract Word Confidence.

Implements matched-resolution OCR evaluation protocol per evaluation-spec.md §5b.
"""

import re
from typing import Dict, Any, Union, Tuple
import numpy as np
from PIL import Image
import pytesseract


def normalize_text_for_cer(text: str) -> str:
    """Normalize text for CER calculation per evaluation-spec.md §5b.

    Rules:
    - Collapse whitespace (multiple spaces, newlines, tabs -> single space)
    - Strip leading and trailing whitespace
    - DO NOT lowercase
    - DO NOT strip punctuation
    """
    if not text:
        return ""
    # Replace any whitespace sequence with a single space
    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.strip()


def levenshtein_distance(str1: str, str2: str) -> int:
    """Compute exact Levenshtein edit distance between two strings."""
    m, n = len(str1), len(str2)
    if m == 0:
        return n
    if n == 0:
        return m

    # Use 2-row DP for O(min(m, n)) space
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if str1[i - 1] == str2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp

    return dp[n]


def compute_cer(pred_text: str, gt_text: str) -> float:
    """Compute Character Error Rate (CER) = Levenshtein(pred, gt) / len(gt).

    Parameters
    ----------
    pred_text : str
        Raw or normalized predicted OCR text.
    gt_text : str
        Ground truth reference text.

    Returns
    -------
    float
        CER ratio (0.0 means perfect match).
    """
    norm_pred = normalize_text_for_cer(pred_text)
    norm_gt = normalize_text_for_cer(gt_text)

    if len(norm_gt) == 0:
        return 0.0 if len(norm_pred) == 0 else 1.0

    dist = levenshtein_distance(norm_pred, norm_gt)
    return float(dist / len(norm_gt))


def run_ocr_on_image(
    image: Union[np.ndarray, Image.Image],
    psm: int = 6,
    lang: str = "eng",
) -> Tuple[str, float]:
    """Run Tesseract OCR on an image and extract predicted text & mean word confidence.

    Single-pass invocation: extracts data dict containing text and word confidences
    in a single Tesseract call to eliminate redundant process spawns.

    Parameters
    ----------
    image : np.ndarray or PIL.Image
        Input image (RGB or Greyscale uint8).
    psm : int, default=6
        Page segmentation mode (--psm 6 assumes a single uniform block of text).
    lang : str, default='eng'

    Returns
    -------
    Tuple[str, float]
        (predicted_text, mean_confidence)
    """
    config = f"--psm {psm}"

    # Extract word confidences and text in one single Tesseract call
    data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    confidences = []
    text_words = []
    if "conf" in data:
        for conf, word in zip(data["conf"], data.get("text", [])):
            word_str = str(word).strip()
            if word_str:
                text_words.append(word_str)
                try:
                    c_val = float(conf)
                    if c_val >= 0:
                        confidences.append(c_val)
                except (ValueError, TypeError):
                    continue

    text = " ".join(text_words)
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    return text, mean_conf
