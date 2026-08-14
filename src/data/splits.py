"""
Split assignment utility for document scans.

Per REQ-14 and REQ-17:
- 80% train / 10% val / 10% test
- Assigned by MD5 hash of filename (stable across runs and environments)
- Both enhancement and corner detection share the exact same split
- Saves splits.json and asserts disjointness
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


def assign_split_by_hash(filename: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    """
    Assign a filename to 'train', 'val', or 'test' using MD5 hash modulo 100.
    """
    hash_hex = hashlib.md5(filename.encode("utf-8")).hexdigest()
    hash_val = int(hash_hex, 16) % 100
    
    train_thresh = int(train_ratio * 100)
    val_thresh = train_thresh + int(val_ratio * 100)
    
    if hash_val < train_thresh:
        return "train"
    elif hash_val < val_thresh:
        return "val"
    else:
        return "test"


def create_splits(
    scans_dir: Path,
    out_file: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Dict[str, List[str]]:
    """
    Scan directory for document files and partition them into train/val/test splits.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    scan_files = sorted([
        f.name for f in scans_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ])
    
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for filename in scan_files:
        split_name = assign_split_by_hash(filename, train_ratio=train_ratio, val_ratio=val_ratio)
        splits[split_name].append(filename)
        
    # Assert disjointness
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    
    assert len(train_set & val_set) == 0, "Train and Val splits overlap!"
    assert len(train_set & test_set) == 0, "Train and Test splits overlap!"
    assert len(val_set & test_set) == 0, "Val and Test splits overlap!"
    assert len(train_set | val_set | test_set) == len(scan_files), "Missing files in split!"
    
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
        
    print(f"Generated splits -> Total: {len(scan_files)} | Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    return splits


if __name__ == "__main__":
    import sys
    root_dir = Path(__file__).resolve().parent.parent.parent
    scans_path = root_dir / "data" / "clean_scans"
    splits_path = root_dir / "data" / "splits" / "splits.json"
    create_splits(scans_path, splits_path)
