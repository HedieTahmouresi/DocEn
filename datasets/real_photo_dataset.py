"""
PyTorch Dataset classes for real phone photos used in evaluation.

NOTE: Annotations are in COCO polygon segmentation format (NOT keypoints).
Corner coordinates must be extracted from polygon vertices and sorted
into TL, TR, BR, BL order using sort_corners_clockwise() from utils/io_utils.py.
Filenames are mapped via images[i].extra.name in the annotation JSON.

Classes:
    RealEnhancementDataset:
        Loads rectified real photos + reference scans for enhancement evaluation.
        For each sample:
        1. Load raw photo from data/real_photos/raw/
        2. Load annotated polygon corners from COCO JSON (sorted to TL,TR,BR,BL)
        3. Rectify the photo using corners + cv2.getPerspectiveTransform
        4. Load matching reference scan from data/real_photos/reference_scans/
        5. Resize both to (CFG.IMG_SIZE, CFG.IMG_SIZE), normalize to [0,1]
        Returns: (rectified_input, reference_scan, photo_name)

    RealCornerDataset:
        Loads raw photos + sorted corner coordinates for corner detection evaluation.
        For each sample:
        1. Load raw photo
        2. Load polygon corners from COCO JSON (sorted to TL,TR,BR,BL)
        3. Resize photo to (CFG.IMG_SIZE, CFG.IMG_SIZE)
        4. Scale and normalize corner coordinates to [0,1]
        Returns: (image, corners, photo_name)
"""
