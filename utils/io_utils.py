"""
I/O utilities for the Document Scanning & Enhancement project.

Functions:
    load_image(path) -> np.ndarray
        Load an image from disk with OpenCV, convert BGR->RGB, return uint8.

    save_image(image, path) -> None
        Convert RGB->BGR and save with OpenCV.

    load_coco_polygon_annotations(json_path) -> dict
        Parse the RoboFlow COCO polygon segmentation JSON.
        Returns: {original_filename: np.ndarray of shape (4, 2) in TL,TR,BR,BL order}

        The annotation JSON uses polygon segmentation (NOT keypoints).
        Each annotation's segmentation field is [[x1,y1, x2,y2, x3,y3, x4,y4, x1,y1]].
        The 5th point repeats the 1st (closing the polygon) and should be discarded.
        Image filenames are recovered from images[i].extra.name (not file_name,
        which is the RoboFlow-mangled version).

    sort_corners_clockwise(points: np.ndarray) -> np.ndarray
        Sort 4 unordered polygon vertices into TL, TR, BR, BL order.
        Algorithm: compute centroid, classify each point by quadrant relative
        to centroid. TL = (x<cx, y<cy), TR = (x>cx, y<cy), etc.
        Input:  shape (4, 2) in arbitrary order
        Output: shape (4, 2) in [TL, TR, BR, BL] order

    load_split_definition(split_file) -> list[str]
        Load scan filenames from a split definition text file.

    save_frozen_sample(sample_dict, save_dir, index) -> None
        Save a generated sample (images as PNG, corners as .npy) to disk.
"""
