#!/usr/bin/env python3
"""
Core utility functions for fisheye stitching.

This module provides shared functions used by stitching scripts:
- load_fisheye_npz() - Load camera calibration
- undistort_fisheye() - Undistort fisheye images
"""
import numpy as np
import cv2


def load_fisheye_npz(npz_path: str):
    """Load fisheye camera calibration from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    
    # Support common key names
    for kK in ["K", "camera_matrix", "mtx"]:
        if kK in data:
            K = data[kK].astype(np.float64)
            break
    else:
        raise KeyError(f"{npz_path}: could not find K/camera_matrix/mtx")

    for kD in ["D", "dist_coeffs", "dist"]:
        if kD in data:
            D = data[kD].astype(np.float64).reshape(-1, 1)
            break
    else:
        raise KeyError(f"{npz_path}: could not find D/dist_coeffs/dist")

    return K, D


def undistort_fisheye(img, K, D, balance=0.35, fov_scale=1.0):
    """Undistort a fisheye image."""
    h, w = img.shape[:2]
    dim = (w, h)
    R = np.eye(3, dtype=np.float64)

    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, dim, R, balance=balance
    ).astype(np.float64)

    # fov_scale > 1.0 = zoom in a bit (smaller FOV), < 1.0 = wider
    newK[0, 0] *= fov_scale
    newK[1, 1] *= fov_scale

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, newK, dim, cv2.CV_16SC2
    )
    und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return und

