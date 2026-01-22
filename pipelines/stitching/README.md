# Fisheye Video Stitching Pipeline

This directory contains tools for stitching videos from two fisheye cameras into a single panoramic video. The pipeline supports camera calibration, homography estimation, and multiple stitching methods.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [Workflow](#workflow)
- [Scripts Overview](#scripts-overview)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### 1. Camera Calibration

First, calibrate your fisheye cameras to get distortion parameters:

```bash
python pipelines/stitching/calibrate_cameras.py \
    --left_dir data/calibration/left \
    --right_dir data/calibration/right \
    --out_dir data/calibration_new
```

This generates:
- `left_fisheye_calib.npz` - Left camera calibration
- `right_fisheye_calib.npz` - Right camera calibration

### 2. Homography Calibration (Optional but Recommended)

For better alignment, manually calibrate the homography between cameras:

```bash
python pipelines/stitching/calibrate_homography_fisheye_v5.py \
    --left_img data/calibration_new/left/left_000.png \
    --right_img data/calibration_new/right/right_000.png \
    --left_calib data/calibration_new/left_fisheye_calib.npz \
    --right_calib data/calibration_new/right_fisheye_calib.npz \
    --out_H data/calibration_new/homography_right_to_left_fisheye.npy \
    --do_ecc \
    --do_exposure
```

**Interactive Process:**
- Click corresponding turf points on LEFT then RIGHT (alternating)
- Need at least 4 point pairs
- Press `ENTER` when done
- Press `u` to undo, `r` to reset

### 3. Stitch Videos

Use the recommended stitching method:

```bash
python pipelines/stitching/stitch_open_stitcher_final.py \
    --left_video data/cam0-video.mp4 \
    --right_video data/cam1-video.mp4 \
    --left_calib data/calibration_new/left_fisheye_calib.npz \
    --right_calib data/calibration_new/right_fisheye_calib.npz \
    --out data/output/stitched_video.mp4 \
    --balance 0.8 \
    --fov_scale 1.0 \
    --ref_frame 120 \
    --stitch_mode PANORAMA \
    --right_offset 0
```

## 📁 Directory Structure

```
pipelines/stitching/
├── README.md                          # This file
├── FILES_ORGANIZATION.md              # Detailed file organization guide
│
├── Core Utilities
│   └── stitch_fisheye_panaroma.py    # Core functions (DO NOT DELETE)
│
├── Video Stitching (Main Scripts)
│   ├── stitch_open_stitcher_final.py  # ⭐ RECOMMENDED - OpenCV Stitcher with frame offset
│   ├── stitch_fisheye_video_opencv_only.py  # Simpler version (no frame offset)
│   ├── stitch_fisheye_video_hybrid.py # Alternative: Hybrid method
│   └── stitch_fisheye_video_autoh_fixedblend.py  # Auto-homography method
│
├── Calibration
│   ├── calibrate_cameras.py           # Camera fisheye calibration
│   ├── calibrate_homography_fisheye_v5.py  # Manual homography calibration (fisheye)
│   ├── calibrate_homography.py        # Manual homography calibration (non-fisheye)
│   └── pick_turf_pts.py              # Helper for point picking
│
└── Utilities
    ├── extract_video_segment.py      # Extract time segments from videos
    ├── stabilize_video.py             # Post-process stabilization
    └── stitch_turf_plane_preview.py   # Preview tool (optional)
```

## 🔄 Workflow

```
1. Camera Calibration
   └─> Generate .npz calibration files

2. Homography Calibration (Optional)
   └─> Generate .npy homography matrix

3. Video Stitching
   └─> Generate stitched panoramic video

4. Post-Processing (Optional)
   └─> Stabilize if needed
```

## 📝 Scripts Overview

### Core Utilities

#### `stitch_fisheye_panaroma.py`
**Purpose:** Core utility module with shared functions  
**Functions:**
- `load_fisheye_npz()` - Load camera calibration
- `undistort_fisheye()` - Undistort fisheye images
- `find_homography_auto()` - Auto homography computation
- `compute_canvas()` - Canvas size computation
- `linear_blend_lr()` - Blending functions
- `crop_to_mask()` - Cropping utilities

**⚠️ DO NOT DELETE** - Other scripts import from this file.

### Video Stitching Scripts

#### `stitch_open_stitcher_final.py` ⭐ **RECOMMENDED**
**Purpose:** Stitch videos using OpenCV Stitcher with fixed transform  
**Features:**
- No jitter (fixed transform per frame)
- Frame offset support (`--right_offset`) - handles camera sync issues
- Best quality output
- Uses `estimateTransform()` once + `composePanorama()` per frame

**When to use:** Default choice for most use cases, especially when cameras may be out of sync

#### `stitch_fisheye_video_opencv_only.py`
**Purpose:** Simpler version without frame offset support  
**Features:**
- Same core functionality as `stitch_open_stitcher_final.py`
- No `--right_offset` parameter
- Simpler interface

**When to use:** When cameras are perfectly synchronized and you don't need offset functionality

#### `stitch_fisheye_video_hybrid.py`
**Purpose:** Hybrid stitching method  
**Features:**
- OpenCV Stitcher quality on first frame
- Fixed homography for remaining frames
- Combines best of both approaches

**When to use:** Alternative if `stitch_open_stitcher_final.py` has issues

#### `stitch_fisheye_video_autoh_fixedblend.py`
**Purpose:** Auto-estimate homography from clean frame  
**Features:**
- Automatically estimates homography from reference frame
- Saves homography for reuse
- Fixed feather blending

**When to use:** When saved homography doesn't align (camera shifts, different setup)

### Calibration Scripts

#### `calibrate_cameras.py`
**Purpose:** Calibrate fisheye camera distortion parameters  
**Input:** Chessboard images from both cameras  
**Output:** `.npz` files with K (camera matrix) and D (distortion coefficients)

#### `calibrate_homography_fisheye_v5.py`
**Purpose:** Manually calibrate homography between fisheye cameras  
**Process:** Interactive point picking on undistorted fisheye images  
**Output:** `.npy` homography matrix (right→left transformation)

**Options:**
- `--do_ecc` - Enable ECC refinement for better alignment
- `--do_exposure` - Match exposure between cameras
- `--crop` - Crop preview images

#### `calibrate_homography.py`
**Purpose:** Manually calibrate homography for regular (non-fisheye) cameras  
**Process:** Interactive point picking on regular images (no undistortion)  
**Output:** `.npy` homography matrix (right→left transformation)

**When to use:** For standard cameras without fisheye distortion

### Utility Scripts

#### `extract_video_segment.py`
**Purpose:** Extract time segments from videos  
**Example:**
```bash
python extract_video_segment.py \
    --input data/video.mp4 \
    --output data/segment.mp4 \
    --start 03:30 \
    --end 05:30
```

#### `stabilize_video.py`
**Purpose:** Stabilize already-stitched videos  
**Use case:** Post-process videos with jitter

#### `stitch_turf_plane_preview.py`
**Purpose:** Preview tool for debugging stitching  
**Use case:** Visual inspection of calibration results

## 💡 Examples

### Basic Stitching

```bash
python pipelines/stitching/stitch_open_stitcher_final.py \
    --left_video data/cam0.mp4 \
    --right_video data/cam1.mp4 \
    --left_calib data/calibration_new/left_fisheye_calib.npz \
    --right_calib data/calibration_new/right_fisheye_calib.npz \
    --out data/output/stitched.mp4
```

### With Frame Offset (Right Camera Late)

If your right camera is 2 frames behind:

```bash
python pipelines/stitching/stitch_open_stitcher_final.py \
    --left_video data/cam0.mp4 \
    --right_video data/cam1.mp4 \
    --left_calib data/calibration_new/left_fisheye_calib.npz \
    --right_calib data/calibration_new/right_fisheye_calib.npz \
    --out data/output/stitched.mp4 \
    --right_offset 2
```

### Using SCANS Mode (For Side-by-Side Fixed Cameras)

```bash
python pipelines/stitching/stitch_open_stitcher_final.py \
    --left_video data/cam0.mp4 \
    --right_video data/cam1.mp4 \
    --left_calib data/calibration_new/left_fisheye_calib.npz \
    --right_calib data/calibration_new/right_fisheye_calib.npz \
    --out data/output/stitched.mp4 \
    --stitch_mode SCANS
```

### Extract Video Segment for Testing

```bash
python pipelines/stitching/extract_video_segment.py \
    --input data/long_video.mp4 \
    --output data/test_segment.mp4 \
    --start 00:30 \
    --end 01:00
```

## 🔧 Parameters Explained

### Stitching Parameters

- `--balance` (default: 0.8)
  - Fisheye undistort balance (0.0 = crop more, 1.0 = keep more FOV)
  - Lower = less distortion but smaller FOV
  - Higher = more FOV but more distortion at edges

- `--fov_scale` (default: 1.0)
  - Field of view scale factor
  - 1.0 = original FOV
  - < 1.0 = zoom in
  - > 1.0 = zoom out

- `--ref_frame` (default: 120)
  - Reference frame for estimating transform
  - Pick a frame with clear overlap and no players crossing seam
  - Should be representative of typical scene

- `--stitch_mode` (default: PANORAMA)
  - `PANORAMA` - For rotating/moving cameras
  - `SCANS` - For side-by-side fixed cameras (often better)

- `--right_offset` (default: 0)
  - Frame offset for right camera
  - Positive = right camera is late
  - Negative = right camera is early

### Calibration Parameters

- `--ransac_px` (default: 5.0)
  - RANSAC reprojection threshold in pixels
  - Lower = stricter matching
  - Higher = more tolerant

- `--do_ecc`
  - Enable ECC (Enhanced Correlation Coefficient) refinement
  - Improves alignment accuracy
  - Slower but better results

- `--do_exposure`
  - Match exposure between cameras
  - Adjusts right camera brightness to match left

## 🐛 Troubleshooting

### Issue: Seam jitter/wobble in output

**Solution:** Use `stitch_open_stitcher_final.py` (fixed transform method)

### Issue: Poor alignment

**Solutions:**
1. Re-calibrate homography with `calibrate_homography_fisheye_v5.py`
2. Try different `--ref_frame` value
3. Use `--do_ecc` flag for refinement
4. Check that cameras haven't moved since calibration

### Issue: Stretching/distortion at edges

**Solutions:**
1. Adjust `--balance` parameter (try 0.6-0.9)
2. Adjust `--fov_scale` parameter
3. Use `stitch_open_stitcher_final.py` instead of homography-based methods

### Issue: Cameras out of sync

**Solution:** Use `--right_offset` parameter to align frames

### Issue: OpenCV Stitcher fails

**Solutions:**
1. Try `--stitch_mode SCANS` instead of `PANORAMA`
2. Use different `--ref_frame` (pick frame with clear overlap)
3. Check that undistorted images have sufficient overlap
4. Try `stitch_fisheye_video_hybrid.py` as alternative

## 📚 Additional Resources

- See `FILES_ORGANIZATION.md` for detailed file descriptions
- Camera calibration requires chessboard images (see `calibrate_cameras.py`)
- Homography calibration is interactive - follow on-screen instructions

## 🔗 Dependencies

- OpenCV (cv2)
- NumPy
- Pathlib (standard library)

## 📄 License

Part of the Skillytics project.

