# Stitching Folder - File Organization Guide

**Last Updated:** 2025-01-02  
**Status:** Organized for GitHub

## ✅ **ESSENTIAL FILES** (Keep These)

### Core Utility (Required by all scripts)
- **`stitch_fisheye_panaroma.py`** ⭐ **MOST IMPORTANT**
  - Contains core functions used by all other scripts:
    - `load_fisheye_npz()` - Load camera calibration
    - `undistort_fisheye()` - Undistort fisheye images
    - `find_homography_auto()` - Auto homography computation
    - `compute_canvas()` - Canvas size computation
    - `linear_blend_lr()` - Blending functions
    - `crop_to_mask()` - Cropping utilities
  - **DO NOT DELETE** - Other scripts import from this

### Video Stitching Scripts (Choose based on your needs)
- **`stitch_open_stitcher_final.py`** ⭐ **RECOMMENDED**
  - Best quality + no jitter (uses fixed transform)
  - Uses `estimateTransform()` once + `composePanorama()` per frame
  - **Supports frame offset (`--right_offset`)** - handles camera sync issues
  - **This is your current best option**

- **`stitch_fisheye_video_hybrid.py`**
  - Alternative method (hybrid approach)
  - Uses OpenCV Stitcher quality + fixed homography
  - Keep if you want to compare methods

- **`stitch_fisheye_video_autoh_fixedblend.py`**
  - Auto-estimates homography from clean frame
  - Useful when saved homography doesn't align (camera shifts)
  - Saves homography for reuse

### Post-Processing Tools
- **`stabilize_video.py`**
  - Stabilizes already-stitched videos
  - Useful if you have jittery output from other methods

### Utility Scripts
- **`extract_video_segment.py`**
  - Extract time segments from videos
  - Useful for testing on shorter clips

### Calibration Scripts
- **`calibrate_cameras.py`**
  - Camera calibration (fisheye parameters)
  - Needed for initial setup
  - Generates `.npz` calibration files

- **`calibrate_homography_fisheye_v5.py`** ⭐ **LATEST VERSION**
  - Manual homography calibration (latest version)
  - Interactive point picking
  - Generates `.npy` homography matrix
  - Keep only v5, remove v3 and v4 if they exist

- **`pick_turf_pts.py`**
  - Utility for picking calibration points
  - Needed for manual calibration

### Preview/Debug Tools (Optional)
- **`stitch_turf_plane_preview.py`**
  - Preview tool for debugging
  - Optional but useful for visual inspection

---

## 🗑️ **FILES TO REMOVE** (Redundant/Deprecated)

### Duplicate/Redundant Files
- **`stitch_fisheye_video_opencv_only.py`** 
  - **Status:** Duplicate of `stitch_open_stitcher_final.py` but without `right_offset` support
  - **Action:** Can be removed (use `stitch_open_stitcher_final.py` instead)

### Deprecated/Old Versions
- **`calibrate_homography.py`**
  - Non-fisheye version
  - **Action:** Remove if you only use fisheye cameras

### Files Not Present (Already Cleaned)
- `stitch_backup.py` - Already removed
- `stitch_fisheye_panorama_fixed.py` - Already removed
- `calibrate_homography_fisheye_v3.py` - Already removed
- `calibrate_homography_fisheye_v4.py` - Already removed
- `stitch.py` - Already removed
- `stitch_video.py` - Already removed
- `stitch_preview.py` - Already removed

---

## 📋 **MINIMUM REQUIRED FILES** (For Clean Installation)

### Absolute Minimum:
1. `stitch_fisheye_panaroma.py` ⭐ (core utilities - DO NOT DELETE)
2. `stitch_open_stitcher_final.py` ⭐ (best video stitching)
3. `calibrate_cameras.py` (camera calibration)
4. `calibrate_homography_fisheye_v5.py` (homography calibration)

### Recommended Additional:
5. `stabilize_video.py` (post-processing)
6. `extract_video_segment.py` (utility)
7. `stitch_fisheye_video_hybrid.py` (alternative method)
8. `stitch_fisheye_video_autoh_fixedblend.py` (auto-homography method)
9. `pick_turf_pts.py` (calibration helper)
10. `stitch_turf_plane_preview.py` (preview tool)

---

## 📊 **Current File Status** (2025-01-02)

### ✅ Keep (Essential):
- `stitch_fisheye_panaroma.py` - Core utilities
- `stitch_open_stitcher_final.py` - **RECOMMENDED** stitching script
- `stitch_fisheye_video_hybrid.py` - Alternative method
- `stitch_fisheye_video_autoh_fixedblend.py` - Auto-homography method
- `stabilize_video.py` - Post-processing
- `extract_video_segment.py` - Utility
- `calibrate_cameras.py` - Camera calibration
- `calibrate_homography_fisheye_v5.py` - Homography calibration
- `pick_turf_pts.py` - Calibration helper

### ⚠️ Optional (Keep if useful):
- `stitch_turf_plane_preview.py` - Preview/debug tool
- `stitch_fisheye_video_opencv_only.py` - Simpler stitching (no frame offset)
- `calibrate_homography.py` - For non-fisheye cameras

### ⚠️ Alternative Versions (Keep if needed):
- `stitch_fisheye_video_opencv_only.py` - Simpler version without `right_offset` support
  - Use if you don't need frame offset functionality
  - Otherwise, prefer `stitch_open_stitcher_final.py`
  
- `calibrate_homography.py` - Non-fisheye version
  - For regular (non-fisheye) cameras
  - Use `calibrate_homography_fisheye_v5.py` for fisheye cameras

---

## 🎯 **Quick Summary**

**For video stitching (your main use case):**
- **Use:** `stitch_open_stitcher_final.py` ⭐ (best quality, no jitter, supports frame offset)
- **Backup:** `stitch_fisheye_video_hybrid.py` (alternative)
- **Auto-H:** `stitch_fisheye_video_autoh_fixedblend.py` (when saved H doesn't work)
- **Core:** `stitch_fisheye_panaroma.py` (provides utilities - DO NOT DELETE)

**For calibration:**
- **Camera:** `calibrate_cameras.py`
- **Homography:** `calibrate_homography_fisheye_v5.py`
- **Helper:** `pick_turf_pts.py`

**For utilities:**
- **Extract segments:** `extract_video_segment.py`
- **Stabilize:** `stabilize_video.py`
- **Preview:** `stitch_turf_plane_preview.py`

---

## 📝 **Migration Notes**

If you were using `stitch_fisheye_video_opencv_only.py`, migrate to `stitch_open_stitcher_final.py`:

**Old:**
```bash
python stitch_fisheye_video_opencv_only.py --left_video ... --right_video ...
```

**New:**
```bash
python stitch_open_stitcher_final.py --left_video ... --right_video ... --right_offset 0
```

The new script has the same functionality plus frame offset support.
