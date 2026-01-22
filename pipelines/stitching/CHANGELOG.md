# Changelog - Stitching Pipeline

## 2025-01-02 - Organization for GitHub

### Added
- **README.md** - Comprehensive documentation with quick start guide, examples, and troubleshooting
- **FILES_ORGANIZATION.md** - Updated with current file status and migration notes
- **.gitignore** - Ignores Python cache files and temporary files
- **__init__.py** - Package initialization file

### Restored
- **stitch_fisheye_video_opencv_only.py** - Restored (simpler version without frame offset)
- **calibrate_homography.py** - Restored (for non-fisheye cameras)

### Updated
- **FILES_ORGANIZATION.md** - Reflects current state, marks `stitch_open_stitcher_final.py` as recommended

### Current Recommended Scripts
- **Video Stitching:** `stitch_open_stitcher_final.py` (supports frame offset)
- **Camera Calibration:** `calibrate_cameras.py`
- **Homography Calibration:** `calibrate_homography_fisheye_v5.py`

### Migration Notes
If you were using `stitch_fisheye_video_opencv_only.py`, migrate to `stitch_open_stitcher_final.py`:
- Same functionality
- Additional `--right_offset` parameter for camera sync
- Same command-line interface otherwise

