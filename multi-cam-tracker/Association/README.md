# Association Pipeline (Offline, Two Cameras)

This folder contains everything required to run cross-camera association offline for the football match.
The goal is to give each person a **single global ID** across both cameras.

## Folder Layout

```
Association/
  inputs/
    cam0_2026-01-19_18-59-13.json
    cam1_2026-01-19_18-59-13.json
    cam0_2026-01-19_18-59-13.mp4
    cam1_2026-01-19_18-59-13.mp4
  calib/
    H_right_to_left.npy
    left_fisheye_calib.npz
    right_fisheye_calib.npz
  outputs/
    cam0_2026-01-19_18-59-13_global.json
    cam1_2026-01-19_18-59-13_global.json
    association_report.json
    association_map.json
    association_map.csv
    association_overlay.mp4
  scripts/
    associate_offline.py
    export_association_artifacts.py
```

## Environment

Use the venv at:

```
/Users/adityanair/Dev/skillytics/.rf
```

Required packages (inside the venv):
- `opencv-python`
- `numpy<2`
- `supervision`
- `scipy`

If you need to install or fix versions:

```bash
/Users/adityanair/Dev/skillytics/.rf/bin/python -m pip install "numpy<2" supervision scipy
```

## Run Association (Global IDs)

This generates `*_global.json` files with `track_id` and `global_id` fields.

```bash
/Users/adityanair/Dev/skillytics/.rf/bin/python Association/scripts/associate_offline.py
```

Key defaults (in the script):
- Homography: `Association/calib/H_right_to_left.npy`
- Fisheye calibration: `Association/calib/left_fisheye_calib.npz`, `Association/calib/right_fisheye_calib.npz`
- Inputs: `Association/inputs/*.json` and `Association/inputs/*.mp4`
- Outputs: `Association/outputs/*_global.json`
- Association mode: `mutual_best` (one-to-one matching)

You can override parameters. Example:

```bash
/Users/adityanair/Dev/skillytics/.rf/bin/python Association/scripts/associate_offline.py \
  --dist-thresh 60 \
  --min-matches 10
```

## Export Map + Overlay Video

This produces:
- `association_map.json` / `association_map.csv`
- `association_overlay.mp4`

```bash
/Users/adityanair/Dev/skillytics/.rf/bin/python Association/scripts/export_association_artifacts.py
```

Optional flags:
- `--scale 0.6` (overlay resolution)
- `--max-frames 0` (0 = all frames)

## Outputs You Will Use

- **Global IDs**: stored in `Association/outputs/cam*_global.json` under `annotations[*].global_id`
- **Mapping table**: `association_map.csv` (global_id -> left/right track_id)
- **Overlay video**: `association_overlay.mp4` (visual sanity check)

## Notes

- The overlay video is large (approx 2 GB). This is expected for ~49k frames.
- `mutual_best` association ensures one-to-one matching between left and right tracks.
- If you want more matches, increase `--dist-thresh` or lower `--min-matches`.
- If you want stricter matches, reduce `--dist-thresh` or increase `--min-matches`.

## Support

If you want improvements (stronger matching, ReID, faster preview videos), just ask.
