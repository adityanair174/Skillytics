#!/usr/bin/env python3
"""
Stitch fisheye videos using OpenCV Stitcher with FIXED transform (no jitter).

This uses OpenCV Stitcher's two-phase approach:
1. estimateTransform() ONCE on a reference frame (locks the seam/warp)
2. composePanorama() for every frame (reuses the fixed transform)

This eliminates jitter while maintaining OpenCV Stitcher's excellent quality.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

# Import fisheye undistortion functions
sys.path.insert(0, str(Path(__file__).parent))
from stitch_fisheye_panaroma import load_fisheye_npz, undistort_fisheye


def stitch_videos_opencv_only(
    left_video_path: Path,
    right_video_path: Path,
    output_path: Path,
    left_calib: Path,
    right_calib: Path,
    balance: float = 0.35,
    fov_scale: float = 1.0,
    ref_frame: int = 120,
    stitch_mode: str = "PANORAMA",
    right_offset: int = 0,
):
    """
    Stitch two fisheye videos using OpenCV Stitcher with a FIXED transform:
    - estimateTransform() once on a reference frame
    - composePanorama() for every frame (stable, no jitter)

    Offset convention:
      right_offset = +2 means right cam is 2 frames "late",
      so we use right[t + 2] with left[t].

    Args:
        ref_frame: Frame number (in stitched timeline) used for estimating transform
        stitch_mode: "PANORAMA" or "SCANS"
        right_offset: integer frame offset applied to the RIGHT video
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load calibration data
    print("Loading camera calibrations...")
    K1, D1 = load_fisheye_npz(str(left_calib))
    K2, D2 = load_fisheye_npz(str(right_calib))
    print("✅ Calibrations loaded")

    # Open input videos
    print(f"\nOpening input videos...")
    cap_left = cv2.VideoCapture(str(left_video_path))
    cap_right = cv2.VideoCapture(str(right_video_path))

    if not cap_left.isOpened():
        raise RuntimeError(f"Could not open left video: {left_video_path}")
    if not cap_right.isOpened():
        raise RuntimeError(f"Could not open right video: {right_video_path}")

    # Get video properties
    fps_l = cap_left.get(cv2.CAP_PROP_FPS)
    fps_r = cap_right.get(cv2.CAP_PROP_FPS)
    fps = fps_l if fps_l > 0 else fps_r

    left_total = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    right_total = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))

    # Support negative offsets too (shifts LEFT instead)
    left_start = max(0, -right_offset)
    right_start = max(0, right_offset)

    usable_left = left_total - left_start
    usable_right = right_total - right_start
    frame_count = int(min(usable_left, usable_right))

    if frame_count <= 0:
        raise RuntimeError(
            f"Offset too large. left_total={left_total}, right_total={right_total}, "
            f"left_start={left_start}, right_start={right_start}"
        )

    print(f"📹 Video properties:")
    print(f"   FPS left/right: {fps_l:.2f} / {fps_r:.2f}")
    print(f"   Total frames left/right: {left_total} / {right_total}")
    print(f"   Using right_offset={right_offset} → left_start={left_start}, right_start={right_start}")
    print(f"   Usable frames: {frame_count}")

    # -------------------------------
    # ✅ 1) Create stitcher ONCE
    # -------------------------------
    mode = cv2.Stitcher_PANORAMA if stitch_mode.upper() == "PANORAMA" else cv2.Stitcher_SCANS
    stitcher = cv2.Stitcher.create(mode)

    # These often help stability for static rigs
    stitcher.setWaveCorrection(False)  # reduces "wobble" in panoramas

    # -------------------------------
    # ✅ 2) Estimate transform ONCE on a reference frame (in stitched timeline)
    # -------------------------------
    ref_frame = max(0, min(ref_frame, frame_count - 1))
    print(f"\n🎯 Estimating transform once using reference frame: {ref_frame}")
    print(f"   (Pick a frame where overlap is clear and no player crosses the seam)")
    print(f"   Applying offset: left[{left_start + ref_frame}] vs right[{right_start + ref_frame}]")

    cap_left.set(cv2.CAP_PROP_POS_FRAMES, left_start + ref_frame)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, right_start + ref_frame)

    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()
    if not ret_l or not ret_r:
        raise RuntimeError("Could not read reference frames from both videos (after applying offset).")

    left_u = undistort_fisheye(frame_l, K1, D1, balance=balance, fov_scale=fov_scale)
    right_u = undistort_fisheye(frame_r, K2, D2, balance=balance, fov_scale=fov_scale)

    status = stitcher.estimateTransform([left_u, right_u])
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"estimateTransform failed on ref_frame={ref_frame} with status: {status}")

    status, pano0 = stitcher.composePanorama([left_u, right_u])
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"composePanorama failed on ref_frame={ref_frame} with status: {status}")

    out_h, out_w = pano0.shape[:2]
    print(f"🧩 Output panorama size: {out_w}x{out_h}")
    print(f"✅ Using FIXED transform (composePanorama per frame) → seam jitter eliminated")

    # -------------------------------
    # ✅ 3) Reset streams to aligned start and write output
    # -------------------------------
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, left_start)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, right_start)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    last_successful_frame = pano0.copy()
    success_count = 0
    fail_count = 0
    frame_num = 0

    print("\nProcessing frames with fixed transform...")

    while frame_num < frame_count:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        if not ret_l or not ret_r:
            break

        left_u = undistort_fisheye(frame_l, K1, D1, balance=balance, fov_scale=fov_scale)
        right_u = undistort_fisheye(frame_r, K2, D2, balance=balance, fov_scale=fov_scale)

        status, panorama = stitcher.composePanorama([left_u, right_u])

        if status == cv2.Stitcher_OK:
            if panorama.shape[:2] != (out_h, out_w):
                panorama = cv2.resize(panorama, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            out.write(panorama)
            last_successful_frame = panorama.copy()
            success_count += 1
        else:
            out.write(last_successful_frame)
            fail_count += 1
            if fail_count <= 5:
                print(f"   ⚠️ Frame {frame_num} composePanorama failed (status: {status}), using previous frame")

        frame_num += 1
        if frame_num % 60 == 0:
            progress = (frame_num / frame_count) * 100
            print(f"   Processed {frame_num}/{frame_count} ({progress:.1f}%) - "
                  f"Success: {success_count}, Failed: {fail_count}")

    # Cleanup
    cap_left.release()
    cap_right.release()
    out.release()

    print(f"\n" + "=" * 70)
    print(f"✅ Video stitching complete!")
    print(f"   Output: {output_path}")
    print(f"   Total frames written: {frame_num}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {fail_count}")
    if frame_num > 0:
        print(f"   Success rate: {(success_count/frame_num)*100:.1f}%")
    print(f"   Method: Fixed transform (no jitter)")
    print(f"   right_offset applied: {right_offset}")
    print(f"=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Stitch fisheye videos using OpenCV Stitcher with FIXED transform (no jitter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--left_video", type=Path, default=Path("data/cam_left_10min.mp4"))
    parser.add_argument("--right_video", type=Path, default=Path("data/cam_right_10min.mp4"))
    parser.add_argument("--left_calib", type=Path, default=Path("data/calibration_new/left_fisheye_calib.npz"))
    parser.add_argument("--right_calib", type=Path, default=Path("data/calibration_new/right_fisheye_calib.npz"))
    parser.add_argument("--out", type=Path, default=Path("data/output/stitched_video_opencv_only.mp4"))
    parser.add_argument("--balance", type=float, default=0.8)
    parser.add_argument("--fov_scale", type=float, default=1.0)
    parser.add_argument("--ref_frame", type=int, default=120)
    parser.add_argument("--stitch_mode", type=str, default="PANORAMA", choices=["PANORAMA", "SCANS"])

    # ✅ NEW: right camera frame offset
    parser.add_argument(
        "--right_offset",
        type=int,
        default=0,
        help="Frame offset applied to RIGHT video. Use +2 if right cam is ~2 frames late."
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.left_video.exists():
        raise FileNotFoundError(f"Left video not found: {args.left_video}")
    if not args.right_video.exists():
        raise FileNotFoundError(f"Right video not found: {args.right_video}")
    if not args.left_calib.exists():
        raise FileNotFoundError(f"Left calibration not found: {args.left_calib}")
    if not args.right_calib.exists():
        raise FileNotFoundError(f"Right calibration not found: {args.right_calib}")

    stitch_videos_opencv_only(
        args.left_video,
        args.right_video,
        args.out,
        args.left_calib,
        args.right_calib,
        balance=args.balance,
        fov_scale=args.fov_scale,
        ref_frame=args.ref_frame,
        stitch_mode=args.stitch_mode,
        right_offset=args.right_offset,
    )


if __name__ == "__main__":
    main()
