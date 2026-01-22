#!/usr/bin/env python3
"""
Stitch fisheye videos using OpenCV Stitcher with FIXED transform (no jitter).

This uses OpenCV Stitcher's two-phase approach:
1. estimateTransform() ONCE on a reference frame (locks the seam/warp)
2. composePanorama() for every frame (reuses the fixed transform)

This eliminates jitter while maintaining OpenCV Stitcher's excellent quality.

IMPROVEMENTS:
- High-quality intermediate AVI (MJPG) + ffmpeg encoding for best quality
- Stitcher settings to reduce seam/exposure flicker
- Optional padding (disabled by default to reduce flicker)
"""
import argparse
import sys
import subprocess
from pathlib import Path
import numpy as np
import cv2

# Import fisheye undistortion functions
sys.path.insert(0, str(Path(__file__).parent))
from stitch_fisheye_panaroma import load_fisheye_npz, undistort_fisheye


def pad(img, p=200):
    """
    Add padding around an image to preserve content during stitching.
    
    Args:
        img: Input image
        p: Padding size in pixels (default: 200)
    
    Returns:
        Padded image with black borders
    """
    return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def make_writer(path, fps, size):
    """
    Create a VideoWriter using MJPG codec for high-quality intermediate AVI.
    
    This writes an uncompressed/lossless intermediate that will be encoded
    to MP4 with ffmpeg for best quality.
    
    Args:
        path: Output video path (should be .avi)
        fps: Frames per second
        size: (width, height) tuple
    
    Returns:
        cv2.VideoWriter instance
    
    Raises:
        RuntimeError: If VideoWriter cannot be opened
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    print(f"✅ Writing intermediate AVI (MJPG): {path}")
    return out


def encode_with_ffmpeg(avi_path, mp4_path, crf=17, preset="slow"):
    """
    Encode AVI intermediate to high-quality MP4 using ffmpeg.
    
    Args:
        avi_path: Path to input AVI file
        mp4_path: Path to output MP4 file
        crf: Constant Rate Factor (17-20 is high quality, lower = better quality)
        preset: Encoding preset (slow = better quality, faster = faster encoding)
    
    Raises:
        RuntimeError: If ffmpeg is not available or encoding fails
    """
    print(f"\n🎬 Encoding to high-quality MP4 with ffmpeg...")
    print(f"   Input: {avi_path}")
    print(f"   Output: {mp4_path}")
    print(f"   Settings: CRF={crf}, preset={preset}")
    
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(avi_path),
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            str(mp4_path)
        ], check=True, capture_output=True)
        print(f"✅ Encoded HQ MP4: {mp4_path}")
        
        # Clean up intermediate AVI
        if avi_path.exists():
            avi_path.unlink()
            print(f"🗑️  Removed intermediate AVI: {avi_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg encoding failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install it:\n"
            "  macOS: brew install ffmpeg\n"
            "  Linux: sudo apt-get install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/"
        )


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
    padding: int = 0,
    crf: int = 17,
    ffmpeg_preset: str = "slow",
):
    """
    Stitch two fisheye videos using OpenCV Stitcher with a FIXED transform:
    - estimateTransform() once on a reference frame
    - composePanorama() for every frame (stable, no jitter)
    
    Args:
        ref_frame: Frame number to use for estimating transform (default: 120)
        stitch_mode: "PANORAMA" or "SCANS" (SCANS often better for side-by-side fixed cameras)
        padding: Padding size in pixels to add around undistorted frames (default: 0, set >0 to preserve more edges)
        crf: Constant Rate Factor for ffmpeg encoding (17-20 is high quality, default: 17)
        ffmpeg_preset: ffmpeg encoding preset (default: "slow" for best quality)
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
    fps = cap_left.get(cv2.CAP_PROP_FPS)
    frame_count = int(min(
        cap_left.get(cv2.CAP_PROP_FRAME_COUNT),
        cap_right.get(cv2.CAP_PROP_FRAME_COUNT)
    ))

    print(f"📹 Video properties:")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {frame_count}")

    # -------------------------------
    # ✅ 1) Create stitcher ONCE
    # -------------------------------
    mode = cv2.Stitcher_PANORAMA if stitch_mode.upper() == "PANORAMA" else cv2.Stitcher_SCANS
    stitcher = cv2.Stitcher.create(mode)

    # These often help stability for static rigs
    stitcher.setWaveCorrection(False)  # reduces "wobble" in panoramas
    
    # Quality: prevent internal downscale (if available in your OpenCV build)
    try:
        stitcher.setCompositingResol(-1)
        print("✅ Set compositing resolution to full (no downscale)")
    except Exception:
        pass  # Not available in all OpenCV builds
    
    # Stability: reduce seam/exposure flicker
    try:
        # Check if cv2.detail module is available
        if hasattr(cv2, 'detail'):
            # Disable exposure compensation to reduce flicker
            stitcher.setExposureCompensator(
                cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_NO)
            )
            print("✅ Disabled exposure compensation (reduces flicker)")
    except (AttributeError, Exception):
        pass  # May not be available in all OpenCV builds
    
    try:
        # Check if cv2.detail module is available
        if hasattr(cv2, 'detail'):
            # Disable dynamic seam finding to reduce flicker
            stitcher.setSeamFinder(
                cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_NO)
            )
            print("✅ Disabled dynamic seam finding (reduces flicker)")
    except (AttributeError, Exception):
        pass  # May not be available in all OpenCV builds

    # -------------------------------
    # ✅ 2) Estimate transform ONCE on a reference frame
    # -------------------------------
    ref_frame = max(0, min(ref_frame, frame_count - 1))
    print(f"\n🎯 Estimating transform once using reference frame: {ref_frame}")
    print(f"   (Pick a frame where overlap is clear and no player crosses the seam)")

    cap_left.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)

    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()
    if not ret_l or not ret_r:
        raise RuntimeError("Could not read reference frames from both videos.")

    # Undistort frames
    left_u_raw = undistort_fisheye(frame_l, K1, D1, balance=balance, fov_scale=fov_scale)
    right_u_raw = undistort_fisheye(frame_r, K2, D2, balance=balance, fov_scale=fov_scale)
    
    # Debug: print undistorted frame shapes
    print(f"📐 Undistorted frame shapes:")
    print(f"   left_u: {left_u_raw.shape}, right_u: {right_u_raw.shape}")
    
    # Apply padding if specified (default: 0 to reduce flicker)
    if padding > 0:
        left_u = pad(left_u_raw, p=padding)
        right_u = pad(right_u_raw, p=padding)
        print(f"📐 Padded frame shapes: left_u: {left_u.shape}, right_u: {right_u.shape}")
    else:
        left_u = left_u_raw
        right_u = right_u_raw
        print(f"📐 No padding applied (padding=0 to reduce seam flicker)")

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
    # ✅ 3) Reset streams to start and write output
    # -------------------------------
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write high-quality intermediate AVI (will encode to MP4 with ffmpeg later)
    tmp_avi = output_path.with_suffix(".avi")
    out = make_writer(tmp_avi, fps, (out_w, out_h))

    last_successful_frame = pano0.copy()
    success_count = 0
    fail_count = 0
    frame_num = 0

    print("\nProcessing frames with fixed transform...")

    while True:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        if not ret_l or not ret_r:
            break

        # Undistort frames (same as ref frame)
        left_u_raw = undistort_fisheye(frame_l, K1, D1, balance=balance, fov_scale=fov_scale)
        right_u_raw = undistort_fisheye(frame_r, K2, D2, balance=balance, fov_scale=fov_scale)
        
        # Apply padding if specified
        if padding > 0:
            left_u = pad(left_u_raw, p=padding)
            right_u = pad(right_u_raw, p=padding)
        else:
            left_u = left_u_raw
            right_u = right_u_raw

        status, panorama = stitcher.composePanorama([left_u, right_u])

        if status == cv2.Stitcher_OK:
            if panorama.shape[:2] != (out_h, out_w):
                panorama = cv2.resize(panorama, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            out.write(panorama)
            last_successful_frame = panorama.copy()
            success_count += 1
        else:
            # rare fallback
            out.write(last_successful_frame)
            fail_count += 1
            if fail_count <= 5:
                print(f"   ⚠️ Frame {frame_num} composePanorama failed (status: {status}), using previous frame")

        frame_num += 1
        if frame_num % 60 == 0:
            progress = (frame_num / frame_count) * 100
            print(f"   Processed {frame_num}/{frame_count} ({progress:.1f}%) - "
                  f"Success: {success_count}, Failed: {fail_count}")

    # Cleanup video capture and writer
    cap_left.release()
    cap_right.release()
    out.release()

    print(f"\n" + "=" * 70)
    print(f"✅ Video stitching complete!")
    print(f"   Intermediate AVI: {tmp_avi}")
    print(f"   Total frames: {frame_num}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {fail_count}")
    if frame_num > 0:
        print(f"   Success rate: {(success_count/frame_num)*100:.1f}%")
    print(f"=" * 70)
    
    # Encode to high-quality MP4 with ffmpeg
    encode_with_ffmpeg(tmp_avi, output_path, crf=crf, preset=ffmpeg_preset)
    
    print(f"\n" + "=" * 70)
    print(f"✅ Final output: {output_path}")
    print(f"   Method: Fixed transform (no jitter) + High-quality encoding")
    print(f"=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Stitch fisheye videos using OpenCV Stitcher with FIXED transform (no jitter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This method uses OpenCV Stitcher's two-phase approach:
1. estimateTransform() ONCE on a reference frame (locks the seam/warp)
2. composePanorama() for every frame (reuses the fixed transform)

This eliminates jitter while maintaining OpenCV Stitcher's excellent quality.

IMPROVEMENTS:
- High-quality intermediate AVI (MJPG) + ffmpeg encoding for best quality
- Stitcher settings to reduce seam/exposure flicker
- Optional padding (disabled by default to reduce flicker)

Examples:
  # Use default video paths and reference frame 120
  python stitch_fisheye_final.py

  # Custom paths and reference frame
  python stitch_fisheye_final.py \\
    --left_video data/cam_left_10min.mp4 \\
    --right_video data/cam_right_10min.mp4 \\
    --left_calib data/calibration_new/left_fisheye_calib.npz \\
    --right_calib data/calibration_new/right_fisheye_calib.npz \\
    --out data/output/stitched_video_final.mp4 \\
    --balance 0.8 \\
    --fov_scale 1.0 \\
    --ref_frame 120 \\
    --stitch_mode PANORAMA \\
    --padding 0 \\
    --crf 17 \\
    --ffmpeg_preset slow

  # Try SCANS mode with padding to preserve more edges
  python stitch_fisheye_final.py \\
    --stitch_mode SCANS \\
    --ref_frame 60 \\
    --padding 200 \\
    --crf 18
        """
    )
    
    parser.add_argument(
        "--left_video",
        type=Path,
        default=Path("data/cam_left_10min.mp4"),
        help="Path to left fisheye video (default: data/cam_left_10min.mp4)"
    )
    
    parser.add_argument(
        "--right_video",
        type=Path,
        default=Path("data/cam_right_10min.mp4"),
        help="Path to right fisheye video (default: data/cam_right_10min.mp4)"
    )
    
    parser.add_argument(
        "--left_calib",
        type=Path,
        default=Path("data/calibration_new/left_fisheye_calib.npz"),
        help="Path to left camera calibration .npz"
    )
    
    parser.add_argument(
        "--right_calib",
        type=Path,
        default=Path("data/calibration_new/right_fisheye_calib.npz"),
        help="Path to right camera calibration .npz"
    )
    
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/output/stitched_video_final.mp4"),
        help="Output video path"
    )
    
    parser.add_argument(
        "--balance",
        type=float,
        default=0.8,
        help="Fisheye undistort balance (default: 0.8)"
    )
    
    parser.add_argument(
        "--fov_scale",
        type=float,
        default=1.0,
        help="Fisheye FOV scale (default: 1.0)"
    )
    
    parser.add_argument(
        "--ref_frame",
        type=int,
        default=120,
        help="Reference frame number for estimating transform (default: 120). Pick a frame where overlap is clear and no player crosses the seam."
    )
    
    parser.add_argument(
        "--stitch_mode",
        type=str,
        default="PANORAMA",
        choices=["PANORAMA", "SCANS"],
        help="Stitching mode: PANORAMA (default) or SCANS (often better for side-by-side fixed cameras)"
    )
    
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Padding size in pixels to add around undistorted frames (default: 0). Set >0 to preserve more edges, but may increase seam flicker."
    )
    
    parser.add_argument(
        "--crf",
        type=int,
        default=17,
        help="Constant Rate Factor for ffmpeg encoding (default: 17). Lower = higher quality (17-20 is high quality range)."
    )
    
    parser.add_argument(
        "--ffmpeg_preset",
        type=str,
        default="slow",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="ffmpeg encoding preset (default: slow). Slower = better quality but takes longer."
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
    
    # Run stitching
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
        padding=args.padding,
        crf=args.crf,
        ffmpeg_preset=args.ffmpeg_preset
    )


if __name__ == "__main__":
    main()

