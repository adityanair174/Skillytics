import cv2
import numpy as np
from pathlib import Path
import argparse

cv2.ocl.setUseOpenCL(False)

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)

def detect_corners(gray, pattern_size):
    """
    Try the SB detector first (more robust), then fallback.
    pattern_size is (cols, rows) = (inner corners along width, inner corners along height)
    """
    # Try SB if available (OpenCV >= 4.5-ish)
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(
            gray, pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        )
        if ok and corners is not None:
            # corners already good; still refine a bit
            corners = corners.astype(np.float32)
            return True, corners

    # Fallback classic
    ok, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ok or corners is None:
        return False, None

    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), CRITERIA)
    return True, corners

def calibrate_fisheye(image_paths, pattern_size, square_size, debug_dir: Path | None = None, debug_max=20):
    """
    Fisheye calibration using cv2.fisheye.calibrate.
    pattern_size = (cols, rows) inner corners.
    square_size can be 1.0 if you only need undistortion/homography.
    """
    cols, rows = pattern_size
    objp = np.zeros((1, cols * rows, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints = []
    imgpoints = []
    img_shape = None
    debug_written = 0

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (w,h)

        ok, corners = detect_corners(gray, pattern_size)
        if not ok:
            continue

        objpoints.append(objp)
        imgpoints.append(corners)

        if debug_dir is not None and debug_written < debug_max:
            debug_dir.mkdir(parents=True, exist_ok=True)
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, ok)
            cv2.imwrite(str(debug_dir / f"det_{debug_written:03d}_{p.name}"), vis)
            debug_written += 1

    return objpoints, imgpoints, img_shape

def run_calibration(objpoints, imgpoints, img_shape):
    if img_shape is None:
        raise RuntimeError("No images read successfully.")
    if len(objpoints) < 12:
        raise RuntimeError(f"Not enough valid chessboard detections: {len(objpoints)} (need ~12+; 20–40 is best)")

    N = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_CHECK_COND |
             cv2.fisheye.CALIB_FIX_SKEW)

    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, img_shape, K, D, rvecs, tvecs, flags, CRITERIA
    )
    return K, D, img_shape, rms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left_dir", default="data/calibration_new/left", help="Left chessboard images folder")
    ap.add_argument("--right_dir", default="data/calibration_new/right", help="Right chessboard images folder")
    ap.add_argument("--corners", nargs=2, type=int, default=[8, 6], help="Inner corners as: cols rows (e.g., 8 6)")
    ap.add_argument("--square_size", type=float, default=1.0, help="Square size (units). Can be 1.0 for stitching.")
    ap.add_argument("--out_dir", default="data/calibration_new", help="Where to write calibration .npz files")
    ap.add_argument("--debug", action="store_true", help="Write debug detections with drawn corners")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    left_dir = (project_root / args.left_dir).resolve()
    right_dir = (project_root / args.right_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()

    pattern_size = (int(args.corners[0]), int(args.corners[1]))  # (cols, rows)

    left_imgs = list_images(left_dir)
    right_imgs = list_images(right_dir)

    print(f"LEFT dir:  {left_dir}")
    print(f"RIGHT dir: {right_dir}")
    print(f"Pattern (cols,rows): {pattern_size}")
    print(f"LEFT images: {len(left_imgs)} | RIGHT images: {len(right_imgs)}")

    if not left_imgs or not right_imgs:
        raise RuntimeError("No images found. Check folder paths and extensions (jpg/png/etc).")

    left_debug = out_dir / "debug_left" if args.debug else None
    right_debug = out_dir / "debug_right" if args.debug else None

    print("\nDetecting LEFT corners...")
    objL, imgL, sizeL = calibrate_fisheye(left_imgs, pattern_size, args.square_size, left_debug)
    print(f"LEFT detections: {len(objL)}")

    print("\nDetecting RIGHT corners...")
    objR, imgR, sizeR = calibrate_fisheye(right_imgs, pattern_size, args.square_size, right_debug)
    print(f"RIGHT detections: {len(objR)}")

    print("\nCalibrating LEFT (fisheye)...")
    K1, D1, sizeL, rms1 = run_calibration(objL, imgL, sizeL)
    print(f"LEFT size={sizeL} RMS={rms1:.4f}")
    print("K1=\n", K1)
    print("D1=", D1.ravel())

    print("\nCalibrating RIGHT (fisheye)...")
    K2, D2, sizeR, rms2 = run_calibration(objR, imgR, sizeR)
    print(f"RIGHT size={sizeR} RMS={rms2:.4f}")
    print("K2=\n", K2)
    print("D2=", D2.ravel())

    out_dir.mkdir(parents=True, exist_ok=True)
    left_out = out_dir / "left_fisheye_calib.npz"
    right_out = out_dir / "right_fisheye_calib.npz"

    np.savez(left_out, K=K1, D=D1, image_size=np.array(sizeL))
    np.savez(right_out, K=K2, D=D2, image_size=np.array(sizeR))

    print(f"\n✅ Saved:\n  {left_out}\n  {right_out}")

if __name__ == "__main__":
    main()
