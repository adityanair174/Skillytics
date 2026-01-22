#!/usr/bin/env python3
"""
Calibrate homography between two regular (non-fisheye) cameras.

This script is for standard cameras without fisheye distortion.
For fisheye cameras, use calibrate_homography_fisheye_v5.py instead.

Interactive point picking to compute homography matrix that transforms
points from right camera coordinate system to left camera coordinate system.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2


def compute_point_errors(H, ptsR_np, ptsL_np):
    """Compute reprojection errors for homography."""
    proj = cv2.perspectiveTransform(ptsR_np, H).reshape(-1, 2)
    tgt = ptsL_np.reshape(-1, 2)
    err = np.linalg.norm(proj - tgt, axis=1)
    return err


def main():
    ap = argparse.ArgumentParser(
        description="Calibrate homography between two regular (non-fisheye) cameras"
    )
    ap.add_argument("--left_img", required=True, help="Path to left camera image")
    ap.add_argument("--right_img", required=True, help="Path to right camera image")
    ap.add_argument("--out_H", required=True, help="Output path for homography .npy file")
    ap.add_argument("--ransac_px", type=float, default=5.0, help="RANSAC reprojection threshold in pixels")
    ap.add_argument("--max_disp_w", type=int, default=1700, help="Max display width")
    ap.add_argument("--max_disp_h", type=int, default=900, help="Max display height")

    args = ap.parse_args()

    out_H = Path(args.out_H)
    out_H.parent.mkdir(parents=True, exist_ok=True)

    left_bgr = cv2.imread(args.left_img, cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(args.right_img, cv2.IMREAD_COLOR)
    if left_bgr is None or right_bgr is None:
        raise FileNotFoundError("Could not read one of the images. Check paths.")

    h, w = left_bgr.shape[:2]
    if right_bgr.shape[:2] != (h, w):
        right_bgr = cv2.resize(right_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    # Display setup
    disp_scale = min(args.max_disp_h / h, args.max_disp_w / (2 * w), 1.0)
    disp_w = int(w * disp_scale)
    disp_h = int(h * disp_scale)

    left_s = cv2.resize(left_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    right_s = cv2.resize(right_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    base_disp = np.hstack([left_s, right_s])
    win = "Pick points: LEFT then RIGHT (u=undo, r=reset, ENTER=done, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(args.max_disp_w, base_disp.shape[1]), min(args.max_disp_h, base_disp.shape[0]))

    ptsL, ptsR = [], []
    expecting_left = True
    last_click_side = None

    def redraw():
        disp = base_disp.copy()
        n = max(len(ptsL), len(ptsR))
        for i in range(n):
            if i < len(ptsL):
                x, y = ptsL[i]
                xs, ys = int(x * disp_scale), int(y * disp_scale)
                cv2.circle(disp, (xs, ys), 5, (0, 255, 0), -1)
                cv2.putText(disp, str(i + 1), (xs + 8, ys - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if i < len(ptsR):
                x, y = ptsR[i]
                xs, ys = int(x * disp_scale) + disp_w, int(y * disp_scale)
                cv2.circle(disp, (xs, ys), 5, (0, 255, 0), -1)
                cv2.putText(disp, str(i + 1), (xs + 8, ys - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        status = f"Pairs: {min(len(ptsL), len(ptsR))} | Next: {'LEFT' if expecting_left else 'RIGHT'}"
        cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(win, disp)

    def on_mouse(event, x, y, flags, param):
        nonlocal expecting_left, last_click_side
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < 0 or y < 0 or x >= 2 * disp_w or y >= disp_h:
            return

        clicked_left = x < disp_w
        if expecting_left and not clicked_left:
            return
        if (not expecting_left) and clicked_left:
            return

        if clicked_left:
            ox = x / disp_scale
            oy = y / disp_scale
            ptsL.append((ox, oy))
            last_click_side = "L"
            expecting_left = False
            print(f"LEFT point {len(ptsL)}: ({int(ox)}, {int(oy)})")
        else:
            ox = (x - disp_w) / disp_scale
            oy = y / disp_scale
            ptsR.append((ox, oy))
            last_click_side = "R"
            expecting_left = True
            print(f"RIGHT point {len(ptsR)}: ({int(ox)}, {int(oy)})")

        redraw()

    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13):  # ENTER
            if len(ptsL) >= 4 and len(ptsL) == len(ptsR):
                break
            print("Need >=4 pairs and counts must match.")
        if key in (27, ord("q")):
            print("Quit.")
            cv2.destroyAllWindows()
            return
        if key == ord("u"):
            if last_click_side == "R" and len(ptsR) > 0:
                ptsR.pop()
                expecting_left = False
                last_click_side = "L"
            elif last_click_side == "L" and len(ptsL) > 0:
                ptsL.pop()
                expecting_left = True
                last_click_side = "R"
            redraw()
        if key == ord("r"):
            ptsL.clear()
            ptsR.clear()
            expecting_left = True
            last_click_side = None
            redraw()

    cv2.destroyAllWindows()

    ptsL_np = np.array(ptsL, dtype=np.float32).reshape(-1, 1, 2)
    ptsR_np = np.array(ptsR, dtype=np.float32).reshape(-1, 1, 2)

    # Compute homography
    method = cv2.RANSAC
    if hasattr(cv2, "USAC_MAGSAC"):
        method = cv2.USAC_MAGSAC

    H, inliers = cv2.findHomography(ptsR_np, ptsL_np, method, ransacReprojThreshold=args.ransac_px)
    if H is None:
        raise RuntimeError("findHomography failed. Re-pick better corresponding points.")

    inliers = inliers.reshape(-1).astype(bool)
    err = compute_point_errors(H, ptsR_np, ptsL_np)

    n_in = int(inliers.sum())
    print(f"Inliers: {n_in}/{len(inliers)}  (ransac_px={args.ransac_px})")
    print(f"ALL points:    Mean={float(err.mean()):.2f}px  Max={float(err.max()):.2f}px")
    if n_in > 0:
        print(f"INLIERS only:  Mean={float(err[inliers].mean()):.2f}px  Max={float(err[inliers].max()):.2f}px")

    # Refit using only inliers
    if n_in >= 4:
        H_ls, _ = cv2.findHomography(ptsR_np[inliers], ptsL_np[inliers], 0)
        if H_ls is not None:
            H = H_ls

    np.save(str(out_H), H)
    print(f"✅ Saved H (right->left): {out_H}")


if __name__ == "__main__":
    main()

