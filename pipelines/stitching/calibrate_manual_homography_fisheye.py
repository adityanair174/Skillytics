#!/usr/bin/env python3
import os
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import argparse
from pathlib import Path
import numpy as np
import cv2

cv2.ocl.setUseOpenCL(False)

# -------------------------
# Helpers
# -------------------------
def load_fisheye_calib(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)

    def pick(keys):
        for k in keys:
            if k in data:
                return data[k]
        return None

    K = pick(["K", "camera_matrix", "mtx", "K1", "K2"])
    D = pick(["D", "dist", "dist_coeffs", "distCoeffs", "D1", "D2"])

    if K is None or D is None:
        raise ValueError(f"Could not find K/D in {npz_path}. Keys present: {list(data.keys())}")

    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64).reshape(-1)[:4]
    if D.size < 4:
        raise ValueError(f"Distortion must have 4 coeffs for fisheye. Got {D.size}")
    return K, D

def build_undistort_maps(K, D, dim, balance, fov_scale):
    w, h = dim
    R = np.eye(3)
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), R, balance=balance, new_size=(w, h), fov_scale=fov_scale
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, newK, (w, h), m1type=cv2.CV_16SC2
    )
    return map1, map2

def undistort(img, map1, map2):
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def compute_point_errors(H, ptsR, ptsL):
    proj = cv2.perspectiveTransform(ptsR, H)
    err = np.linalg.norm(proj.reshape(-1, 2) - ptsL.reshape(-1, 2), axis=1)
    return err

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left_img", required=True)
    ap.add_argument("--right_img", required=True)
    ap.add_argument("--left_calib", required=True)
    ap.add_argument("--right_calib", required=True)
    ap.add_argument("--balance", type=float, default=0.35)
    ap.add_argument("--fov_scale", type=float, default=1.0)
    ap.add_argument("--ransac_px", type=float, default=5.0)
    ap.add_argument("--out_H", required=True, help="Output .npy homography (RIGHT->LEFT) in UNDISTORTED space")
    ap.add_argument("--max_disp_w", type=int, default=1700)
    ap.add_argument("--max_disp_h", type=int, default=900)
    args = ap.parse_args()

    out_H = Path(args.out_H)
    out_H.parent.mkdir(parents=True, exist_ok=True)

    L = cv2.imread(args.left_img, cv2.IMREAD_COLOR)
    R = cv2.imread(args.right_img, cv2.IMREAD_COLOR)
    if L is None or R is None:
        raise FileNotFoundError("Could not read one of the images. Check paths.")

    h, w = L.shape[:2]
    if R.shape[:2] != (h, w):
        R = cv2.resize(R, (w, h), interpolation=cv2.INTER_LINEAR)

    K1, D1 = load_fisheye_calib(args.left_calib)
    K2, D2 = load_fisheye_calib(args.right_calib)

    map1L, map2L = build_undistort_maps(K1, D1, (w, h), args.balance, args.fov_scale)
    map1R, map2R = build_undistort_maps(K2, D2, (w, h), args.balance, args.fov_scale)

    Lu = undistort(L, map1L, map2L)
    Ru = undistort(R, map1R, map2R)

    # ---- display scaling ----
    disp_scale = min(args.max_disp_h / h, args.max_disp_w / (2 * w), 1.0)
    disp_w = int(w * disp_scale)
    disp_h = int(h * disp_scale)

    left_s = cv2.resize(Lu, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    right_s = cv2.resize(Ru, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
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

        # ✅ HARD ENFORCEMENT:
        # - You can only add a LEFT point when L_count == R_count
        # - You can only add a RIGHT point when R_count == L_count - 1
        if clicked_left:
            if len(ptsL) != len(ptsR):
                return
            ox = x / disp_scale
            oy = y / disp_scale
            ptsL.append((ox, oy))
            last_click_side = "L"
            expecting_left = False
            print(f"LEFT  {len(ptsL)}: ({int(ox)}, {int(oy)})")
        else:
            if len(ptsR) != len(ptsL) - 1:
                return
            ox = (x - disp_w) / disp_scale
            oy = y / disp_scale
            ptsR.append((ox, oy))
            last_click_side = "R"
            expecting_left = True
            print(f"RIGHT {len(ptsR)}: ({int(ox)}, {int(oy)})")

        redraw()

    cv2.setMouseCallback(win, on_mouse)
    redraw()

    print("\nTIP: Pick 12–25 points in the overlap. Use cones + any distinct turf corners. Avoid net/sky/fence.\n")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13):  # ENTER
            if len(ptsL) >= 8 and len(ptsL) == len(ptsR):
                break
            print("Need >=8 pairs and counts must match.")
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

    method = cv2.RANSAC
    if hasattr(cv2, "USAC_MAGSAC"):
        method = cv2.USAC_MAGSAC

    H, inliers = cv2.findHomography(ptsR_np, ptsL_np, method, ransacReprojThreshold=float(args.ransac_px))
    if H is None:
        raise RuntimeError("findHomography failed. Re-pick better overlap points.")

    inliers = inliers.reshape(-1).astype(bool)
    err = compute_point_errors(H, ptsR_np, ptsL_np)

    n_in = int(inliers.sum())
    print(f"\n✅ Inliers: {n_in}/{len(inliers)}  (ransac_px={args.ransac_px})")
    print(f"ALL pts: mean={float(err.mean()):.2f}px max={float(err.max()):.2f}px")
    print(f"INLIER: mean={float(err[inliers].mean()):.2f}px max={float(err[inliers].max()):.2f}px")

    # Save H + points
    np.save(str(out_H), H)
    np.save(str(out_H.with_suffix(".left_pts.npy")), ptsL_np.reshape(-1, 2))
    np.save(str(out_H.with_suffix(".right_pts.npy")), ptsR_np.reshape(-1, 2))
    np.save(str(out_H.with_suffix(".inliers.npy")), inliers.astype(np.uint8))
    print(f"\n✅ Saved H: {out_H}")
    print(f"✅ Saved pts: {out_H.with_suffix('.left_pts.npy')} / {out_H.with_suffix('.right_pts.npy')}")

    # Quick overlay preview in LEFT frame space
    warpR = cv2.warpPerspective(Ru, H, (w, h))
    overlay = cv2.addWeighted(Lu, 0.5, warpR, 0.5, 0)
    diff = cv2.absdiff(Lu, warpR)

    overlay_path = out_H.with_suffix(".overlay.jpg")
    diff_path = out_H.with_suffix(".diff.jpg")
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(diff_path), diff)
    print(f"✅ Overlay: {overlay_path}")
    print(f"✅ Diff:    {diff_path}")

if __name__ == "__main__":
    main()
