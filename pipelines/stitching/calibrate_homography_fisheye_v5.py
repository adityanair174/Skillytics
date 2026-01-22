#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import cv2
import numpy as np

# --- Avoid OpenCL cache crash on some macOS setups ---
try:
    cv2.ocl.setUseOpenCL(False)
    os.environ["OPENCV_OPENCL_CACHE_ENABLE"] = "0"
except Exception:
    pass


def _load_fisheye_calib(npz_path: str):
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
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    if D.size < 4:
        raise ValueError(f"Distortion must have 4 coeffs for fisheye. Got {D.size}")
    D = D[:4]
    return K, D


def undistort_fisheye(img, K, D, balance: float, fov_scale: float):
    h, w = img.shape[:2]
    dim = (w, h)
    R = np.eye(3)

    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, dim, R, balance=balance, new_size=dim, fov_scale=fov_scale
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, newK, dim, m1type=cv2.CV_16SC2
    )
    und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return und


def make_turf_mask(bgr):
    """
    Rough turf mask: HSV green range + morphology.
    This is intentionally "good enough" to avoid blending sky/net/fence.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = (25, 25, 25)
    upper = (95, 255, 255)
    m = cv2.inRange(hsv, lower, upper)

    # clean up
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    # also remove pure black borders from undistort
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    valid = (gray > 0).astype(np.uint8) * 255
    m = cv2.bitwise_and(m, valid)
    return m


def compute_panorama_canvas(left_img, right_img, H_right_to_left):
    """
    Canvas that fits left + warped right, with a safety guard.
    """
    hL, wL = left_img.shape[:2]
    hR, wR = right_img.shape[:2]

    cornersR = np.array([[0, 0], [wR - 1, 0], [wR - 1, hR - 1], [0, hR - 1]], dtype=np.float32).reshape(-1, 1, 2)
    cornersR_w = cv2.perspectiveTransform(cornersR, H_right_to_left)

    cornersL = np.array([[0, 0], [wL - 1, 0], [wL - 1, hL - 1], [0, hL - 1]], dtype=np.float32).reshape(-1, 1, 2)

    all_pts = np.vstack([cornersL, cornersR_w]).reshape(-1, 2)
    min_xy = np.floor(all_pts.min(axis=0)).astype(int)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(int)

    tx = -min_xy[0] if min_xy[0] < 0 else 0
    ty = -min_xy[1] if min_xy[1] < 0 else 0

    canvas_w = int(max_xy[0] + tx + 1)
    canvas_h = int(max_xy[1] + ty + 1)

    # safety guard
    MAX_SIDE = 20000
    MAX_PIXELS = 120_000_000
    if (canvas_w <= 0 or canvas_h <= 0 or canvas_w > MAX_SIDE or canvas_h > MAX_SIDE or canvas_w * canvas_h > MAX_PIXELS):
        margin_x = wL // 2
        margin_y = hL // 10
        canvas_w = wL + wR + 2 * margin_x
        canvas_h = max(hL, hR) + 2 * margin_y
        tx = margin_x
        ty = margin_y
        print(f"⚠️ Auto canvas exploded; using fixed canvas {canvas_w}x{canvas_h} (tx={tx}, ty={ty})")

    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    print(f"Canvas: {canvas_w}x{canvas_h}  tx={tx} ty={ty}")
    return canvas_w, canvas_h, T


def compute_point_errors(H, ptsR_np, ptsL_np):
    proj = cv2.perspectiveTransform(ptsR_np, H).reshape(-1, 2)
    tgt = ptsL_np.reshape(-1, 2)
    err = np.linalg.norm(proj - tgt, axis=1)
    return err


def try_refine_ecc(left_bgr, right_bgr, H_init, mask_template_u8, iters=1200, eps=1e-6):
    """
    ECC refines H so that warp(right, H) aligns to left.
    Returns (cc, H_refined) or (None, H_init) if failed.
    """
    left_g = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    right_g = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    warp = H_init.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps))
    try:
        cc, warp = cv2.findTransformECC(
            templateImage=left_g,
            inputImage=right_g,
            warpMatrix=warp,
            motionType=cv2.MOTION_HOMOGRAPHY,
            criteria=criteria,
            inputMask=mask_template_u8,
            gaussFiltSize=5,
        )
        return float(cc), warp.astype(np.float64)
    except cv2.error:
        return None, H_init


def exposure_gain_on_turf_overlap(left_u, right_u, H_right_to_left, turf_mask_left_u8):
    """
    Compute a single gain for RIGHT so that overlap turf brightness matches LEFT.
    """
    h, w = left_u.shape[:2]
    warpR = cv2.warpPerspective(right_u, H_right_to_left, (w, h))
    grayL = cv2.cvtColor(left_u, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(warpR, cv2.COLOR_BGR2GRAY)

    overlap = (grayL > 0) & (grayR > 0) & (turf_mask_left_u8 > 0)
    if overlap.sum() < 2000:
        return 1.0

    labL = cv2.cvtColor(left_u, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    labR = cv2.cvtColor(warpR, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    mL = float(labL[overlap].mean())
    mR = float(labR[overlap].mean())
    gain = mL / (mR + 1e-6)
    gain = float(np.clip(gain, 0.6, 1.6))
    return gain


def seam_blend_turf(base, warpR, turfL_u8, turfR_u8, feather_band=70):
    """
    Blend ONLY inside turf overlap using a seam (hard cut + small feather band).
    Outside turf: keep base (no ghosting from net/fence).
    """
    H, W = base.shape[:2]
    turfL = (turfL_u8 > 0)
    turfR = (turfR_u8 > 0)

    out = base.copy()

    # regions where only right turf exists
    onlyR = turfR & (~turfL)
    out[onlyR] = warpR[onlyR]

    overlap = turfL & turfR
    if overlap.sum() < 1000:
        return out, None

    ys, xs = np.where(overlap)
    seam_x = int(np.median(xs))

    xgrid = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    d = xgrid - float(seam_x)

    # weight for right: 0 (left) -> 1 (right) across a small band around seam
    wR = np.zeros((H, W), dtype=np.float32)
    wR[overlap] = np.clip((d[overlap] + feather_band) / (2.0 * feather_band), 0.0, 1.0)

    # blend only on overlap turf
    base_f = base.astype(np.float32)
    warp_f = warpR.astype(np.float32)
    wR3 = wR[:, :, None]

    blended = (base_f * (1.0 - wR3) + warp_f * wR3).astype(np.uint8)
    out[overlap] = blended[overlap]

    # debug seam mask image
    seam_vis = np.zeros((H, W, 3), dtype=np.uint8)
    seam_vis[overlap] = (0, 0, 255)
    cv2.line(seam_vis, (seam_x, 0), (seam_x, H - 1), (255, 255, 255), 2)

    return out, seam_vis


def crop_to_content(img, mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 10:
        return img
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return img[y0:y1 + 1, x0:x1 + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left_img", required=True)
    ap.add_argument("--right_img", required=True)
    ap.add_argument("--left_calib", required=True)
    ap.add_argument("--right_calib", required=True)

    ap.add_argument("--balance", type=float, default=0.35)
    ap.add_argument("--fov_scale", type=float, default=1.0)
    ap.add_argument("--ransac_px", type=float, default=5.0)

    ap.add_argument("--do_exposure", action="store_true")
    ap.add_argument("--do_ecc", action="store_true")
    ap.add_argument("--ecc_min_cc", type=float, default=0.35)

    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--out_H", required=True)
    ap.add_argument("--max_disp_w", type=int, default=1700)
    ap.add_argument("--max_disp_h", type=int, default=900)

    args = ap.parse_args()

    out_H = Path(args.out_H)
    out_H.parent.mkdir(parents=True, exist_ok=True)

    left_bgr = cv2.imread(args.left_img, cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(args.right_img, cv2.IMREAD_COLOR)
    if left_bgr is None or right_bgr is None:
        raise FileNotFoundError("Could not read one of the images. Check paths.")

    K1, D1 = _load_fisheye_calib(args.left_calib)
    K2, D2 = _load_fisheye_calib(args.right_calib)

    left_u = undistort_fisheye(left_bgr, K1, D1, args.balance, args.fov_scale)
    right_u = undistort_fisheye(right_bgr, K2, D2, args.balance, args.fov_scale)

    h, w = left_u.shape[:2]
    if right_u.shape[:2] != (h, w):
        right_u = cv2.resize(right_u, (w, h), interpolation=cv2.INTER_LINEAR)

    # --- point picker display ---
    disp_scale = min(args.max_disp_h / h, args.max_disp_w / (2 * w), 1.0)
    disp_w = int(w * disp_scale)
    disp_h = int(h * disp_scale)

    left_s = cv2.resize(left_u, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    right_s = cv2.resize(right_u, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    base_disp = np.hstack([left_s, right_s])
    win = "Pick TURF points: LEFT then RIGHT (u=undo, r=reset, ENTER=done, q=quit)"
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

    # More robust than classic RANSAC if available
    method = cv2.RANSAC
    if hasattr(cv2, "USAC_MAGSAC"):
        method = cv2.USAC_MAGSAC

    H, inliers = cv2.findHomography(ptsR_np, ptsL_np, method, ransacReprojThreshold=args.ransac_px)
    if H is None:
        raise RuntimeError("findHomography failed. Re-pick better turf-only overlap points.")

    inliers = inliers.reshape(-1).astype(bool)
    err = compute_point_errors(H, ptsR_np, ptsL_np)

    n_in = int(inliers.sum())
    print(f"Inliers: {n_in}/{len(inliers)}  (ransac_px={args.ransac_px})")
    print(f"ALL points:    Mean={float(err.mean()):.2f}px  Max={float(err.max()):.2f}px")
    if n_in > 0:
        print(f"INLIERS only:  Mean={float(err[inliers].mean()):.2f}px  Max={float(err[inliers].max()):.2f}px")

    # Refit using only inliers (stabilizes)
    if n_in >= 4:
        H_ls, _ = cv2.findHomography(ptsR_np[inliers], ptsL_np[inliers], 0)
        if H_ls is not None:
            H = H_ls

    # Optional exposure match (turf overlap only)
    turfL = make_turf_mask(left_u)
    if args.do_exposure:
        gain = exposure_gain_on_turf_overlap(left_u, right_u, H, turfL)
        right_u = np.clip(right_u.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        print(f"Exposure gain applied to RIGHT: {gain:.3f}")

    # Optional ECC refine, but only keep if it helps
    if args.do_ecc:
        # ECC mask in LEFT coords: turf only
        ecc_mask = turfL.copy()
        cc, H_ecc = try_refine_ecc(left_u, right_u, H, ecc_mask, iters=1200, eps=1e-6)
        if cc is None:
            print("ECC refinement failed.")
        else:
            err_before = compute_point_errors(H, ptsR_np, ptsL_np)
            err_after = compute_point_errors(H_ecc, ptsR_np, ptsL_np)
            before = float(err_before[inliers].mean()) if n_in > 0 else float(err_before.mean())
            after = float(err_after[inliers].mean()) if n_in > 0 else float(err_after.mean())
            print(f"ECC refinement success. cc={cc:.6f}  inlier_mean_err: {before:.3f}px -> {after:.3f}px")

            if cc >= args.ecc_min_cc and after < before:
                H = H_ecc
            else:
                print("⚠️ ECC rejected (low cc or did not improve error). Keeping RANSAC/LS H.")

    np.save(str(out_H), H)
    print(f"✅ Saved H (right->left): {out_H}")

    # --- previews ---
    canvas_w, canvas_h, T = compute_panorama_canvas(left_u, right_u, H)
    Ht = T @ H

    base = cv2.warpPerspective(left_u, T, (canvas_w, canvas_h))
    warpR = cv2.warpPerspective(right_u, Ht, (canvas_w, canvas_h))

    # masks (warp proper binary masks, NOT gray(warpR)>0)
    validL = (cv2.cvtColor(left_u, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
    validR = (cv2.cvtColor(right_u, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255

    validL_c = cv2.warpPerspective(validL, T, (canvas_w, canvas_h))
    validR_c = cv2.warpPerspective(validR, Ht, (canvas_w, canvas_h))

    turfL_c = cv2.warpPerspective(turfL, T, (canvas_w, canvas_h))
    turfR = make_turf_mask(right_u)
    turfR_c = cv2.warpPerspective(turfR, Ht, (canvas_w, canvas_h))

    overlay = cv2.addWeighted(base, 0.5, warpR, 0.5, 0)
    diff = cv2.absdiff(base, warpR)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_vis = cv2.applyColorMap(cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_TURBO)

    pano, seam_vis = seam_blend_turf(base, warpR, turfL_c, turfR_c, feather_band=70)

    # optional crop
    if args.crop:
        union = ((validL_c > 0) | (validR_c > 0)).astype(np.uint8) * 255
        overlay = crop_to_content(overlay, union)
        diff_vis = crop_to_content(diff_vis, union)
        pano = crop_to_content(pano, union)
        if seam_vis is not None:
            seam_vis = crop_to_content(seam_vis, union)

    overlay_path = out_H.with_name("homography_overlay_fisheye_v5.jpg")
    diff_path = out_H.with_name("homography_diff_fisheye_v5.jpg")
    pano_path = out_H.with_name("panorama_turf_seam_fisheye_v5.jpg")
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(diff_path), diff_vis)
    cv2.imwrite(str(pano_path), pano)

    print(f"✅ Overlay preview: {overlay_path}")
    print(f"✅ Diff preview:    {diff_path}")
    print(f"✅ Panorama (turf seam): {pano_path}")

    if seam_vis is not None:
        seam_path = out_H.with_name("seam_debug_fisheye_v5.jpg")
        cv2.imwrite(str(seam_path), seam_vis)
        print(f"✅ Seam debug:      {seam_path}")


if __name__ == "__main__":
    main()
