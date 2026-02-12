#!/usr/bin/env python3
"""
Offline cross-camera association from COCO JSON detections.

Workflow:
1) Run per-camera tracking (ByteTrack) on JSON boxes to create local track IDs.
2) Undistort bbox bottom-center points using fisheye calibration.
3) Project right camera points into left image plane via homography.
4) Associate tracks across cameras using spatial proximity and evidence accumulation.
5) Write COCO JSON with track_id + global_id, plus an association report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Avoid matplotlib/fontconfig cache permission warnings when supervision pulls matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


# -------------------------
# Calibration utilities
# -------------------------

def _lower_key_map(keys):
    return {k.lower(): k for k in keys}


def _pick_key(data: np.lib.npyio.NpzFile, key_map: Dict[str, str], *names):
    for name in names:
        if name in key_map:
            return data[key_map[name]]
    return None


def load_fisheye_calib(npz_path: str, fallback_dim: Tuple[int, int] | None = None):
    data = np.load(npz_path)
    key_map = _lower_key_map(data.files)

    K = _pick_key(data, key_map, "k", "camera_matrix", "mtx")
    D = _pick_key(data, key_map, "d", "dist_coeffs", "dist", "distortion")
    dim = _pick_key(data, key_map, "dim", "image_size", "size", "resolution")

    if K is None or D is None:
        raise ValueError(f"Missing K/D in calibration file: {npz_path} (keys={data.files})")

    K = np.array(K, dtype=np.float64)
    D = np.array(D, dtype=np.float64).reshape(-1, 1)

    if dim is None:
        if fallback_dim is None:
            raise ValueError(f"Missing DIM in calibration file: {npz_path}")
        dim = np.array(fallback_dim, dtype=np.int32)
    else:
        dim = np.array(dim).astype(np.int32).reshape(-1)

    if dim.size != 2:
        raise ValueError(f"Invalid DIM shape in calibration file: {npz_path} -> {dim}")

    # OpenCV expects (width, height)
    dim = (int(dim[0]), int(dim[1]))

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, dim, np.eye(3), balance=0.0, new_size=dim
    )

    return K, D, dim, new_K


def undistort_points(points: np.ndarray, K: np.ndarray, D: np.ndarray, new_K: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    und = cv2.fisheye.undistortPoints(pts, K, D, P=new_K)
    return und.reshape(-1, 2)


def project_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H.astype(np.float32))
    return dst.reshape(-1, 2)


# -------------------------
# COCO utilities
# -------------------------

FRAME_RE = re.compile(r"_frame_(\d+)")


def parse_frame_number(file_name: str) -> int | None:
    m = FRAME_RE.search(file_name)
    if not m:
        return None
    return int(m.group(1))


def build_frame_index(data: dict) -> Tuple[List[int], Dict[int, List[int]], Dict[int, Tuple[int, int]]]:
    images = data["images"]
    annotations = data["annotations"]

    # Map image_id -> annotation indices
    ann_by_image_id: Dict[int, List[int]] = defaultdict(list)
    for idx, ann in enumerate(annotations):
        ann_by_image_id[int(ann["image_id"])].append(idx)

    image_sizes = {}
    for img in images:
        image_sizes[int(img["id"])] = (int(img["width"]), int(img["height"]))

    image_ids = sorted([int(img["id"]) for img in images])

    return image_ids, ann_by_image_id, image_sizes


def build_frame_index_by_filename(data: dict) -> Tuple[List[int], Dict[int, List[int]], Dict[int, Tuple[int, int]]]:
    images = data["images"]
    annotations = data["annotations"]

    # Map frame_number -> image_id
    frame_to_image_id = {}
    image_sizes = {}
    for img in images:
        frame_num = parse_frame_number(img["file_name"])
        if frame_num is not None:
            frame_to_image_id[frame_num] = int(img["id"])
        image_sizes[int(img["id"])] = (int(img["width"]), int(img["height"]))

    ann_by_image_id: Dict[int, List[int]] = defaultdict(list)
    for idx, ann in enumerate(annotations):
        ann_by_image_id[int(ann["image_id"])].append(idx)

    frame_numbers = sorted(frame_to_image_id.keys())
    image_ids = [frame_to_image_id[f] for f in frame_numbers]

    return image_ids, ann_by_image_id, image_sizes


# -------------------------
# Tracking
# -------------------------

@dataclass
class TrackParams:
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 30


class IOUTracker:
    """Simple greedy IoU tracker (fallback)."""

    def __init__(self, iou_thresh=0.3, max_age=30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}  # id -> (bbox, age)

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        # Age existing tracks
        for tid in list(self.tracks.keys()):
            bbox, age = self.tracks[tid]
            self.tracks[tid] = (bbox, age + 1)
            if self.tracks[tid][1] > self.max_age:
                del self.tracks[tid]

        assigned = [-1] * len(boxes)
        used_tracks = set()

        for i, box in enumerate(boxes):
            best_iou = 0.0
            best_tid = None
            for tid, (tb, _) in self.tracks.items():
                if tid in used_tracks:
                    continue
                iou = self._iou(box, tb)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None and best_iou >= self.iou_thresh:
                assigned[i] = best_tid
                used_tracks.add(best_tid)
                self.tracks[best_tid] = (box, 0)
            else:
                tid = self.next_id
                self.next_id += 1
                assigned[i] = tid
                self.tracks[tid] = (box, 0)

        return assigned


class ByteTrackWrapper:
    def __init__(self, params: TrackParams):
        try:
            import supervision as sv
            import inspect
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "ByteTrack requires 'supervision'. Install with: pip install supervision scipy"
            ) from e

        if hasattr(sv, "ByteTrack"):
            self._sv = sv
            sig = inspect.signature(sv.ByteTrack.__init__)
            kwargs = {}
            if "track_thresh" in sig.parameters:
                kwargs["track_thresh"] = params.track_thresh
            if "track_activation_threshold" in sig.parameters:
                kwargs["track_activation_threshold"] = params.track_thresh
            if "track_buffer" in sig.parameters:
                kwargs["track_buffer"] = params.track_buffer
            if "lost_track_buffer" in sig.parameters:
                kwargs["lost_track_buffer"] = params.track_buffer
            if "match_thresh" in sig.parameters:
                kwargs["match_thresh"] = params.match_thresh
            if "minimum_matching_threshold" in sig.parameters:
                kwargs["minimum_matching_threshold"] = params.match_thresh
            if "frame_rate" in sig.parameters:
                kwargs["frame_rate"] = params.frame_rate
            if "minimum_consecutive_frames" in sig.parameters:
                kwargs["minimum_consecutive_frames"] = 1

            self.tracker = sv.ByteTrack(**kwargs)
            self._mode = "sv"
        else:
            raise RuntimeError(
                "supervision.ByteTrack not found. Please upgrade supervision."
            )

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        if len(boxes) == 0:
            return []

        dets = self._sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32),
            confidence=np.asarray(scores, dtype=np.float32),
            class_id=np.zeros((len(boxes),), dtype=np.int32),
        )

        if hasattr(self.tracker, "update_with_detections"):
            tracked = self.tracker.update_with_detections(dets)
        else:
            tracked = self.tracker.update(dets)

        # tracker_id should align with detections length
        tracker_ids = tracked.tracker_id
        if tracker_ids is None:
            return [-1] * len(boxes)

        tracker_ids = tracker_ids.tolist()
        if len(tracker_ids) != len(boxes):
            # Fall back to best-effort mapping
            return (tracker_ids + [-1] * len(boxes))[: len(boxes)]

        return [int(tid) if tid is not None else -1 for tid in tracker_ids]


# -------------------------
# Association
# -------------------------

class DSU:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# -------------------------
# Main logic
# -------------------------

def get_video_fps(path: str) -> float | None:
    if not path or not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else None


def run_tracking(
    data: dict,
    image_ids: List[int],
    ann_by_image_id: Dict[int, List[int]],
    track_params: TrackParams,
    tracker_name: str,
) -> Tuple[List[int], Dict[int, Tuple[int, int]]]:
    annotations = data["annotations"]
    track_ids_by_ann = [-1] * len(annotations)
    track_spans: Dict[int, List[int]] = {}

    if tracker_name == "bytetrack":
        tracker = ByteTrackWrapper(track_params)
    elif tracker_name == "iou":
        tracker = IOUTracker(iou_thresh=0.3, max_age=track_params.track_buffer)
    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")

    for frame_idx, image_id in enumerate(image_ids):
        ann_idxs = ann_by_image_id.get(image_id, [])
        if not ann_idxs:
            continue

        boxes = []
        scores = []
        for ann_idx in ann_idxs:
            ann = annotations[ann_idx]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            scores.append(float(ann.get("score", 1.0)))

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        keep = scores >= track_params.track_thresh
        if not np.any(keep):
            continue

        keep_indices = np.where(keep)[0]
        boxes_keep = boxes[keep]
        scores_keep = scores[keep]

        track_ids = tracker.update(boxes_keep, scores_keep)
        if len(track_ids) != len(boxes_keep):
            raise RuntimeError(
                f"Tracker returned {len(track_ids)} ids for {len(boxes_keep)} detections"
            )

        for local_i, track_id in zip(keep_indices, track_ids):
            ann_index = ann_idxs[int(local_i)]
            track_ids_by_ann[ann_index] = int(track_id)
            if track_id not in track_spans:
                track_spans[track_id] = [frame_idx, frame_idx]
            else:
                track_spans[track_id][1] = frame_idx

    track_spans = {tid: (span[0], span[1]) for tid, span in track_spans.items()}
    return track_ids_by_ann, track_spans


def associate_tracks(
    left_data: dict,
    right_data: dict,
    left_track_ids_by_ann: List[int],
    right_track_ids_by_ann: List[int],
    left_image_ids: List[int],
    right_image_ids: List[int],
    left_ann_by_image: Dict[int, List[int]],
    right_ann_by_image: Dict[int, List[int]],
    left_calib: Tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray],
    right_calib: Tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray],
    H: np.ndarray,
    dist_thresh_px: float,
) -> dict:
    (K_l, D_l, _, newK_l) = left_calib
    (K_r, D_r, _, newK_r) = right_calib

    evidence = defaultdict(lambda: {"count": 0, "distances": [], "frames": []})

    # Align frames by index (time-aligned videos)
    frame_count = min(len(left_image_ids), len(right_image_ids))

    for f in range(frame_count):
        left_image_id = left_image_ids[f]
        right_image_id = right_image_ids[f]

        left_ann_idxs = left_ann_by_image.get(left_image_id, [])
        right_ann_idxs = right_ann_by_image.get(right_image_id, [])

        if not left_ann_idxs or not right_ann_idxs:
            continue

        left_points = []
        left_track_ids = []
        for ann_idx in left_ann_idxs:
            tid = left_track_ids_by_ann[ann_idx]
            if tid < 0:
                continue
            x, y, w, h = left_data["annotations"][ann_idx]["bbox"]
            left_points.append([x + w / 2.0, y + h])
            left_track_ids.append(tid)

        right_points = []
        right_track_ids = []
        for ann_idx in right_ann_idxs:
            tid = right_track_ids_by_ann[ann_idx]
            if tid < 0:
                continue
            x, y, w, h = right_data["annotations"][ann_idx]["bbox"]
            right_points.append([x + w / 2.0, y + h])
            right_track_ids.append(tid)

        if not left_points or not right_points:
            continue

        left_points = undistort_points(np.array(left_points), K_l, D_l, newK_l)
        right_points = undistort_points(np.array(right_points), K_r, D_r, newK_r)
        right_points = project_homography(right_points, H)

        # Candidate pairs
        pairs = []
        for i, lpt in enumerate(left_points):
            for j, rpt in enumerate(right_points):
                dist = float(np.linalg.norm(lpt - rpt))
                if dist <= dist_thresh_px:
                    pairs.append((dist, left_track_ids[i], right_track_ids[j]))

        if not pairs:
            continue

        pairs.sort(key=lambda x: x[0])
        used_left = set()
        used_right = set()

        for dist, ltid, rtid in pairs:
            if ltid in used_left or rtid in used_right:
                continue
            used_left.add(ltid)
            used_right.add(rtid)
            key = (ltid, rtid)
            ev = evidence[key]
            ev["count"] += 1
            ev["distances"].append(dist)
            if len(ev["frames"]) < 10:
                ev["frames"].append(f)

    return {
        "evidence": evidence,
        "frame_count": frame_count,
    }


def resolve_global_ids(
    left_track_spans: Dict[int, Tuple[int, int]],
    right_track_spans: Dict[int, Tuple[int, int]],
    evidence: dict,
    dist_thresh_px: float,
    min_matches: int,
    mode: str = "mutual_best",
):
    """
    Resolve global IDs across cameras.
    mode:
      - "union": DSU union of all qualifying pairs (can merge many-to-one)
      - "mutual_best": one-to-one via mutual best match (recommended)
    """
    # Build candidate pair list
    candidates = []
    candidate_map = {}
    for (ltid, rtid), ev in evidence.items():
        if ev["count"] < min_matches:
            continue
        median_dist = float(np.median(ev["distances"])) if ev["distances"] else float("inf")
        if median_dist <= dist_thresh_px:
            score = ev["count"] / (median_dist + 1.0)
            candidates.append((ltid, rtid, ev["count"], median_dist, score))
            candidate_map[(ltid, rtid)] = (ev["count"], median_dist, score)

    if mode == "union":
        dsu = DSU()

        # Register all track nodes
        for tid in left_track_spans.keys():
            dsu.find(("left", tid))
        for tid in right_track_spans.keys():
            dsu.find(("right", tid))

        unions = []
        for ltid, rtid, cnt, med, _score in candidates:
            dsu.union(("left", ltid), ("right", rtid))
            unions.append((ltid, rtid, cnt, med))

        # Assign global IDs
        root_to_gid = {}
        next_gid = 1
        track_to_global = {}

        for key in list(dsu.parent.keys()):
            root = dsu.find(key)
            if root not in root_to_gid:
                root_to_gid[root] = next_gid
                next_gid += 1
            track_to_global[key] = root_to_gid[root]

        return track_to_global, unions

    if mode != "mutual_best":
        raise ValueError(f"Unknown association mode: {mode}")

    # Mutual-best one-to-one matching
    best_right_for_left: Dict[int, Tuple[int, float, float]] = {}
    best_left_for_right: Dict[int, Tuple[int, float, float]] = {}

    for ltid, rtid, cnt, med, score in candidates:
        prev = best_right_for_left.get(ltid)
        if prev is None or score > prev[1]:
            best_right_for_left[ltid] = (rtid, score, med)

        prev = best_left_for_right.get(rtid)
        if prev is None or score > prev[1]:
            best_left_for_right[rtid] = (ltid, score, med)

    matches = []
    for ltid, (rtid, score, med) in best_right_for_left.items():
        back = best_left_for_right.get(rtid)
        if back and back[0] == ltid:
            matches.append((ltid, rtid, score, med))

    # Assign global IDs
    track_to_global = {}
    next_gid = 1

    matched_left = set()
    matched_right = set()
    for ltid, rtid, _score, _med in matches:
        track_to_global[("left", ltid)] = next_gid
        track_to_global[("right", rtid)] = next_gid
        matched_left.add(ltid)
        matched_right.add(rtid)
        next_gid += 1

    # Unmatched tracks get unique IDs
    for tid in left_track_spans.keys():
        if tid not in matched_left:
            track_to_global[("left", tid)] = next_gid
            next_gid += 1
    for tid in right_track_spans.keys():
        if tid not in matched_right:
            track_to_global[("right", tid)] = next_gid
            next_gid += 1

    # Convert matches for reporting with real counts
    match_report = []
    for ltid, rtid, score, med in matches:
        cnt, med2, _ = candidate_map.get((ltid, rtid), (0, med, score))
        match_report.append((ltid, rtid, cnt, med2))

    return track_to_global, match_report


def add_ids_to_annotations(data: dict, track_ids_by_ann: List[int], track_to_global: Dict[Tuple[str, int], int], camera_name: str):
    annotations = data["annotations"]
    for i, ann in enumerate(annotations):
        tid = int(track_ids_by_ann[i]) if i < len(track_ids_by_ann) else -1
        ann["track_id"] = tid
        if tid >= 0:
            ann["global_id"] = int(track_to_global.get((camera_name, tid), -1))
        else:
            ann["global_id"] = -1


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Offline cross-camera association (ByteTrack + homography)")
    parser.add_argument("--cam0-json", default="Association/inputs/cam0_2026-01-19_18-59-13.json")
    parser.add_argument("--cam1-json", default="Association/inputs/cam1_2026-01-19_18-59-13.json")
    parser.add_argument("--out-dir", default="Association/outputs")
    parser.add_argument("--homography", default="Association/calib/H_right_to_left.npy")
    parser.add_argument("--left-calib", default="Association/calib/left_fisheye_calib.npz")
    parser.add_argument("--right-calib", default="Association/calib/right_fisheye_calib.npz")
    parser.add_argument("--dist-thresh", type=float, default=80.0)
    parser.add_argument("--min-matches", type=int, default=3)
    parser.add_argument("--track-thresh", type=float, default=0.5)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cam0-video", default="Association/inputs/cam0_2026-01-19_18-59-13.mp4")
    parser.add_argument("--cam1-video", default="Association/inputs/cam1_2026-01-19_18-59-13.mp4")
    parser.add_argument("--tracker", choices=["bytetrack", "iou"], default="bytetrack")
    parser.add_argument("--assoc-mode", choices=["mutual_best", "union"], default="mutual_best")

    args = parser.parse_args()

    # Load JSON
    with open(args.cam0_json, "r") as f:
        left_data = json.load(f)
    with open(args.cam1_json, "r") as f:
        right_data = json.load(f)

    # Build frame indices
    left_image_ids, left_ann_by_image, left_image_sizes = build_frame_index(left_data)
    right_image_ids, right_ann_by_image, right_image_sizes = build_frame_index(right_data)

    if len(left_image_ids) != len(right_image_ids):
        # Fallback to filename-based alignment
        left_image_ids, left_ann_by_image, left_image_sizes = build_frame_index_by_filename(left_data)
        right_image_ids, right_ann_by_image, right_image_sizes = build_frame_index_by_filename(right_data)

    # Determine fallback dims from JSON images
    left_dim = left_image_sizes.get(left_image_ids[0], (1920, 1080))
    right_dim = right_image_sizes.get(right_image_ids[0], (1920, 1080))

    # Load calibration + homography
    K_l, D_l, dim_l, newK_l = load_fisheye_calib(args.left_calib, fallback_dim=left_dim)
    K_r, D_r, dim_r, newK_r = load_fisheye_calib(args.right_calib, fallback_dim=right_dim)
    H = np.load(args.homography)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {H.shape}")

    # Determine fps
    fps = args.fps
    fps0 = get_video_fps(args.cam0_video)
    fps1 = get_video_fps(args.cam1_video)
    if fps0:
        fps = int(round(fps0))
    elif fps1:
        fps = int(round(fps1))

    track_params = TrackParams(
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
        frame_rate=fps,
    )

    # Run per-camera tracking
    print("Running per-camera tracking...")
    left_track_ids_by_ann, left_track_spans = run_tracking(
        left_data, left_image_ids, left_ann_by_image, track_params, args.tracker
    )
    right_track_ids_by_ann, right_track_spans = run_tracking(
        right_data, right_image_ids, right_ann_by_image, track_params, args.tracker
    )

    # Cross-camera association
    print("Associating tracks across cameras...")
    assoc = associate_tracks(
        left_data,
        right_data,
        left_track_ids_by_ann,
        right_track_ids_by_ann,
        left_image_ids,
        right_image_ids,
        left_ann_by_image,
        right_ann_by_image,
        (K_l, D_l, dim_l, newK_l),
        (K_r, D_r, dim_r, newK_r),
        H,
        args.dist_thresh,
    )

    evidence = assoc["evidence"]
    frame_count = assoc["frame_count"]

    track_to_global, unions = resolve_global_ids(
        left_track_spans,
        right_track_spans,
        evidence,
        args.dist_thresh,
        args.min_matches,
        mode=args.assoc_mode,
    )

    # Add IDs to annotations
    add_ids_to_annotations(left_data, left_track_ids_by_ann, track_to_global, "left")
    add_ids_to_annotations(right_data, right_track_ids_by_ann, track_to_global, "right")

    # Add metadata
    info_meta = {
        "homography_path": args.homography,
        "dist_thresh_px": args.dist_thresh,
        "min_matches": args.min_matches,
        "undistort": True,
        "association_mode": args.assoc_mode,
        "track_params": {
            "track_thresh": args.track_thresh,
            "match_thresh": args.match_thresh,
            "track_buffer": args.track_buffer,
            "frame_rate": fps,
            "tracker": args.tracker,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    left_data.setdefault("info", {})
    right_data.setdefault("info", {})
    left_data["info"]["association"] = info_meta
    right_data["info"]["association"] = info_meta

    # Report
    report = {
        "frames_processed": frame_count,
        "left_tracks": len(left_track_spans),
        "right_tracks": len(right_track_spans),
        "global_ids": len(set(track_to_global.values())),
        "cross_camera_unions": len(unions),
        "union_samples": [
            {
                "left_track_id": l,
                "right_track_id": r,
                "match_count": c,
                "median_distance": md,
            }
            for (l, r, c, md) in unions[:50]
        ],
    }

    # Write outputs
    ensure_dir(args.out_dir)
    left_out = Path(args.out_dir) / (Path(args.cam0_json).stem + "_global.json")
    right_out = Path(args.out_dir) / (Path(args.cam1_json).stem + "_global.json")
    report_out = Path(args.out_dir) / "association_report.json"

    with open(left_out, "w") as f:
        json.dump(left_data, f)
    with open(right_out, "w") as f:
        json.dump(right_data, f)
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    print("Done.")
    print(f"Left output: {left_out}")
    print(f"Right output: {right_out}")
    print(f"Report: {report_out}")


if __name__ == "__main__":
    main()
