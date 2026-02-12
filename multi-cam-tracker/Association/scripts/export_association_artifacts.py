#!/usr/bin/env python3
"""
Export association artifacts:
- Mapping table (global_id -> left/right track_id)
- Overlay video with global IDs drawn on both cameras
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

FRAME_RE = re.compile(r"_frame_(\d+)")


def parse_frame_number(file_name: str) -> int | None:
    m = FRAME_RE.search(file_name)
    if not m:
        return None
    return int(m.group(1))


def build_frame_index(data: dict) -> Tuple[List[int], Dict[int, List[int]]]:
    images = data["images"]
    annotations = data["annotations"]

    ann_by_image_id: Dict[int, List[int]] = defaultdict(list)
    for idx, ann in enumerate(annotations):
        ann_by_image_id[int(ann["image_id"])] .append(idx)

    image_ids = sorted([int(img["id"]) for img in images])
    return image_ids, ann_by_image_id


def build_frame_index_by_filename(data: dict) -> Tuple[List[int], Dict[int, List[int]]]:
    images = data["images"]
    annotations = data["annotations"]

    frame_to_image_id = {}
    for img in images:
        frame_num = parse_frame_number(img["file_name"])
        if frame_num is not None:
            frame_to_image_id[frame_num] = int(img["id"])

    ann_by_image_id: Dict[int, List[int]] = defaultdict(list)
    for idx, ann in enumerate(annotations):
        ann_by_image_id[int(ann["image_id"])] .append(idx)

    frame_numbers = sorted(frame_to_image_id.keys())
    image_ids = [frame_to_image_id[f] for f in frame_numbers]
    return image_ids, ann_by_image_id


def color_for_id(gid: int) -> Tuple[int, int, int]:
    # Deterministic color from id
    np.random.seed(gid % 2**16)
    color = np.random.randint(0, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def draw_detections(frame, annotations, ann_indices):
    for ann_idx in ann_indices:
        ann = annotations[ann_idx]
        x, y, w, h = ann["bbox"]
        gid = int(ann.get("global_id", -1))
        if gid < 0:
            continue
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        color = color_for_id(gid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"G:{gid}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def build_mapping(left_data: dict, right_data: dict):
    # Count track occurrences per global id per camera
    left_counts = defaultdict(lambda: defaultdict(int))
    right_counts = defaultdict(lambda: defaultdict(int))

    for ann in left_data["annotations"]:
        gid = int(ann.get("global_id", -1))
        tid = int(ann.get("track_id", -1))
        if gid >= 0 and tid >= 0:
            left_counts[gid][tid] += 1

    for ann in right_data["annotations"]:
        gid = int(ann.get("global_id", -1))
        tid = int(ann.get("track_id", -1))
        if gid >= 0 and tid >= 0:
            right_counts[gid][tid] += 1

    mapping = {}
    all_gids = sorted(set(left_counts.keys()) | set(right_counts.keys()))
    for gid in all_gids:
        left_best = None
        right_best = None
        if gid in left_counts and left_counts[gid]:
            left_best = max(left_counts[gid].items(), key=lambda x: x[1])
        if gid in right_counts and right_counts[gid]:
            right_best = max(right_counts[gid].items(), key=lambda x: x[1])

        mapping[gid] = {
            "left_track_id": left_best[0] if left_best else None,
            "left_count": left_best[1] if left_best else 0,
            "right_track_id": right_best[0] if right_best else None,
            "right_count": right_best[1] if right_best else 0,
        }

    return mapping


def write_mapping(mapping: dict, out_json: str, out_csv: str):
    with open(out_json, "w") as f:
        json.dump(mapping, f, indent=2)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["global_id", "left_track_id", "left_count", "right_track_id", "right_count"])
        for gid in sorted(mapping.keys()):
            row = mapping[gid]
            writer.writerow([gid, row["left_track_id"], row["left_count"], row["right_track_id"], row["right_count"]])


def make_overlay_video(
    left_video: str,
    right_video: str,
    left_data: dict,
    right_data: dict,
    left_image_ids: List[int],
    right_image_ids: List[int],
    left_ann_by_image: Dict[int, List[int]],
    right_ann_by_image: Dict[int, List[int]],
    out_path: str,
    scale: float = 0.6,
    max_frames: int | None = None,
):
    cap_left = cv2.VideoCapture(left_video)
    cap_right = cv2.VideoCapture(right_video)

    if not cap_left.isOpened() or not cap_right.isOpened():
        raise RuntimeError("Failed to open one of the videos")

    fps = cap_left.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = cap_right.get(cv2.CAP_PROP_FPS) or 30

    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()
    if not ret_l or not ret_r:
        raise RuntimeError("Failed to read first frame from one of the videos")

    # Prepare writer
    h, w = frame_l.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    out_w = new_w * 2
    out_h = new_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if out_path.lower().endswith(".mp4"):
        tmp_out = out_path[:-4] + ".tmp.mp4"
    else:
        tmp_out = out_path + ".tmp.mp4"
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {tmp_out}")

    # Reset to first frame
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_frames = min(len(left_image_ids), len(right_image_ids))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    for i in range(total_frames):
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        if not ret_l or not ret_r:
            break

        left_image_id = left_image_ids[i]
        right_image_id = right_image_ids[i]

        left_ann_idxs = left_ann_by_image.get(left_image_id, [])
        right_ann_idxs = right_ann_by_image.get(right_image_id, [])

        draw_detections(frame_l, left_data["annotations"], left_ann_idxs)
        draw_detections(frame_r, right_data["annotations"], right_ann_idxs)

        # Labels
        cv2.putText(frame_l, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame_r, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Resize and combine
        frame_l = cv2.resize(frame_l, (new_w, new_h))
        frame_r = cv2.resize(frame_r, (new_w, new_h))
        combined = np.hstack([frame_l, frame_r])
        writer.write(combined)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{total_frames} frames")

    cap_left.release()
    cap_right.release()
    writer.release()
    os.replace(tmp_out, out_path)


def main():
    parser = argparse.ArgumentParser(description="Export association map + overlay video")
    parser.add_argument("--left-json", default="Association/outputs/cam0_2026-01-19_18-59-13_global.json")
    parser.add_argument("--right-json", default="Association/outputs/cam1_2026-01-19_18-59-13_global.json")
    parser.add_argument("--left-video", default="Association/inputs/cam0_2026-01-19_18-59-13.mp4")
    parser.add_argument("--right-video", default="Association/inputs/cam1_2026-01-19_18-59-13.mp4")
    parser.add_argument("--out-dir", default="Association/outputs")
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")

    args = parser.parse_args()

    with open(args.left_json, "r") as f:
        left_data = json.load(f)
    with open(args.right_json, "r") as f:
        right_data = json.load(f)

    # Build frame indices
    left_image_ids, left_ann_by_image = build_frame_index(left_data)
    right_image_ids, right_ann_by_image = build_frame_index(right_data)
    if len(left_image_ids) != len(right_image_ids):
        left_image_ids, left_ann_by_image = build_frame_index_by_filename(left_data)
        right_image_ids, right_ann_by_image = build_frame_index_by_filename(right_data)

    os.makedirs(args.out_dir, exist_ok=True)

    # Mapping table
    mapping = build_mapping(left_data, right_data)
    map_json = str(Path(args.out_dir) / "association_map.json")
    map_csv = str(Path(args.out_dir) / "association_map.csv")
    write_mapping(mapping, map_json, map_csv)
    print(f"Wrote mapping: {map_json}")
    print(f"Wrote mapping: {map_csv}")

    # Overlay video
    out_video = str(Path(args.out_dir) / "association_overlay.mp4")
    max_frames = None if args.max_frames == 0 else args.max_frames
    make_overlay_video(
        args.left_video,
        args.right_video,
        left_data,
        right_data,
        left_image_ids,
        right_image_ids,
        left_ann_by_image,
        right_ann_by_image,
        out_video,
        scale=args.scale,
        max_frames=max_frames,
    )
    print(f"Wrote overlay: {out_video}")


if __name__ == "__main__":
    main()
