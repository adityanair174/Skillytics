import os
import supervision as sv
from rfdetr import RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES
import numpy as np
from pathlib import Path
import cv2


# ip_path = Path("videos/Game-1/cam0_2025-11-14_19-48-45.mp4")
ip_path = Path("videos/stitched/stitched_night.mp4")

model = RFDETRLarge()

def callback(frame, index):
    detections = model.predict(frame[:, :, ::-1].copy(), threshold=0.5)
    detections = detections[np.isin(detections.class_id, (1, 37))]
    # filter detections for person and sports ball only
    labels = [f"{COCO_CLASSES[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    #  if class_id in (1, 37)
    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
    return annotated_frame


op_path = "output_videos"/ip_path
op_path.parent.mkdir(parents=True, exist_ok=True)

sv.process_video(
    source_path=ip_path,
    target_path=op_path,
    callback=callback,
    max_frames=None,
    show_progress=True
)




# Ref: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb#scrollTo=rQ2PGjhHp8I7

# from sam2.build_sam import build_sam2_camera_predictor
# SAM2_HOME = Path("../segment-anything-2-real-time")
# SAM2_CHECKPOINT = SAM2_HOME / "checkpoints/sam2.1_hiera_tiny.pt"
# SAM2_CONFIG = SAM2_HOME / "configs/sam2.1/sam2.1_hiera_t.yaml"

# predictor = build_sam2_camera_predictor(SAM2_CONFIG.resolve(), SAM2_CHECKPOINT.resolve())

