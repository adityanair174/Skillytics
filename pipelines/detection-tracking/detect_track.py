import os
import supervision as sv
from rfdetr import RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES
import numpy as np
from pathlib import Path
import cv2
from trackers import ByteTrackTracker
import copy
import json

# Flag to enable/disable displaying annotated frame
SHOW_ANNOTATED_FRAME = False

# Initialize tracker
tracker = ByteTrackTracker(track_activation_threshold=0.5, high_conf_det_threshold=0.5, lost_track_buffer=90)

# ip_path = Path("videos/Game-1/cam0_2025-11-14_19-48-45.mp4")
ip_path = Path("datasets/Tournament-ProKick-26th Jan-2026/Game-1/cam1_2026-01-19_18-59-13.mp4")

model = RFDETRLarge()
# model.optimize_for_inference()

# Initialize annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# Flag to control video processing
stop_processing = False

all_frame_detections = []
all_frame_info = []  # Store frame info (index, height, width)

def callback(frame, index):
    global stop_processing
    global all_frame_detections
    global all_frame_info
    # Get detections from model
    detections = model.predict(frame[:, :, ::-1].copy(), threshold=0.5)
    # Filter detections for person (class_id=1) and sports ball (class_id=37) only
    detections = detections[np.isin(detections.class_id, (1, 37))]
    all_frame_detections.append(copy.deepcopy(detections))
    # Store frame dimensions
    height, width = frame.shape[:2]
    all_frame_info.append({"index": index, "height": height, "width": width})
    # return frame

    # Update tracker with detections to get tracking IDs
    detections = tracker.update(detections)
    
    # Create labels with class name, confidence, and tracking ID
    labels = []
    for i, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
        class_name = COCO_CLASSES[class_id]
        tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else None
        if tracker_id is not None:
            labels.append(f"#{tracker_id} {class_name} {confidence:.2f}")
        else:
            labels.append(f"{class_name} {confidence:.2f}")
    
    # Annotate frame with boxes, labels, and traces
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(annotated_frame, detections)
    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
    
    # Display annotated frame using OpenCV (if enabled)
    if SHOW_ANNOTATED_FRAME:
        cv2.imshow("Tracking", annotated_frame)
        # Press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_processing = True
    
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

def convert_to_coco_format(frame_detections, frame_info_list, video_filename):
    """
    Convert supervision Detections to COCO format.
    
    Args:
        frame_detections: List of sv.Detections objects, one per frame
        frame_info_list: List of dicts with frame info (index, height, width)
        video_filename: Name of the video file
    
    Returns:
        dict: COCO format dictionary
    """
    coco_data = {
        "info": {
            "description": "Roboflow detections converted to COCO format",
            "version": "1.0"
        },
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Create categories from COCO_CLASSES (only for classes we're detecting: 1=person, 37=sports ball)
    category_map = {1: 1, 37: 2}  # Map COCO class_id to sequential category_id
    coco_data["categories"] = [
        {"id": 1, "name": "person", "supercategory": "person"},
        {"id": 2, "name": "sports ball", "supercategory": "sports"}
    ]
    
    annotation_id = 1
    
    for frame_idx, (detections, frame_info) in enumerate(zip(frame_detections, frame_info_list)):
        # Add image entry
        image_id = frame_idx + 1
        coco_data["images"].append({
            "id": image_id,
            "width": frame_info["width"],
            "height": frame_info["height"],
            "file_name": f"{video_filename}_frame_{frame_idx:06d}.jpg"  # COCO typically uses image files
        })
        
        # Convert detections to annotations
        if len(detections) > 0:
            for i in range(len(detections)):
                # Convert xyxy to xywh (COCO format)
                x1, y1, x2, y2 = detections.xyxy[i]
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Get category_id (map COCO class_id to sequential category_id)
                coco_class_id = int(detections.class_id[i])
                category_id = category_map.get(coco_class_id, coco_class_id)
                
                # Create annotation
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(x1), float(y1), float(width), float(height)],  # [x, y, width, height]
                    "area": float(area),
                    "iscrowd": 0,
                    "score": float(detections.confidence[i])  # COCO doesn't have score, but useful to keep
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
    
    return coco_data

# Convert to COCO format and save
json_path = ip_path.with_suffix(".json")
coco_data = convert_to_coco_format(all_frame_detections, all_frame_info, ip_path.stem)
with open(json_path, "w") as f:
    json.dump(coco_data, f, indent=2)

# Clean up OpenCV windows (if display was enabled)
if SHOW_ANNOTATED_FRAME:
    cv2.destroyAllWindows()




# Ref: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb#scrollTo=rQ2PGjhHp8I7

# from sam2.build_sam import build_sam2_camera_predictor
# SAM2_HOME = Path("../segment-anything-2-real-time")
# SAM2_CHECKPOINT = SAM2_HOME / "checkpoints/sam2.1_hiera_tiny.pt"
# SAM2_CONFIG = SAM2_HOME / "configs/sam2.1/sam2.1_hiera_t.yaml"

# predictor = build_sam2_camera_predictor(SAM2_CONFIG.resolve(), SAM2_CHECKPOINT.resolve())

